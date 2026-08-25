import json
import httpx
import openai
import pytest
from unittest.mock import MagicMock, patch

import app.llm as llm
from app.models import Author, Paper
from app.pipeline._json import LLMOutputError
from app.pipeline.rerank import rerank_candidates


@pytest.fixture(autouse=True)
def _reset_llm_cooldowns():
    """app.llm parks rate-limited models in a module-global registry. Without a
    reset, a test that exercises a 429 leaves its models cooling and the next
    test silently takes a different failover path."""
    llm._cooldowns.clear()
    yield
    llm._cooldowns.clear()


def _rate_limit_error() -> openai.RateLimitError:
    request = httpx.Request("POST", "https://example.com")
    response = httpx.Response(429, request=request)
    return openai.RateLimitError("rate limited", response=response, body=None)


def _make_paper(arxiv_id: str, score: float | None = None) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        abstract="Some abstract.",
        authors=[Author(name="Alice")],
        categories=["cs.LG"],
        published="2024-01-01",
        url=f"https://arxiv.org/abs/{arxiv_id}",
        relevance_score=score,
    )


def _fake_llm_response(scores: list[dict], raw: str | None = None) -> MagicMock:
    message = MagicMock()
    # JSON mode requires an object at the root, so scores are wrapped in "items".
    message.content = raw if raw is not None else json.dumps({"items": scores})
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


@patch("app.llm._get_client")
def test_rerank_sorts_by_score(mock_get_client):
    candidates = [_make_paper("2401.0001"), _make_paper("2401.0002"), _make_paper("2401.0003")]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _fake_llm_response([
        {"arxiv_id": "2401.0001", "score": 0.4, "rationale": "Somewhat relevant."},
        {"arxiv_id": "2401.0002", "score": 0.9, "rationale": "Highly relevant."},
        {"arxiv_id": "2401.0003", "score": 0.1, "rationale": "Barely relevant."},
    ])
    mock_get_client.return_value = mock_client

    ranked, elapsed_ms, _, _ = rerank_candidates("test topic", candidates)

    assert ranked[0].arxiv_id == "2401.0002"
    assert ranked[1].arxiv_id == "2401.0001"
    assert ranked[2].arxiv_id == "2401.0003"
    assert all(p.relevance_score is not None for p in ranked)
    assert elapsed_ms >= 0


@patch("app.llm._get_client")
def test_rerank_respects_max_retained(mock_get_client):
    candidates = [_make_paper(f"2401.{i:04d}") for i in range(5)]
    scores = [
        {"arxiv_id": f"2401.{i:04d}", "score": float(i) / 10, "rationale": "ok"}
        for i in range(5)
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _fake_llm_response(scores)
    mock_get_client.return_value = mock_client

    # max_retained_papers defaults to 18, but we only have 5 candidates
    ranked, _, _, _ = rerank_candidates("test topic", candidates)
    assert len(ranked) <= 5


@patch("app.llm._get_client")
def test_rerank_handles_markdown_fenced_json(mock_get_client):
    """Regression: fenced JSON used to raise straight through as a hard 502."""
    payload = {"items": [{"arxiv_id": "2401.0001", "score": 0.7, "rationale": "ok"}]}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _fake_llm_response(
        [], raw="```json\n" + json.dumps(payload) + "\n```"
    )
    mock_get_client.return_value = mock_client

    ranked, _, _, _ = rerank_candidates("test topic", [_make_paper("2401.0001")])

    assert ranked[0].relevance_score == 0.7


@patch("app.llm._get_client")
def test_rerank_rejects_out_of_range_score(mock_get_client):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _fake_llm_response(
        [{"arxiv_id": "2401.0001", "score": 4.2, "rationale": "nonsense"}]
    )
    mock_get_client.return_value = mock_client

    with pytest.raises(LLMOutputError):
        rerank_candidates("test topic", [_make_paper("2401.0001")])


@patch("app.llm._get_client")
def test_rerank_tolerates_unscored_candidate(mock_get_client):
    """A paper the model skipped keeps score None and sorts last, but is logged."""
    candidates = [_make_paper("2401.0001"), _make_paper("2401.0002")]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _fake_llm_response(
        [{"arxiv_id": "2401.0001", "score": 0.5, "rationale": "ok"}]
    )
    mock_get_client.return_value = mock_client

    ranked, _, _, _ = rerank_candidates("test topic", candidates)

    assert ranked[0].arxiv_id == "2401.0001"
    assert ranked[-1].arxiv_id == "2401.0002"
    assert ranked[-1].relevance_score is None


@patch("app.llm._get_client")
def test_rerank_batches_candidates_above_batch_size(mock_get_client):
    """22 candidates at a batch size of 15 must issue 2 calls, not 1 giant one."""
    candidates = [_make_paper(f"2401.{i:04d}") for i in range(22)]
    mock_client = MagicMock()

    def side_effect(*, model, max_tokens, messages, response_format=None, **kwargs):
        user = messages[-1]["content"]
        ids = [c.arxiv_id for c in candidates if c.arxiv_id in user]
        return _fake_llm_response(
            [{"arxiv_id": i, "score": 0.5, "rationale": "ok"} for i in ids]
        )

    mock_client.chat.completions.create.side_effect = side_effect
    mock_get_client.return_value = mock_client

    ranked, _, _, _ = rerank_candidates("test topic", candidates)

    assert mock_client.chat.completions.create.call_count == 2  # ceil(22/15)
    assert all(p.relevance_score == 0.5 for p in ranked)


@patch("app.llm._get_client")
def test_rerank_one_failed_batch_does_not_sink_the_others(mock_get_client):
    """A batch that errors out leaves its papers unscored; other batches still score."""
    candidates = [_make_paper(f"2401.{i:04d}") for i in range(20)]

    def side_effect(*, model, max_tokens, messages, response_format=None, **kwargs):
        user = messages[-1]["content"]
        if "2401.0000" in user:  # first batch — force a schema failure
            return _fake_llm_response([{"arxiv_id": "2401.0000", "score": 9.9, "rationale": "x"}])
        ids = [c.arxiv_id for c in candidates if c.arxiv_id in user]
        return _fake_llm_response(
            [{"arxiv_id": i, "score": 0.7, "rationale": "ok"} for i in ids]
        )

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = side_effect
    mock_get_client.return_value = mock_client

    # Inspect `candidates` directly, not the returned `ranked` slice: with 20
    # candidates and the default max_retained_papers=18, the slice would drop 2
    # unscored papers and make the counts below look wrong for the wrong reason.
    rerank_candidates("test topic", candidates)

    scored = [p for p in candidates if p.relevance_score is not None]
    unscored = [p for p in candidates if p.relevance_score is None]
    assert len(scored) == 5  # second batch (indices 15-19) scored fine
    assert len(unscored) == 15  # first batch (indices 0-14) all failed together


@patch("app.llm._get_client")
def test_rerank_all_batches_failing_raises(mock_get_client):
    """Total rerank failure must still surface, not silently succeed unranked."""
    candidates = [_make_paper(f"2401.{i:04d}") for i in range(3)]
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("provider down")
    mock_get_client.return_value = mock_client

    with pytest.raises(LLMOutputError):
        rerank_candidates("test topic", candidates)


@patch("app.llm._get_client")
def test_rerank_fails_over_within_the_pool_on_rate_limit(mock_get_client):
    """A rate-limited primary must not sink the batch — the pool descends.

    Failover is OpenRouter-side: the request carries a models[] array, so the
    single call comes back served by a different model. app.llm notices that and
    cools the primary; rerank itself just sees a normal response.
    """
    response = _fake_llm_response([{"arxiv_id": "2401.0001", "score": 0.5, "rationale": "ok"}])
    response.model = "openai/gpt-oss-120b:free"  # not position 0 of the pool
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = response
    mock_get_client.return_value = mock_client

    ranked, _, _, _ = rerank_candidates("test topic", [_make_paper("2401.0001")])

    assert ranked[0].relevance_score == 0.5
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    # The whole pool prefix travels with the request rather than being retried.
    assert len(kwargs["extra_body"]["models"]) > 1
    assert kwargs["model"] == kwargs["extra_body"]["models"][0]
    # The primary that got skipped is now parked so the next call goes straight
    # to a model we have not seen fail.
    assert llm._is_cooling(kwargs["model"])


@patch("app.llm._get_client")
def test_rerank_raises_when_every_model_is_rate_limited(mock_get_client):
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _rate_limit_error()
    mock_get_client.return_value = mock_client

    with pytest.raises(LLMOutputError):
        rerank_candidates("test topic", [_make_paper("2401.0001")])


@patch("app.llm._get_client")
def test_rerank_reports_token_usage(mock_get_client):
    response = _fake_llm_response([{"arxiv_id": "2401.0001", "score": 0.5, "rationale": "ok"}])
    response.usage.prompt_tokens = 1500
    response.usage.completion_tokens = 300
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = response
    mock_get_client.return_value = mock_client

    _, _, prompt_tokens, completion_tokens = rerank_candidates(
        "test topic", [_make_paper("2401.0001")]
    )

    assert prompt_tokens == 1500
    assert completion_tokens == 300
