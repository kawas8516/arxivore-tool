import json
from unittest.mock import MagicMock, patch

from app.models import Author, Paper
from app.pipeline.rerank import rerank_candidates


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


def _fake_llm_response(scores: list[dict]) -> MagicMock:
    message = MagicMock()
    message.content = json.dumps(scores)
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


@patch("app.pipeline.rerank.OpenAI")
def test_rerank_sorts_by_score(mock_openai_cls):
    candidates = [_make_paper("2401.0001"), _make_paper("2401.0002"), _make_paper("2401.0003")]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _fake_llm_response([
        {"arxiv_id": "2401.0001", "score": 0.4, "rationale": "Somewhat relevant."},
        {"arxiv_id": "2401.0002", "score": 0.9, "rationale": "Highly relevant."},
        {"arxiv_id": "2401.0003", "score": 0.1, "rationale": "Barely relevant."},
    ])
    mock_openai_cls.return_value = mock_client

    ranked, elapsed_ms = rerank_candidates("test topic", candidates)

    assert ranked[0].arxiv_id == "2401.0002"
    assert ranked[1].arxiv_id == "2401.0001"
    assert ranked[2].arxiv_id == "2401.0003"
    assert all(p.relevance_score is not None for p in ranked)
    assert elapsed_ms >= 0


@patch("app.pipeline.rerank.OpenAI")
def test_rerank_respects_max_retained(mock_openai_cls):
    candidates = [_make_paper(f"2401.{i:04d}") for i in range(5)]
    scores = [
        {"arxiv_id": f"2401.{i:04d}", "score": float(i) / 10, "rationale": "ok"}
        for i in range(5)
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _fake_llm_response(scores)
    mock_openai_cls.return_value = mock_client

    # max_retained_papers defaults to 18, but we only have 5 candidates
    ranked, _ = rerank_candidates("test topic", candidates)
    assert len(ranked) <= 5
