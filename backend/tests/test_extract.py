import json
import httpx
import openai
import pytest
from unittest.mock import MagicMock, patch

from app import storage
from app.models import Author, Paper
from app.pipeline.extract import extract_papers
from app.pipeline._json import strip_fences


def _rate_limit_error() -> openai.RateLimitError:
    request = httpx.Request("POST", "https://example.com")
    response = httpx.Response(429, request=request)
    return openai.RateLimitError("rate limited", response=response, body=None)


@pytest.fixture(autouse=True)
def _no_real_storage(monkeypatch):
    """Every test in this file is about the LLM path, not persistence.

    Without this, extract_papers would hit a real SQLite file per call. Tests
    that specifically exercise caching override these within their own body.
    """
    monkeypatch.setattr(storage, "get_cached_extraction", lambda arxiv_id: None)
    monkeypatch.setattr(storage, "save_extraction", lambda paper: None)


def _make_paper(arxiv_id: str) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        abstract="We propose a new method for retrieval-augmented generation.",
        authors=[Author(name="Alice")],
        categories=["cs.LG"],
        published="2024-01-01",
        url=f"https://arxiv.org/abs/{arxiv_id}",
        relevance_score=0.9,
    )


def _fake_response(data: dict) -> MagicMock:
    message = MagicMock()
    message.content = json.dumps(data)
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


_EXTRACTION = {
    "problem": "RAG systems struggle with retrieval quality.",
    "method": "We use a dense retriever with cross-attention reranking.",
    "results": "Achieves 5% improvement on NaturalQuestions benchmark.",
    "contribution": "A novel cross-attention reranking module for RAG pipelines.",
}


@patch("app.pipeline.extract.OpenAI")
def test_extract_maps_fields(mock_openai_cls):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _fake_response(_EXTRACTION)
    mock_openai_cls.return_value = mock_client

    papers = [_make_paper("2401.0001")]
    result, elapsed_ms, errors, _, _ = extract_papers(papers)

    assert errors == 0
    assert elapsed_ms >= 0
    p = result[0]
    assert p.extract_status == "done"
    assert p.problem == _EXTRACTION["problem"]
    assert p.method == _EXTRACTION["method"]
    assert p.results == _EXTRACTION["results"]
    assert p.contribution == _EXTRACTION["contribution"]


@patch("app.pipeline.extract.OpenAI")
def test_extract_single_failure_does_not_fail_batch(mock_openai_cls):
    mock_client = MagicMock()

    def side_effect(**kwargs):
        # Fail on the second paper, succeed on others
        content = kwargs.get("messages", [{}])[-1].get("content", "")
        if "2401.0002" in content:
            raise RuntimeError("LLM timeout")
        return _fake_response(_EXTRACTION)

    mock_client.chat.completions.create.side_effect = side_effect
    mock_openai_cls.return_value = mock_client

    papers = [_make_paper("2401.0001"), _make_paper("2401.0002"), _make_paper("2401.0003")]
    result, _, errors, _, _ = extract_papers(papers)

    assert errors == 1
    statuses = {p.arxiv_id: p.extract_status for p in result}
    assert statuses["2401.0001"] == "done"
    assert statuses["2401.0002"] == "error"
    assert statuses["2401.0003"] == "done"


def test_strip_fences_removes_markdown():
    raw = "```json\n{\"key\": \"value\"}\n```"
    assert strip_fences(raw) == '{"key": "value"}'


def test_strip_fences_passthrough_plain_json():
    raw = '{"key": "value"}'
    assert strip_fences(raw) == raw


@patch("app.pipeline.extract.OpenAI")
def test_extract_rejects_incomplete_extraction(mock_openai_cls):
    """A response missing a field must error, not write empty strings as 'done'."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _fake_response(
        {"problem": "A problem.", "method": "A method."}  # no results/contribution
    )
    mock_openai_cls.return_value = mock_client

    result, _, errors, _, _ = extract_papers([_make_paper("2401.0001")])

    assert errors == 1
    assert result[0].extract_status == "error"
    assert result[0].problem is None


@patch("app.pipeline.extract.OpenAI")
def test_extract_rejects_empty_string_fields(mock_openai_cls):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _fake_response(
        {**_EXTRACTION, "results": ""}
    )
    mock_openai_cls.return_value = mock_client

    result, _, errors, _, _ = extract_papers([_make_paper("2401.0001")])

    assert errors == 1
    assert result[0].extract_status == "error"


@patch("app.pipeline.extract.OpenAI")
def test_extract_falls_back_to_secondary_model_on_rate_limit(mock_openai_cls, monkeypatch):
    """A 429 from the primary model must not immediately fail the paper."""
    import os
    from app.config import get_settings

    monkeypatch.setenv("LLM_API_KEY", os.environ.get("LLM_API_KEY", "test-key"))
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "backup-model")
    get_settings.cache_clear()

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        _rate_limit_error(),
        _fake_response(_EXTRACTION),
    ]
    mock_openai_cls.return_value = mock_client

    result, _, errors, _, _ = extract_papers([_make_paper("2401.0001")])

    get_settings.cache_clear()
    assert errors == 0
    assert result[0].extract_status == "done"
    calls = mock_client.chat.completions.create.call_args_list
    assert calls[0].kwargs["model"] != calls[1].kwargs["model"]
    assert calls[1].kwargs["model"] == "backup-model"


def test_extract_serves_cached_paper_without_calling_llm(monkeypatch):
    cached = Paper(
        arxiv_id="2401.0001",
        title="Cached Paper",
        abstract="An abstract.",
        authors=[Author(name="Alice")],
        categories=["cs.LG"],
        published="2024-01-01",
        url="https://arxiv.org/abs/2401.0001",
        problem="Cached problem.",
        method="Cached method.",
        results="Cached results.",
        contribution="Cached contribution.",
        extract_status="done",
    )
    monkeypatch.setattr(storage, "get_cached_extraction", lambda arxiv_id: cached)

    with patch("app.pipeline.extract.OpenAI") as mock_openai_cls:
        result, _, errors, prompt_tokens, completion_tokens = extract_papers(
            [_make_paper("2401.0001")]
        )
        # A cache hit must never touch the LLM client at all.
        mock_openai_cls.assert_not_called()

    assert errors == 0
    assert prompt_tokens == 0
    assert completion_tokens == 0
    assert result[0].problem == "Cached problem."
    assert result[0].extract_status == "done"


@patch("app.pipeline.extract.OpenAI")
def test_extract_writes_successful_result_to_storage(mock_openai_cls, monkeypatch):
    saved: list[Paper] = []
    monkeypatch.setattr(storage, "save_extraction", lambda paper: saved.append(paper))

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _fake_response(_EXTRACTION)
    mock_openai_cls.return_value = mock_client

    extract_papers([_make_paper("2401.0001")])

    assert len(saved) == 1
    assert saved[0].arxiv_id == "2401.0001"
    assert saved[0].extract_status == "done"


@patch("app.pipeline.extract.OpenAI")
def test_extract_writes_failed_result_to_storage_too(mock_openai_cls, monkeypatch):
    """A cached failure still avoids re-hitting a dead paper in a later run."""
    saved: list[Paper] = []
    monkeypatch.setattr(storage, "save_extraction", lambda paper: saved.append(paper))

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("boom")
    mock_openai_cls.return_value = mock_client

    extract_papers([_make_paper("2401.0001")])

    assert len(saved) == 1
    assert saved[0].extract_status == "error"


@patch("app.pipeline.extract.OpenAI")
def test_extract_storage_write_failure_does_not_fail_the_paper(mock_openai_cls, monkeypatch):
    """A broken cache write must degrade, not take down an otherwise-good result."""

    def _boom(paper):
        raise RuntimeError("disk full")

    monkeypatch.setattr(storage, "save_extraction", _boom)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _fake_response(_EXTRACTION)
    mock_openai_cls.return_value = mock_client

    result, _, errors, _, _ = extract_papers([_make_paper("2401.0001")])

    assert errors == 0
    assert result[0].extract_status == "done"


@patch("app.pipeline.extract.OpenAI")
def test_extract_accumulates_token_usage(mock_openai_cls):
    response = _fake_response(_EXTRACTION)
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 40
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = response
    mock_openai_cls.return_value = mock_client

    _, _, errors, prompt_tokens, completion_tokens = extract_papers(
        [_make_paper("2401.0001"), _make_paper("2401.0002")]
    )

    assert errors == 0
    assert prompt_tokens == 200
    assert completion_tokens == 80
