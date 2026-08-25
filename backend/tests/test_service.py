"""Integration-level tests for run_pipeline: run_id generation, persistence,
and the actual point of Phase 1 — a paper re-surfacing under a second topic
costs zero extraction tokens on the second run."""

import json
from unittest.mock import MagicMock, patch

import pytest

from app import storage
from app.config import get_settings
from app.service import run_pipeline


def _arxiv_result(arxiv_id: str, title: str = "A Paper"):
    from datetime import datetime, timezone

    result = MagicMock()
    result.entry_id = f"https://arxiv.org/abs/{arxiv_id}"
    result.title = title
    result.summary = "An abstract about a research topic."
    author = MagicMock()
    author.name = "A. Researcher"
    result.authors = [author]
    result.categories = ["cs.LG"]
    result.published = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return result


def _extract_response(data: dict) -> MagicMock:
    message = MagicMock()
    message.content = json.dumps(data)
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    response.usage.prompt_tokens = 50
    response.usage.completion_tokens = 25
    return response


def _rerank_response(arxiv_ids: list[str]) -> MagicMock:
    payload = {"items": [{"arxiv_id": i, "score": 0.8, "rationale": "ok"} for i in arxiv_ids]}
    message = MagicMock()
    message.content = json.dumps(payload)
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 50
    return response


def _landscape_response() -> MagicMock:
    payload = {
        "clusters": [{"name": "C", "summary": "S", "arxiv_ids": ["2401.0001"]}],
        "relationships": [],
        "tensions": [],
        "open_problems": [],
    }
    message = MagicMock()
    message.content = json.dumps(payload)
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    response.usage.prompt_tokens = 200
    response.usage.completion_tokens = 80
    return response


def _stage_router(arxiv_ids: list[str], extract_calls: list):
    """One fake client now serves every stage.

    All four stages share app.llm's client singleton, so they can no longer be
    mocked apart by module. Dispatch on the system prompt instead — it is the
    one thing that identifies which stage is calling.
    """

    def create(*, model, max_tokens, messages, **kwargs):
        system = messages[0]["content"].lower()
        if "arxiv" in system and "quer" in system:
            return _extract_response({"queries": []})
        if "rank" in system:
            return _rerank_response(arxiv_ids)
        if "landscape" in system or "synthes" in system:
            return _landscape_response()
        extract_calls.append(messages)
        return _extract_response(_EXTRACTION)

    return create


_EXTRACTION = {
    "problem": "A problem.",
    "method": "A method.",
    "results": "Some results.",
    "contribution": "A contribution.",
}


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    get_settings.cache_clear()
    storage.init_db()
    yield
    get_settings.cache_clear()


@patch("app.llm._get_client")
def test_run_pipeline_returns_a_unique_run_id(mock_get_client):
    mock_get_client.return_value.chat.completions.create.side_effect = _stage_router(
        ["2401.0001"], []
    )

    with patch("app.pipeline.retrieve.arxiv.Client") as mock_arxiv_client_cls:
        mock_arxiv_client_cls.return_value.results.return_value = iter(
            [_arxiv_result("2401.0001")]
        )
        response_a = run_pipeline("topic one")
        response_b = run_pipeline("topic two")

    assert response_a.run_id
    assert response_b.run_id
    assert response_a.run_id != response_b.run_id


@patch("app.llm._get_client")
def test_run_is_persisted_and_retrievable(mock_get_client):
    mock_get_client.return_value.chat.completions.create.side_effect = _stage_router(
        ["2401.0001"], []
    )

    with patch("app.pipeline.retrieve.arxiv.Client") as mock_arxiv_client_cls:
        mock_arxiv_client_cls.return_value.results.return_value = iter(
            [_arxiv_result("2401.0001")]
        )
        response = run_pipeline("a topic")

    stored = storage.get_run(response.run_id)
    assert stored is not None
    assert stored["topic"] == "a topic"
    assert stored["papers"][0].arxiv_id == "2401.0001"
    # Proves the router reached the synthesize branch rather than quietly
    # serving it an extraction that fails schema and lands as landscape=None.
    assert response.landscape is not None
    assert response.landscape.clusters[0].name == "C"


@patch("app.llm._get_client")
def test_second_run_reuses_cached_extraction_for_shared_paper(mock_get_client):
    """The actual point of Phase 1: a paper surfacing under two topics is
    extracted once. The second run's extract call count must not grow."""
    extract_calls: list = []
    mock_get_client.return_value.chat.completions.create.side_effect = _stage_router(
        ["2401.0001"], extract_calls
    )

    with patch("app.pipeline.retrieve.arxiv.Client") as mock_arxiv_client_cls:
        mock_arxiv_client_cls.return_value.results.return_value = iter(
            [_arxiv_result("2401.0001")]
        )
        response_a = run_pipeline("retrieval augmented generation")
        assert len(extract_calls) == 1
        assert response_a.prompt_tokens > 0

        mock_arxiv_client_cls.return_value.results.return_value = iter(
            [_arxiv_result("2401.0001")]
        )
        response_b = run_pipeline("retrieval methods")

    # No second extract call for the same paper — served from the papers cache.
    assert len(extract_calls) == 1
    assert response_b.papers[0].extract_status == "done"
    assert response_b.papers[0].problem == _EXTRACTION["problem"]
