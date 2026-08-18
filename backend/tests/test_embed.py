from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app import storage
from app.config import get_settings
from app.models import Author, Paper
from app.pipeline.embed import prefilter_by_similarity


def _paper(arxiv_id: str) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        abstract="An abstract.",
        authors=[Author(name="Alice")],
        categories=["cs.LG"],
        published="2024-01-01",
        url=f"https://arxiv.org/abs/{arxiv_id}",
    )


def _embedding_response(vectors: list[list[float]]) -> SimpleNamespace:
    return SimpleNamespace(data=[SimpleNamespace(embedding=v) for v in vectors])


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    get_settings.cache_clear()
    storage.init_db()
    yield
    get_settings.cache_clear()


def test_disabled_by_default_returns_candidates_unchanged():
    candidates = [_paper(f"2401.{i:04d}") for i in range(30)]
    assert prefilter_by_similarity("a topic", candidates) == candidates


def test_below_keep_threshold_skips_embedding_entirely(monkeypatch):
    monkeypatch.setenv("EMBED_PREFILTER_ENABLED", "true")
    monkeypatch.setenv("EMBED_PREFILTER_KEEP", "20")
    get_settings.cache_clear()

    candidates = [_paper(f"2401.{i:04d}") for i in range(5)]  # below keep=20
    with patch("app.pipeline.embed.OpenAI") as mock_openai_cls:
        result = prefilter_by_similarity("a topic", candidates)
        mock_openai_cls.assert_not_called()

    assert result == candidates
    get_settings.cache_clear()


@patch("app.pipeline.embed.OpenAI")
def test_keeps_top_n_by_cosine_similarity(mock_openai_cls, monkeypatch):
    monkeypatch.setenv("EMBED_PREFILTER_ENABLED", "true")
    monkeypatch.setenv("EMBED_PREFILTER_KEEP", "2")
    get_settings.cache_clear()

    candidates = [_paper("close"), _paper("far"), _paper("closest")]
    mock_client = MagicMock()
    # Candidate embeddings, then the topic embedding as a separate call.
    mock_client.embeddings.create.side_effect = [
        _embedding_response([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]),
        _embedding_response([[1.0, 0.0]]),  # topic vector — matches "close"/"closest"
    ]
    mock_openai_cls.return_value = mock_client

    kept = prefilter_by_similarity("a topic", candidates)

    get_settings.cache_clear()
    assert {p.arxiv_id for p in kept} == {"close", "closest"}
    assert len(kept) == 2


@patch("app.pipeline.embed.OpenAI")
def test_uses_cached_embedding_and_skips_re_embedding(mock_openai_cls, monkeypatch):
    monkeypatch.setenv("EMBED_PREFILTER_ENABLED", "true")
    monkeypatch.setenv("EMBED_PREFILTER_KEEP", "1")
    get_settings.cache_clear()
    model = get_settings().llm_embedding_model

    storage.save_embedding("cached-paper", model, [1.0, 0.0])

    candidates = [_paper("cached-paper"), _paper("fresh-paper")]
    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = [
        _embedding_response([[0.0, 1.0]]),  # only fresh-paper needs embedding
        _embedding_response([[1.0, 0.0]]),  # topic
    ]
    mock_openai_cls.return_value = mock_client

    prefilter_by_similarity("a topic", candidates)

    # First call embedded exactly one text (the uncached paper), not two.
    first_call_input = mock_client.embeddings.create.call_args_list[0].kwargs["input"]
    assert len(first_call_input) == 1
    get_settings.cache_clear()


@patch("app.pipeline.embed.OpenAI")
def test_falls_back_to_unfiltered_on_any_failure(mock_openai_cls, monkeypatch):
    """An unsupported endpoint/model must degrade to today's behaviour, not
    drop candidates or crash the run."""
    monkeypatch.setenv("EMBED_PREFILTER_ENABLED", "true")
    monkeypatch.setenv("EMBED_PREFILTER_KEEP", "1")
    get_settings.cache_clear()

    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = RuntimeError("404 no such endpoint")
    mock_openai_cls.return_value = mock_client

    candidates = [_paper(f"2401.{i:04d}") for i in range(5)]
    result = prefilter_by_similarity("a topic", candidates)

    get_settings.cache_clear()
    assert result == candidates  # nothing dropped despite keep=1
