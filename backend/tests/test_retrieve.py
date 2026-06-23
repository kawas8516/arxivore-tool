from unittest.mock import MagicMock, patch
from datetime import date, datetime, timezone

from app.models import Paper
from app.pipeline.retrieve import retrieve_candidates


def _make_arxiv_result(arxiv_id: str, title: str, abstract: str):
    result = MagicMock()
    result.entry_id = f"https://arxiv.org/abs/{arxiv_id}"
    result.title = title
    result.summary = abstract
    # name= is a reserved MagicMock kwarg; set .name explicitly
    author_a, author_b = MagicMock(), MagicMock()
    author_a.name, author_b.name = "Alice", "Bob"
    result.authors = [author_a, author_b]
    result.categories = ["cs.LG"]
    result.published = datetime(2024, 1, 15, tzinfo=timezone.utc)
    return result


@patch("app.pipeline.retrieve.arxiv.Client")
def test_retrieve_maps_fields(mock_client_cls):
    fake_result = _make_arxiv_result(
        "2401.00001", "Test Paper", "This paper does stuff."
    )
    mock_client = MagicMock()
    mock_client.results.return_value = iter([fake_result])
    mock_client_cls.return_value = mock_client

    papers, elapsed_ms = retrieve_candidates("test topic")

    assert len(papers) == 1
    p = papers[0]
    assert p.arxiv_id == "2401.00001"
    assert p.title == "Test Paper"
    assert p.abstract == "This paper does stuff."
    assert p.published == "2024-01-15"
    assert p.url == "https://arxiv.org/abs/2401.00001"
    assert elapsed_ms >= 0


@patch("app.pipeline.retrieve.arxiv.Client")
def test_retrieve_empty_returns_empty_list(mock_client_cls):
    mock_client = MagicMock()
    mock_client.results.return_value = iter([])
    mock_client_cls.return_value = mock_client

    papers, _ = retrieve_candidates("some topic")
    assert papers == []
