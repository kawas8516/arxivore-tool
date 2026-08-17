from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from app.pipeline.retrieve import retrieve_candidates


def _make_arxiv_result(arxiv_id: str, title: str = "Test Paper", abstract: str = "Stuff."):
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


@patch("app.pipeline.retrieve.arxiv.Client")
def test_retrieve_strips_version_suffix(mock_client_cls):
    """v2 and v1 are the same paper; the canonical id must carry no suffix."""
    mock_client = MagicMock()
    mock_client.results.return_value = iter([_make_arxiv_result("2401.00001v3")])
    mock_client_cls.return_value = mock_client

    papers, _ = retrieve_candidates("test topic")

    assert papers[0].arxiv_id == "2401.00001"
    # The versioned form is still what links back to arXiv
    assert papers[0].url == "https://arxiv.org/abs/2401.00001v3"


@patch("app.pipeline.retrieve.arxiv.Client")
def test_retrieve_dedupes_across_queries_ignoring_version(mock_client_cls):
    mock_client = MagicMock()
    # Same paper from two queries, different revisions, plus one genuinely new
    mock_client.results.side_effect = [
        iter([_make_arxiv_result("2401.00001v1"), _make_arxiv_result("2401.00002")]),
        iter([_make_arxiv_result("2401.00001v2"), _make_arxiv_result("2401.00003")]),
    ]
    mock_client_cls.return_value = mock_client

    papers, _ = retrieve_candidates(["query one", "query two"])

    ids = [p.arxiv_id for p in papers]
    assert ids == ["2401.00001", "2401.00002", "2401.00003"]


@patch("app.pipeline.retrieve.arxiv.Client")
def test_retrieve_survives_one_failing_query(mock_client_cls):
    mock_client = MagicMock()
    mock_client.results.side_effect = [
        RuntimeError("arxiv 400"),
        iter([_make_arxiv_result("2401.00002")]),
    ]
    mock_client_cls.return_value = mock_client

    papers, _ = retrieve_candidates(["bad query", "good query"])

    assert [p.arxiv_id for p in papers] == ["2401.00002"]


@patch("app.pipeline.retrieve.arxiv.Client")
def test_retrieve_applies_category_filter(mock_client_cls):
    mock_client = MagicMock()
    mock_client.results.return_value = iter([])
    mock_client_cls.return_value = mock_client

    with patch("app.pipeline.retrieve.get_settings") as mock_settings:
        settings = MagicMock()
        settings.arxiv_page_size = 50
        settings.max_candidates = 50
        settings.arxiv_categories = "cs.LG,cs.CL"
        mock_settings.return_value = settings

        with patch("app.pipeline.retrieve.arxiv.Search") as mock_search:
            retrieve_candidates("diffusion policy")
            query = mock_search.call_args.kwargs["query"]

    assert "cat:cs.LG OR cat:cs.CL" in query
    assert "diffusion policy" in query
