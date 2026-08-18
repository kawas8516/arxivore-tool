from unittest.mock import MagicMock, patch

from app.pipeline.fulltext import fetch_excerpt


def _fake_urlopen(pdf_bytes: bytes):
    cm = MagicMock()
    cm.__enter__.return_value.read.return_value = pdf_bytes
    return cm


@patch("app.pipeline.fulltext.PdfReader")
@patch("app.pipeline.fulltext.urllib.request.urlopen")
def test_fetch_excerpt_isolates_results_section(mock_urlopen, mock_reader_cls):
    mock_urlopen.return_value = _fake_urlopen(b"fake-pdf-bytes")
    page1 = MagicMock()
    page1.extract_text.return_value = "Introduction text here.\n\n"
    page2 = MagicMock()
    page2.extract_text.return_value = "5. Results\nWe achieve 92.3% accuracy on the benchmark."
    mock_reader_cls.return_value.pages = [page1, page2]

    excerpt = fetch_excerpt("2401.0001", max_chars=6000)

    assert excerpt is not None
    assert "92.3%" in excerpt
    assert "Introduction text" not in excerpt  # isolated to the Results section onward


@patch("app.pipeline.fulltext.PdfReader")
@patch("app.pipeline.fulltext.urllib.request.urlopen")
def test_fetch_excerpt_falls_back_to_full_text_without_a_matching_header(
    mock_urlopen, mock_reader_cls
):
    mock_urlopen.return_value = _fake_urlopen(b"fake-pdf-bytes")
    page = MagicMock()
    page.extract_text.return_value = "Just some prose with no section headers at all."
    mock_reader_cls.return_value.pages = [page]

    excerpt = fetch_excerpt("2401.0001", max_chars=6000)

    assert excerpt == "Just some prose with no section headers at all."


@patch("app.pipeline.fulltext.urllib.request.urlopen")
def test_fetch_excerpt_returns_none_on_network_failure(mock_urlopen):
    mock_urlopen.side_effect = TimeoutError("connection timed out")
    assert fetch_excerpt("2401.0001", max_chars=6000) is None


@patch("app.pipeline.fulltext.PdfReader")
@patch("app.pipeline.fulltext.urllib.request.urlopen")
def test_fetch_excerpt_returns_none_on_unparseable_pdf(mock_urlopen, mock_reader_cls):
    mock_urlopen.return_value = _fake_urlopen(b"not actually a pdf")
    mock_reader_cls.side_effect = Exception("invalid PDF header")

    assert fetch_excerpt("2401.0001", max_chars=6000) is None


@patch("app.pipeline.fulltext.PdfReader")
@patch("app.pipeline.fulltext.urllib.request.urlopen")
def test_fetch_excerpt_truncates_to_max_chars(mock_urlopen, mock_reader_cls):
    mock_urlopen.return_value = _fake_urlopen(b"fake-pdf-bytes")
    page = MagicMock()
    page.extract_text.return_value = "Results\n" + ("x" * 10000)
    mock_reader_cls.return_value.pages = [page]

    excerpt = fetch_excerpt("2401.0001", max_chars=100)
    assert len(excerpt) <= 100


@patch("app.pipeline.fulltext.PdfReader")
@patch("app.pipeline.fulltext.urllib.request.urlopen")
def test_fetch_excerpt_returns_none_for_empty_extracted_text(mock_urlopen, mock_reader_cls):
    """A scanned-image PDF with no extractable text layer is a clean miss."""
    mock_urlopen.return_value = _fake_urlopen(b"fake-pdf-bytes")
    page = MagicMock()
    page.extract_text.return_value = ""
    mock_reader_cls.return_value.pages = [page]

    assert fetch_excerpt("2401.0001", max_chars=6000) is None


@patch("app.pipeline.fulltext.urllib.request.urlopen")
def test_fetch_excerpt_uses_unversioned_pdf_url(mock_urlopen):
    mock_urlopen.side_effect = TimeoutError("boom")
    fetch_excerpt("2401.12345", max_chars=6000)

    requested_url = mock_urlopen.call_args[0][0].full_url
    assert requested_url == "https://arxiv.org/pdf/2401.12345"
