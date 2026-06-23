import json
from unittest.mock import MagicMock, patch

from app.models import Author, Paper
from app.pipeline.extract import extract_papers, _strip_fences


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
    result, elapsed_ms, errors = extract_papers(papers)

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
    result, _, errors = extract_papers(papers)

    assert errors == 1
    statuses = {p.arxiv_id: p.extract_status for p in result}
    assert statuses["2401.0001"] == "done"
    assert statuses["2401.0002"] == "error"
    assert statuses["2401.0003"] == "done"


def test_strip_fences_removes_markdown():
    raw = "```json\n{\"key\": \"value\"}\n```"
    assert _strip_fences(raw) == '{"key": "value"}'


def test_strip_fences_passthrough_plain_json():
    raw = '{"key": "value"}'
    assert _strip_fences(raw) == raw
