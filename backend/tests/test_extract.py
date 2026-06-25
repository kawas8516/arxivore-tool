import json
from unittest.mock import patch

from app.llm import strip_fences
from app.models import Author, Paper
from app.pipeline.extract import extract_papers


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


_FIELDS = {
    "problem": "RAG systems struggle with retrieval quality.",
    "method": "We use a dense retriever with cross-attention reranking.",
    "results": "Achieves 5% improvement on NaturalQuestions benchmark.",
    "contribution": "A novel cross-attention reranking module for RAG pipelines.",
}


def _batch_response(arxiv_ids: list[str]) -> str:
    """Return the JSON string a model would produce for a batch."""
    return json.dumps([{"arxiv_id": aid, **_FIELDS} for aid in arxiv_ids])


@patch("app.pipeline.extract.complete")
def test_extract_maps_fields(mock_complete):
    mock_complete.side_effect = lambda msgs, **kw: _batch_response(
        [item["arxiv_id"] for item in json.loads(
            msgs[-1]["content"].split("<papers>")[1].split("</papers>")[0]
        )]
    )

    papers = [_make_paper("2401.0001")]
    result, elapsed_ms, errors = extract_papers(papers)

    assert errors == 0
    assert elapsed_ms >= 0
    p = result[0]
    assert p.extract_status == "done"
    assert p.problem == _FIELDS["problem"]
    assert p.method == _FIELDS["method"]
    assert p.results == _FIELDS["results"]
    assert p.contribution == _FIELDS["contribution"]


@patch("app.pipeline.extract.complete")
def test_extract_batch_failure_marks_all_in_batch_as_error(mock_complete):
    """A parse error on a batch marks all papers in that batch as error."""
    call_count = 0

    def side_effect(msgs, **kw):
        nonlocal call_count
        call_count += 1
        # First batch fails, second succeeds
        ids = [item["arxiv_id"] for item in json.loads(
            msgs[-1]["content"].split("<papers>")[1].split("</papers>")[0]
        )]
        if "2401.0001" in ids:
            raise RuntimeError("LLM timeout")
        return _batch_response(ids)

    mock_complete.side_effect = side_effect

    # 5 papers: first batch (ids 1-4) fails, second batch (id 5) succeeds
    papers = [_make_paper(f"2401.000{i}") for i in range(1, 6)]
    result, _, errors = extract_papers(papers)

    statuses = {p.arxiv_id: p.extract_status for p in result}
    assert statuses["2401.0005"] == "done"
    assert all(statuses[f"2401.000{i}"] == "error" for i in range(1, 5))
    assert errors == 4


@patch("app.pipeline.extract.complete")
def test_extract_fewer_calls_than_papers(mock_complete):
    """With 8 papers and batch_size=4, complete() is called exactly twice."""
    mock_complete.side_effect = lambda msgs, **kw: _batch_response(
        [item["arxiv_id"] for item in json.loads(
            msgs[-1]["content"].split("<papers>")[1].split("</papers>")[0]
        )]
    )

    papers = [_make_paper(f"2401.{i:04d}") for i in range(8)]
    result, _, errors = extract_papers(papers)

    assert errors == 0
    assert mock_complete.call_count == 2  # 8 papers / batch_size=4 = 2 calls
    assert all(p.extract_status == "done" for p in result)


def test_strip_fences_removes_markdown():
    raw = "```json\n{\"key\": \"value\"}\n```"
    assert strip_fences(raw) == '{"key": "value"}'


def test_strip_fences_passthrough_plain_json():
    raw = '{"key": "value"}'
    assert strip_fences(raw) == raw
