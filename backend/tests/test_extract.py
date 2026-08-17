import json
from unittest.mock import MagicMock, patch

from app.models import Author, Paper
from app.pipeline.extract import extract_papers
from app.pipeline._json import strip_fences


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
