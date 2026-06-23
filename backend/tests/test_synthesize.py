import json
from unittest.mock import MagicMock, patch

from app.models import Author, Paper
from app.pipeline.synthesize import synthesize_landscape


def _make_paper(arxiv_id: str, status: str = "done") -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        abstract="Abstract text.",
        authors=[Author(name="Alice")],
        categories=["cs.LG"],
        published="2024-01-01",
        url=f"https://arxiv.org/abs/{arxiv_id}",
        relevance_score=0.9,
        problem="A problem.",
        method="A method.",
        results="Some results.",
        contribution="A contribution.",
        extract_status=status,
    )


def _fake_response(data: dict) -> MagicMock:
    message = MagicMock()
    message.content = json.dumps(data)
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


_LANDSCAPE = {
    "clusters": [
        {
            "name": "Dense Retrieval Methods",
            "summary": "Papers using dense vector retrieval.",
            "arxiv_ids": ["2401.0001", "2401.0002"],
        },
        {
            "name": "Reranking Approaches",
            "summary": "Papers focused on reranking.",
            "arxiv_ids": ["2401.0003"],
        },
    ],
    "relationships": [
        {
            "from_cluster": "Reranking Approaches",
            "to_cluster": "Dense Retrieval Methods",
            "kind": "builds-on",
            "description": "Reranking refines dense retrieval outputs.",
        }
    ],
    "tensions": ["Dense vs. sparse retrieval trade-offs remain contested."],
    "open_problems": ["Scaling retrieval to billions of documents."],
}


@patch("app.pipeline.synthesize.OpenAI")
def test_synthesize_parses_landscape(mock_openai_cls):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _fake_response(_LANDSCAPE)
    mock_openai_cls.return_value = mock_client

    papers = [_make_paper("2401.0001"), _make_paper("2401.0002"), _make_paper("2401.0003")]
    landscape, elapsed_ms = synthesize_landscape("retrieval", papers)

    assert elapsed_ms >= 0
    assert len(landscape.clusters) == 2
    assert landscape.clusters[0].name == "Dense Retrieval Methods"
    assert landscape.clusters[0].arxiv_ids == ["2401.0001", "2401.0002"]
    assert len(landscape.relationships) == 1
    assert landscape.relationships[0].kind == "builds-on"
    assert landscape.tensions == _LANDSCAPE["tensions"]
    assert landscape.open_problems == _LANDSCAPE["open_problems"]


@patch("app.pipeline.synthesize.OpenAI")
def test_synthesize_only_sends_extracted_papers(mock_openai_cls):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _fake_response(_LANDSCAPE)
    mock_openai_cls.return_value = mock_client

    papers = [
        _make_paper("2401.0001", status="done"),
        _make_paper("2401.0002", status="error"),  # should be excluded
        _make_paper("2401.0003", status="done"),
    ]
    synthesize_landscape("retrieval", papers)

    # Inspect the payload actually sent to the LLM
    call = mock_client.chat.completions.create.call_args
    user_content = call.kwargs["messages"][-1]["content"]
    assert "2401.0001" in user_content
    assert "2401.0003" in user_content
    assert "2401.0002" not in user_content


@patch("app.pipeline.synthesize.OpenAI")
def test_synthesize_handles_markdown_fenced_json(mock_openai_cls):
    mock_client = MagicMock()
    fenced = MagicMock()
    fenced.content = "```json\n" + json.dumps(_LANDSCAPE) + "\n```"
    choice = MagicMock()
    choice.message = fenced
    response = MagicMock()
    response.choices = [choice]
    mock_client.chat.completions.create.return_value = response
    mock_openai_cls.return_value = mock_client

    papers = [_make_paper("2401.0001")]
    landscape, _ = synthesize_landscape("retrieval", papers)
    assert len(landscape.clusters) == 2
