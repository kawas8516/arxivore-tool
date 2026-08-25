import json
import httpx
import openai
import pytest
from unittest.mock import MagicMock, patch

from app.pipeline.expand import expand_query


def _rate_limit_error() -> openai.RateLimitError:
    request = httpx.Request("POST", "https://example.com")
    response = httpx.Response(429, request=request)
    return openai.RateLimitError("rate limited", response=response, body=None)


def _fake_response(data: dict) -> MagicMock:
    message = MagicMock()
    message.content = json.dumps(data)
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


@patch("app.llm._get_client")
def test_expand_returns_topic_plus_generated_queries(mock_get_client):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _fake_response(
        {"queries": ['all:"retrieval augmented generation"', 'abs:"RAG"']}
    )
    mock_get_client.return_value = mock_client

    queries, elapsed_ms, _, _ = expand_query("retrieval augmented generation")

    # The user's own wording leads; model rewrites follow.
    assert queries[0] == "retrieval augmented generation"
    assert 'abs:"RAG"' in queries
    assert elapsed_ms >= 0


@patch("app.llm._get_client")
def test_expand_falls_back_to_topic_on_llm_failure(mock_get_client):
    """Expansion is additive — a failure must cost recall, never the run."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("provider down")
    mock_get_client.return_value = mock_client

    queries, _, prompt_tokens, completion_tokens = expand_query("diffusion policy")

    assert queries == ["diffusion policy"]
    assert prompt_tokens == 0
    assert completion_tokens == 0


@patch("app.llm._get_client")
def test_expand_falls_back_on_off_schema_response(mock_get_client):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _fake_response({"queries": []})
    mock_get_client.return_value = mock_client

    queries, _, _, _ = expand_query("state space models")

    assert queries == ["state space models"]


@patch("app.llm._get_client")
def test_expand_degrades_to_topic_when_every_model_is_rate_limited(mock_get_client):
    """Expansion is best-effort: an exhausted pool costs recall, never the run."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _rate_limit_error()
    mock_get_client.return_value = mock_client

    queries, _, prompt_tokens, completion_tokens = expand_query("topic")

    assert queries == ["topic"]
    assert (prompt_tokens, completion_tokens) == (0, 0)


@patch("app.llm._get_client")
def test_expand_dedupes_and_drops_blanks(mock_get_client):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _fake_response(
        {"queries": ["  ", "topic", "topic", 'ti:"x"']}
    )
    mock_get_client.return_value = mock_client

    queries, _, _, _ = expand_query("topic")

    assert queries == ["topic", 'ti:"x"']
