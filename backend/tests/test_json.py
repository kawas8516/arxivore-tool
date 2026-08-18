import json
import httpx
import openai
import pytest
from unittest.mock import MagicMock

from pydantic import BaseModel, Field

from app.pipeline._json import (
    LLMOutputError,
    RateLimitExceeded,
    call_json,
    call_json_with_fallback,
    content_of,
    parse_model,
    strip_fences,
    usage_of,
)


class _Sample(BaseModel):
    name: str = Field(min_length=1)
    count: int


def _response(content: str | None) -> MagicMock:
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def test_strip_fences_handles_bare_and_tagged_fences():
    assert strip_fences('```\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_fences('{"a": 1}') == '{"a": 1}'


def test_content_of_strips_fences():
    assert content_of(_response('```json\n{"a": 1}\n```')) == '{"a": 1}'


def test_content_of_raises_on_empty():
    with pytest.raises(LLMOutputError):
        content_of(_response(None))
    with pytest.raises(LLMOutputError):
        content_of(_response(""))


def test_parse_model_accepts_valid():
    parsed = parse_model(json.dumps({"name": "ok", "count": 3}), _Sample)
    assert parsed.name == "ok"
    assert parsed.count == 3


def test_parse_model_rejects_malformed_json():
    with pytest.raises(LLMOutputError):
        parse_model("{not json", _Sample)


def test_parse_model_rejects_off_schema():
    """Valid JSON with the wrong shape must fail, not silently default."""
    with pytest.raises(LLMOutputError):
        parse_model(json.dumps({"name": "", "count": 3}), _Sample)
    with pytest.raises(LLMOutputError):
        parse_model(json.dumps({"count": 3}), _Sample)


def test_usage_of_reads_integers():
    response = MagicMock()
    response.usage.prompt_tokens = 120
    response.usage.completion_tokens = 45
    assert usage_of(response) == (120, 45)


def test_usage_of_defaults_when_absent():
    response = MagicMock()
    response.usage = None
    assert usage_of(response) == (0, 0)


def test_usage_of_defaults_when_not_integers():
    """A bare MagicMock yields MagicMock attributes; accounting must not break."""
    assert usage_of(MagicMock()) == (0, 0)


def _rate_limit_error() -> openai.RateLimitError:
    request = httpx.Request("POST", "https://example.com")
    response = httpx.Response(429, request=request)
    return openai.RateLimitError("rate limited", response=response, body=None)


def test_call_json_raises_rate_limit_exceeded_on_429():
    client = MagicMock()
    client.chat.completions.create.side_effect = _rate_limit_error()

    with pytest.raises(RateLimitExceeded):
        call_json(
            client, model="m", system="s", user="u", max_tokens=10, schema=_Sample
        )


def test_call_json_with_fallback_retries_on_rate_limit():
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _rate_limit_error(),
        _response(json.dumps({"name": "ok", "count": 1})),
    ]

    result, _, _ = call_json_with_fallback(
        client,
        model="primary",
        fallback_model="backup",
        system="s",
        user="u",
        max_tokens=10,
        schema=_Sample,
    )

    assert result.name == "ok"
    calls = client.chat.completions.create.call_args_list
    assert calls[0].kwargs["model"] == "primary"
    assert calls[1].kwargs["model"] == "backup"


def test_call_json_with_fallback_disabled_reraises():
    client = MagicMock()
    client.chat.completions.create.side_effect = _rate_limit_error()

    with pytest.raises(RateLimitExceeded):
        call_json_with_fallback(
            client,
            model="primary",
            fallback_model="",
            system="s",
            user="u",
            max_tokens=10,
            schema=_Sample,
        )
    # No fallback model configured: only the primary was ever tried.
    assert client.chat.completions.create.call_count == 1


def test_call_json_with_fallback_propagates_non_rate_limit_errors():
    """A validation failure must not trigger a fallback retry — only 429s do."""
    client = MagicMock()
    client.chat.completions.create.return_value = _response("{not json")

    with pytest.raises(LLMOutputError):
        call_json_with_fallback(
            client,
            model="primary",
            fallback_model="backup",
            system="s",
            user="u",
            max_tokens=10,
            schema=_Sample,
        )
    assert client.chat.completions.create.call_count == 1
