import json
import pytest
from unittest.mock import MagicMock

from pydantic import BaseModel, Field

from app.pipeline._json import (
    LLMOutputError,
    content_of,
    parse_model,
    strip_fences,
    usage_of,
)

# _json.py is parsing and accounting only. The request itself, model failover,
# and rate-limit handling live in app/llm.py and are covered by test_llm.py.


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
