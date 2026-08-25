import json
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest
from pydantic import BaseModel, Field

import app.llm as llm
from app.llm import AllModelsRateLimited, call_json, complete, resolve_pool
from app.pipeline._json import LLMOutputError


@pytest.fixture(autouse=True)
def _reset_state():
    """Cooldown registry and catalog cache are module globals — reset per test."""
    llm._cooldowns.clear()
    llm._catalog_cache = None
    llm._catalog_expires = 0.0
    yield
    llm._cooldowns.clear()


def _rate_limit_error(retry_after: str | None = "30") -> openai.RateLimitError:
    headers = {"retry-after": retry_after} if retry_after else {}
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(429, headers=headers, request=request)
    return openai.RateLimitError("rate limited", response=response, body=None)


def _ok_response(content: str, model: str) -> MagicMock:
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    response.model = model
    return response


def test_complete_success_passes_models_array():
    client = MagicMock()
    client.chat.completions.create.return_value = _ok_response("hello", model="a")
    with patch("app.llm._get_client", return_value=client):
        out, prompt_tokens, completion_tokens = complete(
            [{"role": "user", "content": "hi"}], pool=["a", "b", "c"], max_tokens=10
        )

    assert out == "hello"
    # Usage travels with the content so budget.add() has something to count.
    assert (prompt_tokens, completion_tokens) == (0, 0)
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "a"
    # OpenRouter caps the fallback array at 3; we pass up to 3 models.
    assert kwargs["extra_body"]["models"] == ["a", "b", "c"]


def test_complete_rate_limited_raises_and_cools_models():
    client = MagicMock()
    client.chat.completions.create.side_effect = _rate_limit_error()
    with patch("app.llm._get_client", return_value=client):
        with pytest.raises(AllModelsRateLimited):
            complete([{"role": "user", "content": "hi"}], pool=["a", "b"], max_tokens=10)

    # Both attempted models are now on cooldown and skipped next time.
    assert llm._is_cooling("a")
    assert llm._is_cooling("b")


def test_complete_skips_models_on_cooldown():
    llm._cool(["a"], 60)  # pretend "a" was just rate-limited
    client = MagicMock()
    client.chat.completions.create.return_value = _ok_response("ok", model="b")
    with patch("app.llm._get_client", return_value=client):
        out, _, _ = complete(
            [{"role": "user", "content": "hi"}], pool=["a", "b"], max_tokens=10
        )

    assert out == "ok"
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "b"  # cooling "a" was dropped
    assert kwargs["extra_body"]["models"] == ["b"]


def test_complete_cools_primary_when_openrouter_falls_back():
    """When OpenRouter silently serves from a fallback model, cool the primary."""
    client = MagicMock()
    # response.model = "b" even though we requested "a" → OpenRouter inner fallback
    client.chat.completions.create.return_value = _ok_response("ok", model="b")
    with patch("app.llm._get_client", return_value=client):
        complete([{"role": "user", "content": "hi"}], pool=["a", "b"], max_tokens=10)

    assert llm._is_cooling("a")  # primary should now be cooled
    assert not llm._is_cooling("b")  # fallback model is fine


def test_complete_empty_pool_raises():
    with pytest.raises(AllModelsRateLimited):
        complete([{"role": "user", "content": "hi"}], pool=[], max_tokens=10)


def test_complete_upstream_5xx_raises_rate_limited():
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(503, request=request)
    err = openai.APIStatusError("server error", response=response, body=None)
    client = MagicMock()
    client.chat.completions.create.side_effect = err
    with patch("app.llm._get_client", return_value=client):
        with pytest.raises(AllModelsRateLimited):
            complete([{"role": "user", "content": "hi"}], pool=["a"], max_tokens=10)


def _catalog() -> dict:
    def m(mid, ctx, prompt="0", completion="0"):
        return {
            "id": mid,
            "pricing": {"prompt": prompt, "completion": completion},
            "context_length": ctx,
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
        }

    return {
        "data": [
            m("big:free", 200000),
            m("small:free", 40000),
            m("paid:model", 500000, prompt="0.001", completion="0.002"),  # not free
            m("foo-rerank:free", 300000),  # specialized reranker — excluded
        ]
    }


def test_resolve_pool_auto_discovers_and_splits():
    resp = MagicMock()
    resp.json.return_value = _catalog()
    resp.raise_for_status.return_value = None
    with patch("app.llm.httpx.get", return_value=resp):
        synthesis = resolve_pool("auto", "synthesis")
        rerank = resolve_pool("auto", "rerank")

    # synthesis pool keeps only large-context free chat models
    assert synthesis == ["big:free"]
    # rerank pool keeps all free chat models, context desc; paid + reranker dropped
    assert rerank == ["big:free", "small:free"]


def test_resolve_pool_explicit_list_splits_csv():
    assert resolve_pool("a:free, b:free ,c:free", "rerank") == ["a:free", "b:free", "c:free"]


# --- call_json: JSON mode, schema validation, usage plumbing -----------------


class _Sample(BaseModel):
    name: str = Field(min_length=1)
    count: int


def _usage_response(content: str, *, prompt: int, completion: int) -> MagicMock:
    response = _ok_response(content, model="a")
    usage = MagicMock()
    usage.prompt_tokens = prompt
    usage.completion_tokens = completion
    response.usage = usage
    return response


def test_call_json_requests_json_mode_and_returns_usage():
    client = MagicMock()
    client.chat.completions.create.return_value = _usage_response(
        json.dumps({"name": "ok", "count": 2}), prompt=11, completion=7
    )
    with patch("app.llm._get_client", return_value=client):
        result, prompt_tokens, completion_tokens = call_json(
            [{"role": "user", "content": "hi"}], pool=["a"], max_tokens=10, schema=_Sample
        )

    assert (result.name, result.count) == ("ok", 2)
    assert (prompt_tokens, completion_tokens) == (11, 7)
    assert client.chat.completions.create.call_args.kwargs["response_format"] == {
        "type": "json_object"
    }


def test_call_json_retries_without_json_mode_on_400():
    """A route that rejects response_format still works — parsing covers us."""
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    bad_request = openai.BadRequestError(
        "unsupported", response=httpx.Response(400, request=request), body=None
    )
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        bad_request,
        _usage_response(json.dumps({"name": "ok", "count": 1}), prompt=1, completion=1),
    ]
    with patch("app.llm._get_client", return_value=client):
        result, _, _ = call_json(
            [{"role": "user", "content": "hi"}], pool=["a"], max_tokens=10, schema=_Sample
        )

    assert result.name == "ok"
    calls = client.chat.completions.create.call_args_list
    assert "response_format" in calls[0].kwargs
    assert "response_format" not in calls[1].kwargs


def test_call_json_strips_fences_before_parsing():
    client = MagicMock()
    client.chat.completions.create.return_value = _usage_response(
        '```json\n{"name": "ok", "count": 3}\n```', prompt=0, completion=0
    )
    with patch("app.llm._get_client", return_value=client):
        result, _, _ = call_json(
            [{"role": "user", "content": "hi"}], pool=["a"], max_tokens=10, schema=_Sample
        )

    assert result.count == 3


def test_call_json_rejects_off_schema_response():
    client = MagicMock()
    client.chat.completions.create.return_value = _usage_response(
        json.dumps({"name": "", "count": 1}), prompt=0, completion=0
    )
    with patch("app.llm._get_client", return_value=client):
        with pytest.raises(LLMOutputError):
            call_json(
                [{"role": "user", "content": "hi"}], pool=["a"], max_tokens=10, schema=_Sample
            )


def test_call_json_does_not_retry_a_rate_limit_as_a_400():
    """Only a 400 falls through to the no-JSON-mode retry; a 429 propagates."""
    client = MagicMock()
    client.chat.completions.create.side_effect = _rate_limit_error()
    with patch("app.llm._get_client", return_value=client):
        with pytest.raises(AllModelsRateLimited):
            call_json(
                [{"role": "user", "content": "hi"}], pool=["a"], max_tokens=10, schema=_Sample
            )
    assert client.chat.completions.create.call_count == 1


# --- withdrawn model ids: 400/404, not 429 ----------------------------------


def _status_error(status: int, message: str) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(status, request=request, json={"error": {"message": message}})
    return openai.APIStatusError(message, response=response, body=None)


def test_complete_fails_over_past_a_withdrawn_model_id():
    """Regression: a retired id returns 400, so rate-limit failover never saw it
    and the whole stage died on a config that worked last month."""
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _status_error(400, "dead/model:free is not a valid model ID"),
        _ok_response("recovered", model="live/model:free"),
    ]
    with patch("app.llm._get_client", return_value=client):
        out, _, _ = complete(
            [{"role": "user", "content": "hi"}],
            pool=["dead/model:free", "live/model:free"],
            max_tokens=10,
        )

    assert out == "recovered"
    # Only the id the provider named is dropped; the survivor is tried next.
    assert client.chat.completions.create.call_args_list[1].kwargs["model"] == "live/model:free"
    assert llm._is_cooling("dead/model:free")
    assert not llm._is_cooling("live/model:free")


def test_complete_raises_when_every_model_id_is_unservable():
    client = MagicMock()
    client.chat.completions.create.side_effect = _status_error(
        404, "No endpoints found for the requested model"
    )
    with patch("app.llm._get_client", return_value=client):
        with pytest.raises(AllModelsRateLimited):
            complete(
                [{"role": "user", "content": "hi"}], pool=["a", "b"], max_tokens=10
            )
    # Unnamed models drop one per pass, so the loop terminates instead of spinning.
    assert client.chat.completions.create.call_count == 2


def test_complete_does_not_treat_a_plain_400_as_a_dead_model():
    """A route rejecting response_format is a request problem, not a model one —
    call_json's no-JSON-mode retry depends on that error reaching it intact."""
    client = MagicMock()
    client.chat.completions.create.side_effect = _status_error(
        400, "response_format is not supported by this provider"
    )
    with patch("app.llm._get_client", return_value=client):
        with pytest.raises(openai.APIStatusError):
            complete([{"role": "user", "content": "hi"}], pool=["a", "b"], max_tokens=10)
    assert client.chat.completions.create.call_count == 1
    assert not llm._is_cooling("a")
