from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest

import app.llm as llm
from app.llm import AllModelsRateLimited, complete, resolve_pool


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
        out = complete([{"role": "user", "content": "hi"}], pool=["a", "b", "c"], max_tokens=10)

    assert out == "hello"
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
        out = complete([{"role": "user", "content": "hi"}], pool=["a", "b"], max_tokens=10)

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
