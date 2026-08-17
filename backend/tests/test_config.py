"""Guards against the class of bug that shipped a retired model ID as a default.

`meta-llama/llama-3.3-70b-instruct:free` was the default rerank model but had been
withdrawn from the provider catalog, so a fresh clone failed at the rerank *and*
extract stages. The live check below catches a recurrence; it is opt-in because it
needs network.

Run it with:  pytest -m live_models
"""

import os
import urllib.request

import pytest

from app.config import Settings, get_settings


def _settings() -> Settings:
    # A key is required by the schema but irrelevant to these assertions.
    os.environ.setdefault("LLM_API_KEY", "test-key-not-used")
    get_settings.cache_clear()
    return get_settings()


def test_no_configured_model_uses_a_retired_id():
    """Regression: this exact ID does not exist in the OpenRouter catalog."""
    settings = _settings()
    retired = {"meta-llama/llama-3.3-70b-instruct:free"}
    configured = {
        settings.llm_rerank_model,
        settings.llm_extract_model,
        settings.llm_synthesis_model,
        settings.llm_expand_model,
    }
    assert not (configured & retired)


def test_every_stage_has_its_own_model_setting():
    """Extraction must not silently inherit the rerank model."""
    settings = _settings()
    for field in (
        "llm_rerank_model",
        "llm_extract_model",
        "llm_synthesis_model",
        "llm_expand_model",
    ):
        assert getattr(settings, field), f"{field} must not be empty"


def test_extract_concurrency_is_conservative():
    """Free-tier providers rate-limit; high concurrency lowers completion."""
    settings = _settings()
    assert 1 <= settings.extract_concurrency <= 5


@pytest.mark.live_models
def test_configured_models_exist_in_provider_catalog():
    settings = _settings()
    url = f"{settings.llm_base_url.rstrip('/')}/models"
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
        import json

        catalog = {m["id"] for m in json.load(response)["data"]}

    for field in (
        "llm_rerank_model",
        "llm_extract_model",
        "llm_synthesis_model",
        "llm_expand_model",
    ):
        model = getattr(settings, field)
        assert model in catalog, f"{field}={model!r} is not in the provider catalog"
