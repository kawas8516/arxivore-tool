"""Guards against the class of bug that shipped a retired model ID as a default.

`meta-llama/llama-3.3-70b-instruct:free` was once the default rerank model but
had been withdrawn from the provider catalog, so a fresh clone failed at the
rerank *and* extract stages. Failover does not save you from this: an unknown
model id comes back 400, not 429, and `app.llm.complete` only fails over on a
rate limit or a 5xx.

The offline tests below check shape; the live check catches a recurrence but is
opt-in because it needs network.

Run it with:  pytest -m live_models
"""

import os
import re
import urllib.request
from pathlib import Path

import pytest

from app.config import Settings, get_settings
from app.llm import _split_csv

# IDs known to have been withdrawn from the OpenRouter catalog. Add to this as
# models retire — a name here is cheaper than a broken fresh clone.
_RETIRED = {
    "meta-llama/llama-3.3-70b-instruct:free",
    "openai/gpt-oss-120b:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "openrouter/owl-alpha",
}


def _settings() -> Settings:
    # A key is required by the schema but irrelevant to these assertions.
    os.environ.setdefault("LLM_API_KEY", "test-key-not-used")
    get_settings.cache_clear()
    return get_settings()


def _default(field: str) -> str:
    """The value a *fresh clone* gets, ignoring any local .env.

    These assertions are about what ships, so they must not read get_settings():
    a developer's own .env would mask a retired default here and let it reach
    everyone else.
    """
    return Settings.model_fields[field].default


def _default_models() -> set[str]:
    """Every model id a fresh clone can dial, pools included."""
    return (
        set(_split_csv(_default("llm_synthesis_models")))
        | set(_split_csv(_default("llm_rerank_models")))
        | {_default("llm_synthesis_model"), _default("llm_rerank_model")}
    )


def test_no_default_model_uses_a_retired_id():
    """Regression: these IDs do not exist in the OpenRouter catalog."""
    assert not (_default_models() & _RETIRED)


def test_pools_are_non_empty_and_have_no_duplicates():
    """A stage with an empty pool raises AllModelsRateLimited on every call."""
    for field in ("llm_synthesis_models", "llm_rerank_models"):
        pool = _split_csv(_default(field))
        assert pool, f"{field} must not be empty"
        assert len(pool) == len(set(pool)), f"{field} has a duplicate entry"


def test_legacy_single_model_vars_match_pool_position_zero():
    """The legacy vars document the default, so drift makes them a lie."""
    assert _default("llm_synthesis_model") == _split_csv(_default("llm_synthesis_models"))[0]
    assert _default("llm_rerank_model") == _split_csv(_default("llm_rerank_models"))[0]


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

    # Checks the shipped defaults *and* whatever this machine's .env resolves
    # to — a local override pointing at a dead model is worth catching too.
    configured = _default_models() | set(_split_csv(settings.llm_synthesis_models)) | set(
        _split_csv(settings.llm_rerank_models)
    )
    missing = sorted(configured - catalog)
    assert not missing, f"not in the provider catalog: {', '.join(missing)}"


# --- the Space's models (HF Inference API), not the backend's ---------------
#
# These live here because nothing else covered them, and that gap is exactly how
# `microsoft/Phi-4-mini-instruct` shipped: it was never served by any provider,
# the Space built and ran green, and extract/synthesize failed on every call
# with only empty Papers/Landscape tabs to show for it.
#
# The Space's llm.py is read as text rather than imported: it sits at the repo
# root and would shadow backend/app/llm.py on sys.path.

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


_SPACE_LLM = _REPO_ROOT / "llm.py"

# `main` is the deployable backend and ships no Gradio app, so these skip there
# rather than fail. They run on hf-spaces-prototype, where the Space lives.
_needs_space = pytest.mark.skipif(
    not _SPACE_LLM.exists(), reason="no Gradio Space on this branch (no root llm.py)"
)


def _space_chat_model() -> str:
    m = re.search(r'_MODEL = "([^"]+)"', _SPACE_LLM.read_text(encoding="utf-8"))
    assert m, "could not find _MODEL in the Space's llm.py"
    return m.group(1)


@_needs_space
def test_space_model_is_not_a_known_dead_id():
    """Offline guard: these were configured and never served."""
    assert _space_chat_model() not in {
        "microsoft/Phi-4-mini-instruct",  # no provider hosts it; variant does not exist
    }


@_needs_space
@pytest.mark.live_models
def test_space_chat_model_is_actually_callable():
    """Catalog presence is not enough — the model must be servable for this account."""
    from huggingface_hub import InferenceClient

    model = _space_chat_model()
    client = InferenceClient()
    response = client.chat_completion(
        messages=[{"role": "user", "content": "ping"}], model=model, max_tokens=5
    )
    content = response.choices[0].message.content
    assert content is not None, (
        f"{model} returned empty content — thinking models do this and break "
        "the JSON-only extraction prompt"
    )
