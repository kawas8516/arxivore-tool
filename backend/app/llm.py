"""Single entry point for all LLM calls, with multi-model failover.

Every pipeline stage (rerank, extract, synthesize) goes through `complete()`.
It talks to OpenRouter (OpenAI-compatible) and gives us resilience against the
free tier's per-model rate limits:

* **Native fallback** — each request carries OpenRouter's `models[]` array, so a
  rate-limited model is skipped *inside a single request* before any output
  tokens are generated (token-optimal).
* **Cooldown memory** — when the whole pool comes back 429, every attempted
  model is parked for a cooldown window so the *next* call doesn't waste a
  round-trip on a model we already know is limited.

Pool ordering is strongest-first and position 0 is the model used today, so a
healthy request is identical to before — failover only ever *descends* the list
under rate-limit pressure.

Pools come from config as a comma-separated list, or the literal ``auto`` to
discover free models from OpenRouter's catalog at runtime (gated server-side by
the account's allowed-models list).
"""

import logging
import threading
import time

import httpx
from openai import APIStatusError, OpenAI, RateLimitError

from app.config import get_settings

logger = logging.getLogger(__name__)


class AllModelsRateLimited(Exception):
    """Raised when every model in a pool is rate-limited / unavailable."""


def strip_fences(raw: str) -> str:
    """Remove a wrapping markdown code fence if the model added one."""
    if raw.startswith("```"):
        lines = raw.splitlines()
        inner = [l for l in lines[1:] if l.strip() != "```"]
        return "\n".join(inner).strip()
    return raw


# --- shared client (created once; reused across stages and threads) ----------
_client: OpenAI | None = None
_client_lock = threading.Lock()


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                settings = get_settings()
                _client = OpenAI(
                    api_key=settings.llm_api_key,
                    base_url=settings.llm_base_url,
                    max_retries=2,
                )
    return _client


# --- per-model cooldown registry (thread-safe; extract runs concurrently) ----
_cooldowns: dict[str, float] = {}
_cooldown_lock = threading.Lock()


def _is_cooling(model: str) -> bool:
    with _cooldown_lock:
        until = _cooldowns.get(model)
        if until is None:
            return False
        if time.monotonic() >= until:
            del _cooldowns[model]
            return False
        return True


def _cool(models: list[str], seconds: float) -> None:
    until = time.monotonic() + seconds
    with _cooldown_lock:
        for m in models:
            _cooldowns[m] = until


def _retry_after_seconds(exc: RateLimitError, default: float) -> float:
    """Best-effort parse of the Retry-After header; fall back to the default."""
    try:
        raw = exc.response.headers.get("retry-after")
    except Exception:
        raw = None
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass  # HTTP-date form — not worth parsing; use default
    return default


# --- model pool resolution ---------------------------------------------------
# discover_free_models() result is cached for llm_models_cache_ttl seconds so the
# `auto` mode picks up OpenRouter changes without a restart or per-call fetch.
_catalog_cache: dict | None = None
_catalog_expires: float = 0.0
_catalog_lock = threading.Lock()

_SYNTHESIS_MIN_CONTEXT = 131072  # synthesis reads all retained papers at once

# Free models that aren't usable as chat/JSON producers (rerankers, embeddings,
# audio, content-safety, vision-only). Matched as substrings of the model id.
_NON_CHAT_HINTS = (
    "rerank",
    "embed",
    "guard",
    "safety",
    "moderation",
    "lyria",
    "whisper",
    "tts",
)


def _is_free(model: dict) -> bool:
    pricing = model.get("pricing") or {}
    prompt = str(pricing.get("prompt", "1"))
    completion = str(pricing.get("completion", "1"))
    return prompt in ("0", "0.0") and completion in ("0", "0.0")


def _is_text_chat(model: dict) -> bool:
    mid = model.get("id", "").lower()
    if any(hint in mid for hint in _NON_CHAT_HINTS):
        return False
    arch = model.get("architecture") or {}
    inputs = arch.get("input_modalities") or []
    outputs = arch.get("output_modalities") or []
    modality = arch.get("modality") or ""
    # Newer schema lists modalities; older one uses a "text->text" string.
    if outputs:
        return "text" in outputs and "text" in (inputs or ["text"])
    if modality:
        return modality.endswith("text") and modality.startswith("text")
    return True  # unknown schema — assume usable rather than over-filter


def discover_free_models() -> dict[str, list[str]]:
    """Fetch OpenRouter's catalog and build two ordered free-model pools.

    Returns ``{"synthesis": [...], "rerank": [...]}`` ordered context-length
    desc. Cached for the configured TTL. On any failure, falls back to the
    curated lists in config so the app never hard-fails on a flaky catalog call.
    """
    global _catalog_cache, _catalog_expires
    now = time.monotonic()
    with _catalog_lock:
        if _catalog_cache is not None and now < _catalog_expires:
            return _catalog_cache

    settings = get_settings()
    try:
        resp = httpx.get(
            f"{settings.llm_base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
    except Exception:
        logger.warning("auto model discovery failed; using curated defaults", exc_info=True)
        return {
            "synthesis": _split_csv(settings.llm_synthesis_models),
            "rerank": _split_csv(settings.llm_rerank_models),
        }

    chat = [m for m in data if _is_free(m) and _is_text_chat(m)]
    chat.sort(key=lambda m: -(m.get("context_length") or 0))
    ids = [m["id"] for m in chat]
    pools = {
        "synthesis": [
            m["id"] for m in chat if (m.get("context_length") or 0) >= _SYNTHESIS_MIN_CONTEXT
        ]
        or ids,
        "rerank": ids,
    }
    logger.info(
        "discovered free models: synthesis=%d rerank=%d",
        len(pools["synthesis"]),
        len(pools["rerank"]),
    )
    with _catalog_lock:
        _catalog_cache = pools
        _catalog_expires = time.monotonic() + settings.llm_models_cache_ttl
    return pools


def _split_csv(spec: str) -> list[str]:
    return [s.strip() for s in spec.split(",") if s.strip()]


def resolve_pool(spec: str, kind: str) -> list[str]:
    """Turn a config spec into an ordered model list.

    ``kind`` is "synthesis" or "rerank"; only used when ``spec`` is "auto".
    """
    if spec.strip().lower() == "auto":
        return discover_free_models()[kind]
    return _split_csv(spec)


# --- the one call everyone uses ----------------------------------------------
def complete(messages: list[dict], *, pool: list[str], max_tokens: int) -> str:
    """Run a chat completion against the pool, failing over on rate limits.

    Returns the message content string. Raises ``AllModelsRateLimited`` when the
    whole pool is unavailable, or the underlying error for non-rate-limit
    failures.
    """
    if not pool:
        raise AllModelsRateLimited("model pool is empty")

    settings = get_settings()
    client = _get_client()

    # Skip models we already know are cooling; if all are cooling, try the full
    # pool anyway (cooldowns may have just lifted, or it's our only shot).
    available = [m for m in pool if not _is_cooling(m)] or list(pool)

    try:
        # Native fallback: model=available[0] is primary, the rest are tried
        # in-order within this single request before any tokens are produced.
        # OpenRouter caps the fallback array at 3 models per request.
        response = client.chat.completions.create(
            model=available[0],
            max_tokens=max_tokens,
            messages=messages,
            extra_body={"models": available[:3]},
        )
    except RateLimitError as exc:
        # OpenRouter exhausted the whole list and still hit a limit — park them.
        cooldown = _retry_after_seconds(exc, settings.llm_cooldown_seconds)
        _cool(available, cooldown)
        logger.warning(
            "all %d models rate-limited; cooling for %.0fs", len(available), cooldown
        )
        raise AllModelsRateLimited(
            f"all {len(available)} models rate-limited"
        ) from exc
    except APIStatusError as exc:
        if exc.status_code and 500 <= exc.status_code < 600:
            raise AllModelsRateLimited("upstream model error") from exc
        raise

    content = response.choices[0].message.content if response.choices else None
    if not content:
        raise ValueError("model returned empty content")
    used = getattr(response, "model", None) or available[0]
    if used != available[0]:
        # OpenRouter's inner fallback served from a different model — the primary
        # is rate-limited at their end. Cool it now so subsequent calls in this
        # run skip it directly rather than waiting for OpenRouter to reroute again.
        _cool([available[0]], settings.llm_cooldown_seconds)
        logger.info("failed over to model=%s; cooling primary=%s", used, available[0])
    return content
