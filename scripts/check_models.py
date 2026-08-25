#!/usr/bin/env python
"""Check every model id this project can dial, against both providers.

Two independent LLM surfaces ship from this repo and each can rot silently:

* **OpenRouter** powers `backend/` (rerank, extract, expand, synthesize) through
  ordered failover pools. A withdrawn id there returns 400, not 429, so pool
  failover does not rescue it — the stage dies.
* **The HF Inference API** powers the root Gradio Space. `microsoft/Phi-4-mini-instruct`
  was configured for months and was never served by any provider: the Space
  built green, ran green, and returned empty Papers and Landscape tabs. Catalog
  presence is not enough — a model must be servable *for this account*, so this
  script actually calls the HF model rather than just looking it up.

Usage:
    python scripts/check_models.py              # check both providers
    python scripts/check_models.py --openrouter # backend pools only
    python scripts/check_models.py --hf         # Space model only
    python scripts/check_models.py --suggest    # list live replacements for anything dead

Exits non-zero if any configured model is unavailable, so it can gate a release.

Credentials, both optional-but-recommended:
    LLM_API_KEY   OpenRouter key, read from .env (the catalog is public, but an
                  authenticated view reflects your account's allowed models)
    HF token      from `hf auth login` / ~/.cache/huggingface/token / HF_TOKEN
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

_OPENROUTER_CATALOG = "https://openrouter.ai/api/v1/models"
_HF_ROUTER_CATALOG = "https://router.huggingface.co/v1/models"

# Free models that are not usable as chat/JSON producers, for --suggest.
_NON_CHAT_HINTS = ("rerank", "embed", "guard", "safety", "moderation", "lyria", "whisper", "tts")

_OK = "  OK   "
_DEAD = "  DEAD "
_WARN = "  WARN "


# --------------------------------------------------------------------------
# reading what the project is configured to use
# --------------------------------------------------------------------------
def _load_env() -> None:
    """Populate os.environ from .env without requiring pydantic-settings."""
    env = _ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _split_csv(spec: str) -> list[str]:
    return [s.strip() for s in spec.split(",") if s.strip()]


def openrouter_configured() -> dict[str, list[str]]:
    """Backend pools, from Settings defaults overlaid with any .env override.

    Read from the config source rather than imported, so this script keeps
    working when backend deps are not installed.
    """
    src = (_ROOT / "backend" / "app" / "config.py").read_text(encoding="utf-8")
    out: dict[str, list[str]] = {}
    for field in ("llm_synthesis_models", "llm_rerank_models"):
        env_override = os.environ.get(field.upper())
        if env_override:
            out[field] = _split_csv(env_override)
            continue
        # Defaults are written as adjacent string literals inside parentheses.
        m = re.search(rf"{field}: str = \((.*?)\)", src, re.S)
        if m:
            out[field] = _split_csv("".join(re.findall(r'"([^"]*)"', m.group(1))))
    for field in ("llm_synthesis_model", "llm_rerank_model"):
        env_override = os.environ.get(field.upper())
        if env_override:
            out[field] = [env_override]
            continue
        m = re.search(rf'{field}: str = "([^"]+)"', src)
        if m:
            out[field] = [m.group(1)]
    return out


def hf_configured() -> dict[str, str]:
    """The Space's models: the chat model in llm.py and the local cross-encoder."""
    out = {}
    m = re.search(r'_MODEL = "([^"]+)"', (_ROOT / "llm.py").read_text(encoding="utf-8"))
    if m:
        out["llm.py::_MODEL"] = m.group(1)
    m = re.search(r'CrossEncoder\("([^"]+)"\)', (_ROOT / "pipeline" / "rerank.py").read_text(encoding="utf-8"))
    if m:
        out["pipeline/rerank.py::_RERANKER"] = m.group(1)
    return out


# --------------------------------------------------------------------------
# provider catalogs
# --------------------------------------------------------------------------
def _get_json(url: str, token: str | None = None, timeout: int = 30):
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as response:  # noqa: S310
        return json.load(response)


def _hf_token() -> str | None:
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    path = Path.home() / ".cache" / "huggingface" / "token"
    if path.exists():
        return path.read_text(encoding="utf-8").strip() or None
    return None


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------
def check_openrouter(suggest: bool) -> int:
    print("== OpenRouter (backend pools) " + "=" * 42)
    configured = openrouter_configured()
    if not configured:
        print("  could not read any model config from backend/app/config.py")
        return 1

    try:
        data = _get_json(_OPENROUTER_CATALOG, os.environ.get("LLM_API_KEY"))["data"]
    except Exception as exc:  # network, auth, outage
        print(f"  {_WARN} could not fetch the catalog: {type(exc).__name__}: {exc}")
        return 1

    catalog = {m["id"] for m in data}
    print(f"  catalog: {len(catalog)} models\n")

    dead = []
    for field, ids in configured.items():
        print(f"  {field}:")
        for position, mid in enumerate(ids):
            alive = mid in catalog
            mark = _OK if alive else _DEAD
            note = "" if alive else "  <-- not in catalog; returns 400, NOT 429, so failover will not cover it"
            print(f"  {mark} [{position}] {mid}{note}")
            if not alive:
                dead.append((field, position, mid))
        print()

    if dead and suggest:
        print("  -- live free chat models, longest context first --")
        for mid, ctx in _openrouter_free_chat(data)[:10]:
            print(f"       {ctx:>9,}  {mid}")
        print()

    if dead:
        primary = [d for d in dead if d[1] == 0]
        print(f"  {len(dead)} dead id(s).")
        if primary:
            print(f"  {len(primary)} at POSITION 0 — every healthy request hits a dead model first.")
        return 1
    print("  all configured OpenRouter models are in the catalog.")
    return 0


def _openrouter_free_chat(data) -> list[tuple[str, int]]:
    def is_free(m):
        p = m.get("pricing") or {}
        return str(p.get("prompt", "1")) in ("0", "0.0") and str(p.get("completion", "1")) in ("0", "0.0")

    out = [
        (m["id"], m.get("context_length") or 0)
        for m in data
        if is_free(m) and not any(h in m["id"].lower() for h in _NON_CHAT_HINTS)
    ]
    return sorted(out, key=lambda t: -t[1])


def check_hf(suggest: bool) -> int:
    print("== Hugging Face (Gradio Space) " + "=" * 41)
    configured = hf_configured()
    token = _hf_token()
    print(f"  token: {'found' if token else 'NONE — using anonymous access'}\n")

    failures = 0
    for where, mid in configured.items():
        if "CrossEncoder" in where or "RERANKER" in where:
            # Downloaded and run locally on the Space's CPU, never served over an
            # API, so catalog membership is what matters, not an inference call.
            ok = _hf_repo_exists(mid)
            print(f"  {_OK if ok else _DEAD} {where}\n         {mid}  (local cross-encoder)")
            failures += 0 if ok else 1
            continue

        ok, detail = _hf_chat_reachable(mid, token)
        print(f"  {_OK if ok else _DEAD} {where}\n         {mid}")
        if not ok:
            print(f"         -> {detail}")
            failures += 1

    print()
    if failures and suggest:
        print("  -- models this account can actually call --")
        for mid in _hf_served_chat_models()[:15]:
            print(f"       {mid}")
        print()

    if failures:
        print(f"  {failures} unavailable. The Space will still BUILD and RUN — the")
        print("  failure only shows as empty Papers/Landscape tabs at runtime.")
        return 1
    print("  all configured HF models are reachable.")
    return 0


def _hf_repo_exists(repo_id: str) -> bool:
    try:
        _get_json(f"https://huggingface.co/api/models/{repo_id}")
        return True
    except Exception:
        return False


def _hf_chat_reachable(mid: str, token: str | None) -> tuple[bool, str]:
    """Actually call the model — being in the catalog does not mean it is served."""
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        return False, "huggingface_hub not installed (pip install -r requirements.txt)"
    try:
        client = InferenceClient(token=token)
        client.chat_completion(
            messages=[{"role": "user", "content": "ping"}], model=mid, max_tokens=5
        )
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc).splitlines()[-1][:160]}"


def _hf_served_chat_models() -> list[str]:
    try:
        data = _get_json(_HF_ROUTER_CATALOG)
        models = data.get("data", data)
        return [
            m["id"]
            for m in models
            if not any(h in m.get("id", "").lower() for h in _NON_CHAT_HINTS)
        ]
    except Exception:
        return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--openrouter", action="store_true", help="check backend pools only")
    parser.add_argument("--hf", action="store_true", help="check the Space's models only")
    parser.add_argument("--suggest", action="store_true", help="list live replacements")
    args = parser.parse_args()

    _load_env()
    both = not (args.openrouter or args.hf)

    status = 0
    if both or args.openrouter:
        status |= check_openrouter(args.suggest)
        print()
    if both or args.hf:
        status |= check_hf(args.suggest)

    print()
    print("FAIL — update the ids above." if status else "OK — every configured model is available.")
    return status


if __name__ == "__main__":
    sys.exit(main())
