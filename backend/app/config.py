from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# .env lives at the project root (one level above backend/)
_ENV_FILE = Path(__file__).parent.parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    llm_api_key: str
    llm_base_url: str = "https://openrouter.ai/api/v1"
    # Legacy single-model vars — kept so older .env files still boot. Each is the
    # position-0 default of the corresponding pool below.
    llm_synthesis_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    llm_rerank_model: str = "meta-llama/llama-3.3-70b-instruct:free"

    # Ordered failover pools (strongest-first). Comma-separated model ids, or the
    # literal "auto" to discover free models from OpenRouter at runtime. Position 0
    # is what runs on a healthy request, so accuracy is unchanged until a model is
    # rate-limited and we descend the list. See app/llm.py.
    llm_synthesis_models: str = (
        "nvidia/nemotron-3-ultra-550b-a55b:free,"
        "nvidia/nemotron-3-super-120b-a12b:free,"
        "nousresearch/hermes-3-llama-3.1-405b:free,"
        "openai/gpt-oss-120b:free,"
        "openrouter/owl-alpha"
    )
    llm_rerank_models: str = (
        "meta-llama/llama-3.3-70b-instruct:free,"
        "openai/gpt-oss-120b:free,"
        "google/gemma-4-31b-it:free,"
        "nvidia/nemotron-3-nano-30b-a3b:free,"
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
    )
    # When a model returns 429, skip it for this long (fallback if no Retry-After).
    llm_cooldown_seconds: int = 60
    # How long to cache the discovered free-model catalog (auto mode).
    llm_models_cache_ttl: int = 3600

    max_candidates: int = 50
    max_retained_papers: int = 18
    max_concurrent_runs: int = 3
    daily_token_budget: int = 2_000_000
    rate_limit_per_minute: int = 10  # per-IP cap on /api/search

    arxiv_page_size: int = 50

    backend_host: str = "127.0.0.1"
    backend_port: int = 8000
    # The UI is served same-origin by this app; these origins only matter for
    # cross-origin API clients. Keep locked to known hosts (no wildcard).
    cors_allow_origins: str = "http://127.0.0.1:8000,http://localhost:8000"

    database_url: str = "sqlite:///./data/app.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()
