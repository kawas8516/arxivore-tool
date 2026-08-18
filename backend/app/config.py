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
    # Model IDs must exist in the provider catalog. A retired ID fails the whole
    # stage at runtime, so verify against GET {llm_base_url}/models before
    # changing these (see tests/test_config.py).
    llm_synthesis_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    llm_rerank_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    # Extraction is per-paper and accuracy-sensitive, so it gets its own setting
    # rather than reusing the rerank model.
    llm_extract_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    # Query expansion is a cheap single call; reuse the rerank-tier model.
    llm_expand_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    # On a 429 from the primary model, retry once against this model instead of
    # exhausting client-side retries against a daily quota that won't recover
    # mid-run. Empty disables fallback (raises RateLimitExceeded as before).
    llm_fallback_model: str = ""

    max_candidates: int = 50
    max_retained_papers: int = 18
    max_concurrent_runs: int = 3
    daily_token_budget: int = 2_000_000
    rate_limit_per_minute: int = 10  # per-IP cap on /api/search
    # Parallel extraction calls. Kept low because free-tier providers rate-limit
    # aggressively — raising this into a 429 wall lowers completion, not latency.
    extract_concurrency: int = 3

    arxiv_page_size: int = 50
    # Number of arXiv queries to generate from one plain-English topic.
    expand_queries: int = 5
    # Optional arXiv category filter, comma-separated (e.g. "cs.LG,cs.CL").
    # Empty means no filter — a narrow default would silently drop papers from
    # adjacent categories (cs.RO, cs.CV, stat.ML).
    arxiv_categories: str = ""

    backend_host: str = "127.0.0.1"
    backend_port: int = 8000
    # The UI is served same-origin by this app; these origins only matter for
    # cross-origin API clients. Keep locked to known hosts (no wildcard).
    cors_allow_origins: str = "http://127.0.0.1:8000,http://localhost:8000"

    database_url: str = "sqlite:///./data/app.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()
