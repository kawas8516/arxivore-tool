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

    # Ordered failover pools, strongest-first: comma-separated model ids, or the
    # literal "auto" to discover free models from OpenRouter's catalog at
    # runtime. Position 0 is what serves a healthy request, so accuracy is
    # unchanged until a model is rate-limited and app/llm.py descends the list.
    #
    # Model IDs must exist in the provider catalog. A retired ID at position 0
    # costs a failover rather than the stage, but verify against
    # GET {llm_base_url}/models anyway (see tests/test_config.py).
    llm_synthesis_models: str = (
        "nvidia/nemotron-3-ultra-550b-a55b:free,"
        "nvidia/nemotron-3-super-120b-a12b:free,"
        "google/gemma-4-31b-it:free,"
        "minimax/minimax-m3:free"
    )
    llm_rerank_models: str = (
        "google/gemma-4-31b-it:free,"
        "nvidia/nemotron-3-super-120b-a12b:free,"
        "google/gemma-4-26b-a4b-it:free,"
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
    )
    # When a model returns 429, skip it for this long (used when the upstream
    # Retry-After header is absent), and how long the "auto" catalog is cached.
    llm_cooldown_seconds: int = 60
    llm_models_cache_ttl: int = 3600

    # Legacy single-model vars — kept so older .env files still boot. Each is the
    # position-0 default of the corresponding pool above.
    llm_synthesis_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    llm_rerank_model: str = "google/gemma-4-31b-it:free"

    # Off by default: OpenRouter's /embeddings coverage is much narrower than
    # its chat-completions catalog, and support for a given model is unverified
    # until you confirm it against your own LLM_BASE_URL. Any failure (wrong
    # endpoint, unsupported model, network error) falls back to sending every
    # candidate straight to rerank unfiltered — enabling this can only ever cost
    # you latency on a bad config, never recall.
    embed_prefilter_enabled: bool = False
    llm_embedding_model: str = "openai/text-embedding-3-small"
    # Candidates kept after the prefilter, before the LLM rerank call sees them.
    embed_prefilter_keep: int = 20

    # Off by default: a PDF fetch+parse per paper adds real seconds of latency
    # and a network dependency on arxiv.org, and would otherwise make the
    # offline eval harness (evals/run.py --offline) attempt real HTTP calls for
    # its synthetic paper ids. Abstracts rarely contain hard numbers, so this
    # exists to give extraction's `results` field something to work with beyond
    # vague abstract language, for the handful of papers the run cares most
    # about.
    full_text_enabled: bool = False
    # Only the top N papers by rerank score get full-text treatment — fetching
    # and parsing a PDF for every retained paper isn't worth the latency.
    full_text_top_n: int = 5
    full_text_max_chars: int = 6000

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
