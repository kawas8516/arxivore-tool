from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=300)

    @field_validator("topic")
    @classmethod
    def strip_and_clean(cls, v: str) -> str:
        # Strip control chars that could interfere with prompt construction
        cleaned = "".join(c for c in v if c.isprintable())
        stripped = cleaned.strip()
        if not stripped:
            raise ValueError("topic must not be empty after stripping")
        return stripped


class Author(BaseModel):
    name: str


class Paper(BaseModel):
    arxiv_id: str
    title: str
    abstract: str
    authors: list[Author]
    categories: list[str]
    published: str  # ISO 8601 date
    url: str
    # rerank
    relevance_score: float | None = None
    relevance_rationale: str | None = None
    # extract
    problem: str | None = None
    method: str | None = None
    results: str | None = None
    contribution: str | None = None
    extract_status: str = "pending"  # pending | done | error


# --- LLM output contracts -------------------------------------------------
# Every model response is validated against one of these before it touches a
# Paper. Without them a garbage response yields empty strings marked "done",
# which is worse than a visible error.


class ExpandOut(BaseModel):
    """Stage 0 — arXiv-syntax queries generated from a plain-English topic."""

    queries: list[str] = Field(min_length=1)


class RerankItem(BaseModel):
    arxiv_id: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class RerankOut(BaseModel):
    """Scores wrapped in an object — JSON mode requires an object at the root."""

    items: list[RerankItem]


class ExtractionOut(BaseModel):
    problem: str = Field(min_length=1)
    method: str = Field(min_length=1)
    results: str = Field(min_length=1)
    contribution: str = Field(min_length=1)


class Cluster(BaseModel):
    name: str = Field(min_length=1)
    summary: str
    arxiv_ids: list[str]  # member papers, by arxiv_id


class Relationship(BaseModel):
    from_cluster: str  # cluster name
    to_cluster: str  # cluster name
    kind: str  # e.g. builds-on, alternative-to, complements
    description: str


class Landscape(BaseModel):
    # clusters is required and non-empty: a landscape with no clusters is a
    # failed synthesis, and defaulting it to [] let a `{}` response validate as
    # success and render as a blank page with no error logged.
    clusters: list[Cluster] = Field(min_length=1)
    relationships: list[Relationship] = []
    tensions: list[str] = []
    open_problems: list[str] = []


class SearchResponse(BaseModel):
    topic: str
    candidates_retrieved: int
    papers_returned: int
    papers: list[Paper]
    landscape: Landscape | None = None
    queries: list[str] = []  # arXiv queries actually issued (stage 0 output)
    retrieve_ms: int
    rerank_ms: int
    expand_ms: int = 0
    extract_ms: int = 0
    extract_errors: int = 0
    synthesize_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
