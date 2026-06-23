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


class Cluster(BaseModel):
    name: str
    summary: str
    arxiv_ids: list[str]  # member papers, by arxiv_id


class Relationship(BaseModel):
    from_cluster: str  # cluster name
    to_cluster: str  # cluster name
    kind: str  # e.g. builds-on, alternative-to, complements
    description: str


class Landscape(BaseModel):
    clusters: list[Cluster] = []
    relationships: list[Relationship] = []
    tensions: list[str] = []
    open_problems: list[str] = []


class SearchResponse(BaseModel):
    topic: str
    candidates_retrieved: int
    papers_returned: int
    papers: list[Paper]
    landscape: Landscape | None = None
    retrieve_ms: int
    rerank_ms: int
    extract_ms: int = 0
    extract_errors: int = 0
    synthesize_ms: int = 0
