from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=300)

    @field_validator("topic")
    @classmethod
    def strip_and_clean(cls, v: str) -> str:
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
    published: str
    url: str
    relevance_score: float | None = None
    relevance_rationale: str | None = None
    problem: str | None = None
    method: str | None = None
    results: str | None = None
    contribution: str | None = None
    extract_status: str = "pending"


class Cluster(BaseModel):
    name: str
    summary: str
    arxiv_ids: list[str]


class Relationship(BaseModel):
    from_cluster: str
    to_cluster: str
    kind: str
    description: str


class Landscape(BaseModel):
    clusters: list[Cluster] = []
    relationships: list[Relationship] = []
    tensions: list[str] = []
    open_problems: list[str] = []
