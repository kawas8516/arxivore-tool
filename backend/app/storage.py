"""SQLite persistence for the reading map.

Dedup rule (resolved during planning): one row per paper, keyed on the
version-stripped arxiv_id (see pipeline/retrieve.py). A paper's extraction is
cached once and reused by every later run that retrieves it — this is where the
token savings actually land, not in the schema itself. A join table tags which
runs surfaced which papers, since relevance is topic-specific and can't live on
the shared papers row.

Deviation from the original schema sketch: read_status lives on `papers`, not
`run_papers`. PATCH /api/papers/{arxiv_id} only has an arxiv_id, no run_id, so a
per-run read_status would have no way to resolve which row to update — read
state is inherently global to a paper, not to the run that surfaced it.

Second deviation: embeddings live in their own table, not a column on `papers`.
The embedding prefilter (pipeline/embed.py) runs before rerank — before
extraction, before a paper has ever earned a `papers` row — so a column there
would mean either inserting a half-populated row early or losing the cache for
every candidate that gets filtered out before extraction. A standalone table
keyed on arxiv_id has neither problem.
"""

import array
import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.models import Author, Landscape, Paper, SearchResponse

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    authors_json TEXT NOT NULL,
    categories_json TEXT NOT NULL,
    published TEXT NOT NULL,
    url TEXT NOT NULL,
    problem TEXT,
    method TEXT,
    results TEXT,
    contribution TEXT,
    extract_status TEXT NOT NULL DEFAULT 'pending',
    read_status TEXT NOT NULL DEFAULT 'unread',
    first_seen_at TEXT NOT NULL,
    extracted_at TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    queries_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    landscape_json TEXT,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    retrieve_ms INTEGER NOT NULL DEFAULT 0,
    rerank_ms INTEGER NOT NULL DEFAULT 0,
    extract_ms INTEGER NOT NULL DEFAULT 0,
    synthesize_ms INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS run_papers (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    arxiv_id TEXT NOT NULL REFERENCES papers(arxiv_id),
    relevance_score REAL,
    relevance_rationale TEXT,
    PRIMARY KEY (run_id, arxiv_id)
);

CREATE INDEX IF NOT EXISTS idx_run_papers_arxiv_id ON run_papers(arxiv_id);

CREATE TABLE IF NOT EXISTS embeddings (
    arxiv_id TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    vector BLOB NOT NULL,
    created_at TEXT NOT NULL
);
"""

_VALID_READ_STATUSES = {"unread", "read", "to_read", "skipped"}

# One lock guards all writes. SQLite serializes writers regardless, and the
# app's write volume (one run every few seconds at most) makes a single lock
# simpler than a connection pool with no measurable cost.
_write_lock = threading.Lock()


def _db_path() -> Path:
    settings = get_settings()
    url = settings.database_url
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        raise ValueError(f"only sqlite:/// URLs are supported, got: {url!r}")
    raw_path = url[len(prefix) :]
    path = Path(raw_path)
    if not path.is_absolute():
        # Resolve relative to backend/, not the process cwd, so the database
        # lands in the same place regardless of where uvicorn was launched from.
        path = Path(__file__).parent.parent / path
    return path


@contextmanager
def _connect():
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they don't exist. Call once at startup."""
    with _write_lock, _connect() as conn:
        conn.executescript(_SCHEMA)
    logger.info("storage initialised at %s", _db_path())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_paper(row: sqlite3.Row) -> Paper:
    return Paper(
        arxiv_id=row["arxiv_id"],
        title=row["title"],
        abstract=row["abstract"],
        authors=[Author(**a) for a in json.loads(row["authors_json"])],
        categories=json.loads(row["categories_json"]),
        published=row["published"],
        url=row["url"],
        problem=row["problem"],
        method=row["method"],
        results=row["results"],
        contribution=row["contribution"],
        extract_status=row["extract_status"],
    )


def get_cached_extraction(arxiv_id: str) -> Paper | None:
    """Return a paper's cached extraction, or None on a miss or incomplete entry.

    Only a completed extraction is served from cache — a row can exist with
    extract_status='pending' if a paper was retrieved but never reached
    extraction in an earlier run, and that must not be mistaken for a
    successful cache hit.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM papers WHERE arxiv_id = ? AND extract_status = 'done'",
            (arxiv_id,),
        ).fetchone()
    return _row_to_paper(row) if row else None


def save_extraction(paper: Paper) -> None:
    """Upsert a paper's metadata and extraction.

    first_seen_at is preserved across upserts (COALESCE against the existing
    row); extracted_at is only set when the incoming status is 'done'.
    """
    now = _now()
    with _write_lock, _connect() as conn:
        conn.execute(
            """
            INSERT INTO papers (
                arxiv_id, title, abstract, authors_json, categories_json,
                published, url, problem, method, results, contribution,
                extract_status, first_seen_at, extracted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(arxiv_id) DO UPDATE SET
                title=excluded.title,
                abstract=excluded.abstract,
                authors_json=excluded.authors_json,
                categories_json=excluded.categories_json,
                published=excluded.published,
                url=excluded.url,
                problem=excluded.problem,
                method=excluded.method,
                results=excluded.results,
                contribution=excluded.contribution,
                extract_status=excluded.extract_status,
                extracted_at=excluded.extracted_at
            """,
            (
                paper.arxiv_id,
                paper.title,
                paper.abstract,
                json.dumps([a.model_dump() for a in paper.authors]),
                json.dumps(paper.categories),
                paper.published,
                paper.url,
                paper.problem,
                paper.method,
                paper.results,
                paper.contribution,
                paper.extract_status,
                now,
                now if paper.extract_status == "done" else None,
            ),
        )


def update_read_status(arxiv_id: str, status: str) -> bool:
    """Set a paper's read status. Returns False if the paper doesn't exist."""
    if status not in _VALID_READ_STATUSES:
        raise ValueError(f"invalid read_status {status!r}, must be one of {_VALID_READ_STATUSES}")
    with _write_lock, _connect() as conn:
        cursor = conn.execute(
            "UPDATE papers SET read_status = ? WHERE arxiv_id = ?", (status, arxiv_id)
        )
        return cursor.rowcount > 0


def save_run(run_id: str, response: SearchResponse) -> None:
    """Persist a run's outcome and tag which papers it surfaced.

    Paper rows must already exist (extract_papers / save_extraction runs first
    in the pipeline) — this only records the run-specific relevance scores.
    """
    with _write_lock, _connect() as conn:
        conn.execute(
            """
            INSERT INTO runs (
                run_id, topic, queries_json, created_at, landscape_json,
                prompt_tokens, completion_tokens,
                retrieve_ms, rerank_ms, extract_ms, synthesize_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO NOTHING
            """,
            (
                run_id,
                response.topic,
                json.dumps(response.queries),
                _now(),
                response.landscape.model_dump_json() if response.landscape else None,
                response.prompt_tokens,
                response.completion_tokens,
                response.retrieve_ms,
                response.rerank_ms,
                response.extract_ms,
                response.synthesize_ms,
            ),
        )
        for paper in response.papers:
            conn.execute(
                """
                INSERT INTO run_papers (run_id, arxiv_id, relevance_score, relevance_rationale)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(run_id, arxiv_id) DO UPDATE SET
                    relevance_score=excluded.relevance_score,
                    relevance_rationale=excluded.relevance_rationale
                """,
                (run_id, paper.arxiv_id, paper.relevance_score, paper.relevance_rationale),
            )


def get_run(run_id: str) -> dict | None:
    """Return a run's stored state: metadata, landscape, and its papers."""
    with _connect() as conn:
        run_row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if run_row is None:
            return None
        paper_rows = conn.execute(
            """
            SELECT p.*, rp.relevance_score, rp.relevance_rationale
            FROM run_papers rp JOIN papers p ON p.arxiv_id = rp.arxiv_id
            WHERE rp.run_id = ?
            ORDER BY rp.relevance_score DESC NULLS LAST
            """,
            (run_id,),
        ).fetchall()

    papers = []
    for row in paper_rows:
        paper = _row_to_paper(row)
        paper.relevance_score = row["relevance_score"]
        paper.relevance_rationale = row["relevance_rationale"]
        papers.append(paper)

    return {
        "run_id": run_row["run_id"],
        "topic": run_row["topic"],
        "queries": json.loads(run_row["queries_json"]),
        "created_at": run_row["created_at"],
        "landscape": Landscape.model_validate_json(run_row["landscape_json"])
        if run_row["landscape_json"]
        else None,
        "papers": papers,
        "prompt_tokens": run_row["prompt_tokens"],
        "completion_tokens": run_row["completion_tokens"],
        "retrieve_ms": run_row["retrieve_ms"],
        "rerank_ms": run_row["rerank_ms"],
        "extract_ms": run_row["extract_ms"],
        "synthesize_ms": run_row["synthesize_ms"],
    }


def list_runs() -> list[dict]:
    """Summaries of every run, newest first — the reading map's run list."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT r.run_id, r.topic, r.created_at, COUNT(rp.arxiv_id) AS paper_count
            FROM runs r LEFT JOIN run_papers rp ON rp.run_id = r.run_id
            GROUP BY r.run_id
            ORDER BY r.created_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_embedding(arxiv_id: str, model: str) -> list[float] | None:
    """A cached embedding, or None on a miss. Keyed on (arxiv_id, model): a
    changed LLM_EMBEDDING_MODEL is a cache miss, not a mismatch — old vectors
    aren't comparable across embedding models."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT vector FROM embeddings WHERE arxiv_id = ? AND model = ?",
            (arxiv_id, model),
        ).fetchone()
    if row is None:
        return None
    vector = array.array("f")
    vector.frombytes(row["vector"])
    return list(vector)


def save_embedding(arxiv_id: str, model: str, vector: list[float]) -> None:
    packed = array.array("f", vector).tobytes()
    with _write_lock, _connect() as conn:
        conn.execute(
            """
            INSERT INTO embeddings (arxiv_id, model, vector, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(arxiv_id) DO UPDATE SET
                model=excluded.model, vector=excluded.vector, created_at=excluded.created_at
            """,
            (arxiv_id, model, packed, _now()),
        )
