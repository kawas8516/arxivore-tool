"""Reading map endpoints — PRD.md section 9.

These are read/update-only against storage.py; no LLM or arXiv calls happen
here, so they don't need the /api/search rate limit, concurrency cap, or token
budget guard.
"""

import json
import logging
import time
from collections.abc import Generator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app import run_status, storage
from app.models import Landscape, Paper

logger = logging.getLogger(__name__)
router = APIRouter()

# How often the stream re-checks run_status for changes. run_status.get() is an
# in-memory dict read under a lock — cheap enough that a short poll interval
# costs nothing measurable at this scale; no need for condition variables.
_POLL_INTERVAL_SECONDS = 0.3
_TERMINAL_STATES = ("COMPLETE", "FAILED")


class RunSummary(BaseModel):
    run_id: str
    topic: str
    created_at: str
    paper_count: int


class RunDetail(BaseModel):
    run_id: str
    topic: str
    queries: list[str]
    created_at: str
    landscape: Landscape | None
    papers: list[Paper]
    prompt_tokens: int
    completion_tokens: int
    retrieve_ms: int
    rerank_ms: int
    extract_ms: int
    synthesize_ms: int


class ReadStatusUpdate(BaseModel):
    read_status: str


def _get_run_or_404(run_id: str) -> dict:
    run = storage.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/runs", response_model=list[RunSummary])
def list_runs() -> list[RunSummary]:
    """All prior runs, newest first — the reading map's run list (PRD FR10)."""
    return [RunSummary(**row) for row in storage.list_runs()]


@router.get("/runs/{run_id}", response_model=RunDetail)
def get_run(run_id: str) -> RunDetail:
    return RunDetail(**_get_run_or_404(run_id))


@router.get("/runs/{run_id}/papers", response_model=list[Paper])
def get_run_papers(run_id: str) -> list[Paper]:
    return _get_run_or_404(run_id)["papers"]


@router.get("/runs/{run_id}/landscape", response_model=Landscape | None)
def get_run_landscape(run_id: str) -> Landscape | None:
    return _get_run_or_404(run_id)["landscape"]


def _sse_event(snapshot: dict) -> str:
    return f"data: {json.dumps(snapshot)}\n\n"


def _stream_events(run_id: str) -> Generator[str, None, None]:
    last_seq = -1
    while True:
        snapshot = run_status.get(run_id)
        if snapshot is None:
            # Run finished and was swept, or (shouldn't happen — checked
            # before streaming starts) never existed. Either way, stop.
            break
        if snapshot["seq"] != last_seq:
            last_seq = snapshot["seq"]
            yield _sse_event(snapshot)
        if snapshot["state"] in _TERMINAL_STATES:
            break
        time.sleep(_POLL_INTERVAL_SECONDS)


@router.get("/runs/{run_id}/stream")
def stream_run(run_id: str) -> StreamingResponse:
    """Live stage-by-stage progress, matching ARCHITECTURE.md section 3's state
    machine (QUEUED -> RETRIEVING -> RERANKING -> EXTRACTING -> SYNTHESIZING ->
    COMPLETE | FAILED). Sourced from run_status, not storage — a run is only
    persisted once it finishes, so this is the only way to see it in progress.
    """
    if run_status.get(run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return StreamingResponse(_stream_events(run_id), media_type="text/event-stream")


@router.patch("/papers/{arxiv_id}")
def update_paper_read_status(arxiv_id: str, body: ReadStatusUpdate) -> dict:
    """Mark a paper read/to_read/skipped (PRD FR11). Global to the paper, not
    scoped to the run that surfaced it — see storage.py's module docstring for
    why read_status lives on `papers` rather than `run_papers`."""
    try:
        found = storage.update_read_status(arxiv_id, body.read_status)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not found:
        raise HTTPException(status_code=404, detail="Paper not found")
    return {"arxiv_id": arxiv_id, "read_status": body.read_status}
