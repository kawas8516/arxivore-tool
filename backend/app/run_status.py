"""In-memory run-progress registry, sourcing the SSE stream and matching the
state machine documented in ARCHITECTURE.md section 3.

In-process only, like the rate limiter and budget ledger — fine for a
single-instance v1, would need a shared store (Redis, etc.) to scale
horizontally. Deliberately separate from storage.py: this tracks live,
in-flight progress for a run that may not exist in SQLite yet (a run is only
persisted once it finishes), while storage.py is the durable record of
completed runs.
"""

import copy
import logging
import threading
import time

logger = logging.getLogger(__name__)

# ARCHITECTURE.md section 3's state machine, verbatim.
STATES = ("QUEUED", "RETRIEVING", "RERANKING", "EXTRACTING", "SYNTHESIZING", "COMPLETE", "FAILED")
_TERMINAL_STATES = ("COMPLETE", "FAILED")

# Sweep entries this long after they finish, so a process that runs for days
# doesn't accumulate one entry per run forever.
_RETENTION_SECONDS = 3600.0

_lock = threading.Lock()
_runs: dict[str, dict] = {}


def _new_stage() -> dict:
    return {"status": "pending"}


def _sweep_locked(now: float) -> None:
    stale = [
        run_id
        for run_id, run in _runs.items()
        if run["state"] in _TERMINAL_STATES and now - run["updated_at"] > _RETENTION_SECONDS
    ]
    for run_id in stale:
        del _runs[run_id]
    if stale:
        logger.debug("run_status swept %d stale entries", len(stale))


def start(run_id: str, topic: str) -> None:
    """Register a new run as QUEUED. Call before the pipeline begins so an SSE
    client connecting immediately after POST /api/search sees something."""
    now = time.time()
    with _lock:
        _sweep_locked(now)
        _runs[run_id] = {
            "run_id": run_id,
            "topic": topic,
            "state": "QUEUED",
            "stages": {
                "retrieve": _new_stage(),
                "rerank": _new_stage(),
                "extract": _new_stage(),
                "synthesize": _new_stage(),
            },
            "error": None,
            "created_at": now,
            "updated_at": now,
            "seq": 0,
        }


def set_state(run_id: str, state: str) -> None:
    """Advance the run's overall state. A no-op if the run was never started
    (e.g. run_pipeline called directly, outside the API layer) or already
    finished — intentional, so callers don't need to check first."""
    with _lock:
        run = _runs.get(run_id)
        if run is None or run["state"] in _TERMINAL_STATES:
            return
        run["state"] = state
        run["updated_at"] = time.time()
        run["seq"] += 1


def set_stage(run_id: str, stage: str, **fields) -> None:
    """Update one stage's sub-fields (status, count, ms, done/total, ...)."""
    with _lock:
        run = _runs.get(run_id)
        if run is None or stage not in run["stages"]:
            return
        run["stages"][stage].update(fields)
        run["updated_at"] = time.time()
        run["seq"] += 1


def set_error(run_id: str, message: str) -> None:
    """Move the run to FAILED with an error message. Terminal — no further
    updates are accepted after this (see set_state/set_stage's guard)."""
    with _lock:
        run = _runs.get(run_id)
        if run is None:
            return
        run["state"] = "FAILED"
        run["error"] = message
        run["updated_at"] = time.time()
        run["seq"] += 1


def get(run_id: str) -> dict | None:
    """A deep-copied snapshot, safe to serialize without racing further updates."""
    with _lock:
        run = _runs.get(run_id)
        return copy.deepcopy(run) if run is not None else None
