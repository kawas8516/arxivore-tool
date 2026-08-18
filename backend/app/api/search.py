import logging
import threading
import time
import uuid
from collections import deque
from datetime import date

from fastapi import APIRouter, HTTPException, Request

from app import budget, run_status
from app.config import get_settings
from app.models import SearchAccepted, SearchRequest
from app.service import run_pipeline, PipelineError

logger = logging.getLogger(__name__)
router = APIRouter()

_settings = get_settings()

# Concurrency cap (security.md 3.2): bound in-flight pipeline runs so a burst of
# requests can't fan out into unbounded expensive LLM work.
_run_semaphore = threading.BoundedSemaphore(_settings.max_concurrent_runs)

# Per-IP sliding-window rate limit (security.md 3.2). In-process only — fine for
# a single-instance v1 deployment; move to a shared store if horizontally scaled.
_RATE_WINDOW_SECONDS = 60.0
_SWEEP_EVERY_SECONDS = 300.0
_ip_hits: dict[str, deque] = {}
_ip_lock = threading.Lock()
_last_sweep = 0.0


def _sweep_locked(now: float) -> None:
    """Drop IPs with no hits inside the window. Caller must hold _ip_lock.

    Pruning only the requesting IP's own deque would still leak: a one-shot
    visitor's key is never touched again. This sweeps every IP periodically so the
    dict stays proportional to *active* clients, not to every IP ever seen.
    """
    global _last_sweep
    if now - _last_sweep < _SWEEP_EVERY_SECONDS:
        return
    _last_sweep = now
    stale = [ip for ip, hits in _ip_hits.items() if not hits or now - hits[-1] > _RATE_WINDOW_SECONDS]
    for ip in stale:
        del _ip_hits[ip]
    if stale:
        logger.debug("rate limiter swept %d stale ip entries", len(stale))


def _check_rate_limit(client_ip: str) -> None:
    limit = _settings.rate_limit_per_minute
    now = time.monotonic()
    with _ip_lock:
        _sweep_locked(now)
        hits = _ip_hits.setdefault(client_ip, deque())
        while hits and now - hits[0] > _RATE_WINDOW_SECONDS:
            hits.popleft()
        if len(hits) >= limit:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please wait a moment and try again.",
            )
        hits.append(now)


def _run_in_background(run_id: str, topic: str, published_after: date | None) -> None:
    """Runs the full pipeline off the request thread. Errors are recorded onto
    run_status (run_pipeline does this itself before raising PipelineError) —
    there is no HTTP response left to attach them to by the time this runs."""
    try:
        run_pipeline(topic, published_after, run_id=run_id)
        # run_pipeline's real implementation already calls
        # run_status.set_state(run_id, "COMPLETE") itself; set_state is
        # terminal-guarded so this is a no-op there. It's a safety net for
        # anything that substitutes run_pipeline without that instrumentation
        # (a test double, a future alternate implementation) — a run must never
        # get stuck at QUEUED/RETRIEVING/etc. forever just because the callee
        # forgot to mark completion.
        run_status.set_state(run_id, "COMPLETE")
    except PipelineError as exc:
        # Same reasoning as above, for the failure path: run_pipeline already
        # calls run_status.set_error() before raising this.
        run_status.set_error(run_id, exc.message)
    except Exception:
        logger.exception("unexpected pipeline crash run_id=%s", run_id)
        run_status.set_error(run_id, "Internal error")
    finally:
        _run_semaphore.release()


@router.post("/search", response_model=SearchAccepted)
def search(request: SearchRequest, http_request: Request) -> SearchAccepted:
    """Starts a pipeline run in the background and returns immediately.

    Poll GET /api/runs/{run_id} or subscribe to GET /api/runs/{run_id}/stream
    for progress and the final result — a full run can take minutes on
    free-tier models, and blocking the request for that long was the reason
    this changed (see IMPROVEMENT_PLAN.md P3-1).
    """
    client_ip = http_request.client.host if http_request.client else "unknown"
    _check_rate_limit(client_ip)

    # Spend ceiling (security.md 3.2). Checked before the run starts; a run
    # already in flight finishes, so the ceiling overshoots by at most one run.
    if budget.exhausted():
        logger.warning("daily token budget exhausted — rejecting search")
        raise HTTPException(
            status_code=429,
            detail="Daily usage limit reached. Please try again tomorrow.",
        )

    if not _run_semaphore.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="Server is busy with other searches. Please try again shortly.",
        )

    run_id = uuid.uuid4().hex
    run_status.start(run_id, request.topic)
    # A plain daemon thread, not FastAPI's BackgroundTasks: BackgroundTasks only
    # starts after the response is sent but still ties the task to this request's
    # lifecycle in ways that complicate testing and don't buy anything here — the
    # semaphore already bounds concurrency, so a thread per accepted run is fine
    # at this scale.
    threading.Thread(
        target=_run_in_background,
        args=(run_id, request.topic, request.published_after),
        daemon=True,
    ).start()

    return SearchAccepted(run_id=run_id, state="QUEUED")
