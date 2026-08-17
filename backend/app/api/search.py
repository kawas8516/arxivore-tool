import logging
import threading
import time
from collections import deque

from fastapi import APIRouter, HTTPException, Request

from app import budget
from app.config import get_settings
from app.models import SearchRequest, SearchResponse
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


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest, http_request: Request) -> SearchResponse:
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
    try:
        return run_pipeline(request.topic)
    except PipelineError as exc:
        raise HTTPException(status_code=502, detail=exc.message) from exc
    finally:
        _run_semaphore.release()
