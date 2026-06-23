import logging
import threading
import time
from collections import defaultdict, deque

from fastapi import APIRouter, HTTPException, Request

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
_ip_hits: dict[str, deque] = defaultdict(deque)
_ip_lock = threading.Lock()


def _check_rate_limit(client_ip: str) -> None:
    limit = _settings.rate_limit_per_minute
    now = time.monotonic()
    with _ip_lock:
        hits = _ip_hits[client_ip]
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
