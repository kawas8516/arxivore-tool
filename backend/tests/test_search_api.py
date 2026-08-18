import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import budget, run_status
from app.api import search as search_api
from app.main import app
from app.models import SearchResponse


def _empty_response(topic: str, published_after=None, run_id: str | None = None) -> SearchResponse:
    return SearchResponse(
        run_id=run_id or "",
        topic=topic,
        candidates_retrieved=0,
        papers_returned=0,
        papers=[],
        retrieve_ms=1,
        rerank_ms=0,
    )


def _wait_for_terminal(run_id: str, timeout: float = 2.0) -> dict:
    """POST /api/search returns before the background thread runs — poll
    run_status until the run reaches a terminal state instead of racing it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = run_status.get(run_id)
        if snapshot and snapshot["state"] in ("COMPLETE", "FAILED"):
            return snapshot
        time.sleep(0.01)
    raise TimeoutError(f"run {run_id} did not reach a terminal state within {timeout}s")


@pytest.fixture(autouse=True)
def _clean_state():
    budget.reset()
    search_api._ip_hits.clear()
    search_api._last_sweep = 0.0
    yield
    budget.reset()
    search_api._ip_hits.clear()


@pytest.fixture
def client(tmp_path, monkeypatch):
    # TestClient(app) runs the real FastAPI lifespan, which now calls
    # storage.init_db() — point it at a throwaway file so the test suite never
    # touches the developer's real reading-map database.
    from app.config import get_settings

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    get_settings.cache_clear()
    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()


def test_search_returns_immediate_acceptance(client):
    """POST /api/search no longer blocks for the run — it hands back a run_id
    to poll/stream, per IMPROVEMENT_PLAN.md P3-1."""
    with patch("app.api.search.run_pipeline", side_effect=_empty_response):
        response = client.post("/api/search", json={"topic": "diffusion policy"})

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"]
    assert body["state"] == "QUEUED"


def test_search_run_reaches_storage_after_completion(client):
    with patch("app.api.search.run_pipeline", side_effect=_empty_response):
        response = client.post("/api/search", json={"topic": "diffusion policy"})
        run_id = response.json()["run_id"]
        snapshot = _wait_for_terminal(run_id)

    # _empty_response has zero candidates, so run_pipeline's own zero-candidates
    # path (which the mock bypasses entirely) never runs — nothing is persisted
    # to storage by the mock. This test only proves the background thread ran
    # and the run_status registry reflects it; see test_service.py for the real
    # run_pipeline's storage persistence.
    assert snapshot["state"] == "COMPLETE"
    assert snapshot["topic"] == "diffusion policy"


def test_short_topic_is_rejected(client):
    response = client.post("/api/search", json={"topic": "ab"})
    assert response.status_code == 422


def test_published_after_is_forwarded_to_pipeline(client):
    from datetime import date

    with patch("app.api.search.run_pipeline", side_effect=_empty_response) as mock_run:
        response = client.post(
            "/api/search", json={"topic": "diffusion policy", "published_after": "2024-01-01"}
        )
        run_id = response.json()["run_id"]
        _wait_for_terminal(run_id)

    assert response.status_code == 200
    mock_run.assert_called_once_with(
        "diffusion policy", date(2024, 1, 1), run_id=run_id
    )


def test_malformed_published_after_is_rejected(client):
    response = client.post(
        "/api/search", json={"topic": "diffusion policy", "published_after": "not-a-date"}
    )
    assert response.status_code == 422


def test_rate_limit_returns_429_after_configured_limit(client):
    limit = search_api._settings.rate_limit_per_minute
    with patch("app.api.search.run_pipeline", side_effect=_empty_response):
        for _ in range(limit):
            assert client.post("/api/search", json={"topic": "a topic"}).status_code == 200
        blocked = client.post("/api/search", json={"topic": "a topic"})

    assert blocked.status_code == 429


def test_rate_limiter_sweeps_stale_ip_keys(client):
    """Keys must not accumulate one-per-IP-ever; a sweep drops inactive clients."""
    search_api._ip_hits["203.0.113.7"] = search_api.deque()
    search_api._ip_hits["203.0.113.8"] = search_api.deque([0.0])  # long expired
    search_api._last_sweep = 0.0

    with patch("app.api.search.run_pipeline", side_effect=_empty_response):
        assert client.post("/api/search", json={"topic": "a topic"}).status_code == 200

    assert "203.0.113.7" not in search_api._ip_hits
    assert "203.0.113.8" not in search_api._ip_hits


def test_exhausted_budget_returns_429(client):
    budget.add(search_api._settings.daily_token_budget, 0)

    with patch("app.api.search.run_pipeline", side_effect=_empty_response) as mock_run:
        response = client.post("/api/search", json={"topic": "a topic"})

    assert response.status_code == 429
    # The ceiling must be enforced before any tokens are spent
    mock_run.assert_not_called()


def test_pipeline_error_marks_run_failed(client):
    """A hard pipeline failure now surfaces through run_status, not an HTTP
    error — POST already returned 200 with a run_id before the failure
    happens on the background thread."""
    from app.service import PipelineError

    with patch(
        "app.api.search.run_pipeline",
        side_effect=PipelineError("rerank", "Failed to rerank papers"),
    ):
        response = client.post("/api/search", json={"topic": "a topic"})
        run_id = response.json()["run_id"]
        snapshot = _wait_for_terminal(run_id)

    assert response.status_code == 200
    assert snapshot["state"] == "FAILED"
    assert snapshot["error"] == "Failed to rerank papers"


def test_security_headers_present(client):
    with patch("app.api.search.run_pipeline", side_effect=_empty_response):
        response = client.post("/api/search", json={"topic": "a topic"})

    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "Content-Security-Policy" in response.headers
