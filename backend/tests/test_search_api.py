from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import budget
from app.api import search as search_api
from app.main import app
from app.models import SearchResponse


def _empty_response(topic: str) -> SearchResponse:
    return SearchResponse(
        topic=topic,
        candidates_retrieved=0,
        papers_returned=0,
        papers=[],
        retrieve_ms=1,
        rerank_ms=0,
    )


@pytest.fixture(autouse=True)
def _clean_state():
    budget.reset()
    search_api._ip_hits.clear()
    search_api._last_sweep = 0.0
    yield
    budget.reset()
    search_api._ip_hits.clear()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_search_returns_pipeline_result(client):
    with patch("app.api.search.run_pipeline", side_effect=_empty_response):
        response = client.post("/api/search", json={"topic": "diffusion policy"})

    assert response.status_code == 200
    assert response.json()["topic"] == "diffusion policy"


def test_short_topic_is_rejected(client):
    response = client.post("/api/search", json={"topic": "ab"})
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


def test_pipeline_error_surfaces_as_502(client):
    from app.service import PipelineError

    with patch(
        "app.api.search.run_pipeline",
        side_effect=PipelineError("rerank", "Failed to rerank papers"),
    ):
        response = client.post("/api/search", json={"topic": "a topic"})

    assert response.status_code == 502
    assert response.json()["detail"] == "Failed to rerank papers"


def test_security_headers_present(client):
    with patch("app.api.search.run_pipeline", side_effect=_empty_response):
        response = client.post("/api/search", json={"topic": "a topic"})

    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "Content-Security-Policy" in response.headers
