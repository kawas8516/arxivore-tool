import pytest
from fastapi.testclient import TestClient

from app import storage
from app.main import app
from app.models import Author, Cluster, Landscape, Paper, SearchResponse


def _paper(arxiv_id: str) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        abstract="An abstract.",
        authors=[Author(name="Alice")],
        categories=["cs.LG"],
        published="2024-01-01",
        url=f"https://arxiv.org/abs/{arxiv_id}",
        problem="A problem.",
        method="A method.",
        results="Results.",
        contribution="A contribution.",
        extract_status="done",
        relevance_score=0.9,
        relevance_rationale="Relevant.",
    )


@pytest.fixture
def client(tmp_path, monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    get_settings.cache_clear()
    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()


def _seed_run(run_id: str, topic: str = "retrieval") -> None:
    paper = _paper("2401.0001")
    response = SearchResponse(
        run_id=run_id,
        topic=topic,
        candidates_retrieved=1,
        papers_returned=1,
        papers=[paper],
        landscape=Landscape(clusters=[Cluster(name="C", summary="s", arxiv_ids=["2401.0001"])]),
        queries=[topic],
        retrieve_ms=10,
        rerank_ms=5,
        prompt_tokens=100,
        completion_tokens=50,
    )
    storage.save_extraction(paper)
    storage.save_run(run_id, response)


def test_get_run_returns_404_for_unknown_run(client):
    response = client.get("/api/runs/does-not-exist")
    assert response.status_code == 404


def test_get_run_returns_stored_run(client):
    _seed_run("run-1")
    response = client.get("/api/runs/run-1")

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "run-1"
    assert body["topic"] == "retrieval"
    assert body["prompt_tokens"] == 100
    assert len(body["papers"]) == 1
    assert body["papers"][0]["arxiv_id"] == "2401.0001"
    assert body["landscape"]["clusters"][0]["name"] == "C"


def test_get_run_papers(client):
    _seed_run("run-1")
    response = client.get("/api/runs/run-1/papers")

    assert response.status_code == 200
    papers = response.json()
    assert len(papers) == 1
    assert papers[0]["problem"] == "A problem."


def test_get_run_papers_404_for_unknown_run(client):
    assert client.get("/api/runs/nope/papers").status_code == 404


def test_get_run_landscape(client):
    _seed_run("run-1")
    response = client.get("/api/runs/run-1/landscape")

    assert response.status_code == 200
    assert response.json()["clusters"][0]["name"] == "C"


def test_list_runs_newest_first(client):
    _seed_run("run-a", topic="a")
    _seed_run("run-b", topic="b")

    response = client.get("/api/runs")

    assert response.status_code == 200
    ids = [r["run_id"] for r in response.json()]
    assert ids == ["run-b", "run-a"]
    assert response.json()[0]["paper_count"] == 1


def test_patch_paper_updates_read_status(client):
    _seed_run("run-1")
    response = client.patch("/api/papers/2401.0001", json={"read_status": "read"})

    assert response.status_code == 200
    assert response.json()["read_status"] == "read"


def test_patch_paper_404_for_unknown_paper(client):
    response = client.patch("/api/papers/does-not-exist", json={"read_status": "read"})
    assert response.status_code == 404


def test_patch_paper_422_for_invalid_status(client):
    _seed_run("run-1")
    response = client.patch("/api/papers/2401.0001", json={"read_status": "not-a-real-status"})
    assert response.status_code == 422
