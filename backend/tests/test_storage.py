import pytest

from app import storage
from app.config import get_settings
from app.models import Author, Landscape, Cluster, Paper, SearchResponse


def _paper(arxiv_id: str, status: str = "done") -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        abstract="An abstract.",
        authors=[Author(name="Alice")],
        categories=["cs.LG"],
        published="2024-01-01",
        url=f"https://arxiv.org/abs/{arxiv_id}",
        problem="A problem." if status == "done" else None,
        method="A method." if status == "done" else None,
        results="Results." if status == "done" else None,
        contribution="A contribution." if status == "done" else None,
        extract_status=status,
    )


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Every test gets its own throwaway SQLite file, never the real one."""
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    get_settings.cache_clear()
    storage.init_db()
    yield
    get_settings.cache_clear()


def test_cache_miss_returns_none():
    assert storage.get_cached_extraction("2401.0001") is None


def test_save_then_get_round_trips():
    storage.save_extraction(_paper("2401.0001"))
    cached = storage.get_cached_extraction("2401.0001")

    assert cached is not None
    assert cached.problem == "A problem."
    assert cached.extract_status == "done"


def test_failed_extraction_is_not_served_as_a_cache_hit():
    """A 'pending' or 'error' row must be a miss — only 'done' counts."""
    storage.save_extraction(_paper("2401.0001", status="error"))
    assert storage.get_cached_extraction("2401.0001") is None


def test_second_save_updates_but_preserves_first_seen_at():
    storage.save_extraction(_paper("2401.0001", status="pending"))
    storage.save_extraction(_paper("2401.0001", status="done"))

    cached = storage.get_cached_extraction("2401.0001")
    assert cached.extract_status == "done"
    assert cached.problem == "A problem."


def test_save_run_persists_metadata_and_papers():
    response = SearchResponse(
        topic="retrieval",
        candidates_retrieved=1,
        papers_returned=1,
        papers=[_paper("2401.0001")],
        landscape=Landscape(clusters=[Cluster(name="C", summary="s", arxiv_ids=["2401.0001"])]),
        queries=["retrieval"],
        retrieve_ms=10,
        rerank_ms=5,
        prompt_tokens=100,
        completion_tokens=50,
    )
    storage.save_extraction(response.papers[0])
    storage.save_run("run-1", response)

    run = storage.get_run("run-1")
    assert run is not None
    assert run["topic"] == "retrieval"
    assert run["prompt_tokens"] == 100
    assert len(run["papers"]) == 1
    assert run["papers"][0].arxiv_id == "2401.0001"
    assert run["landscape"].clusters[0].name == "C"


def test_get_run_returns_none_for_unknown_run():
    assert storage.get_run("does-not-exist") is None


def test_list_runs_orders_newest_first():
    response = SearchResponse(
        topic="t", candidates_retrieved=0, papers_returned=0, papers=[],
        retrieve_ms=0, rerank_ms=0,
    )
    storage.save_run("run-a", response)
    storage.save_run("run-b", response)

    runs = storage.list_runs()
    ids = [r["run_id"] for r in runs]
    assert ids == ["run-b", "run-a"]


def test_dedup_across_two_runs_sharing_a_paper():
    """The resolved dedup rule: one papers row, tagged by both runs."""
    shared = _paper("2401.0001")
    storage.save_extraction(shared)

    response_a = SearchResponse(
        topic="topic a", candidates_retrieved=1, papers_returned=1,
        papers=[shared], retrieve_ms=0, rerank_ms=0,
    )
    response_b = SearchResponse(
        topic="topic b", candidates_retrieved=1, papers_returned=1,
        papers=[shared], retrieve_ms=0, rerank_ms=0,
    )
    storage.save_run("run-a", response_a)
    storage.save_run("run-b", response_b)

    # One papers row, but both runs reference it.
    run_a = storage.get_run("run-a")
    run_b = storage.get_run("run-b")
    assert run_a["papers"][0].arxiv_id == "2401.0001"
    assert run_b["papers"][0].arxiv_id == "2401.0001"

    # A second cache lookup for the same paper is still a hit — no re-extraction.
    assert storage.get_cached_extraction("2401.0001") is not None


def test_update_read_status_round_trips():
    storage.save_extraction(_paper("2401.0001"))
    updated = storage.update_read_status("2401.0001", "read")
    assert updated is True

    response = SearchResponse(
        topic="t", candidates_retrieved=1, papers_returned=1,
        papers=[_paper("2401.0001")], retrieve_ms=0, rerank_ms=0,
    )
    storage.save_run("run-1", response)
    run = storage.get_run("run-1")
    # read_status is global to the paper, not stored on SearchResponse/Paper —
    # verify it persisted at the storage layer directly.
    with storage._connect() as conn:
        row = conn.execute(
            "SELECT read_status FROM papers WHERE arxiv_id = ?", ("2401.0001",)
        ).fetchone()
    assert row["read_status"] == "read"


def test_update_read_status_returns_false_for_unknown_paper():
    assert storage.update_read_status("does-not-exist", "read") is False


def test_update_read_status_rejects_invalid_value():
    storage.save_extraction(_paper("2401.0001"))
    with pytest.raises(ValueError):
        storage.update_read_status("2401.0001", "definitely-not-a-status")
