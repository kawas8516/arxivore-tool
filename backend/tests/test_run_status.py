from app import run_status


def _reset():
    run_status._runs.clear()


def setup_function(_):
    _reset()


def test_start_registers_queued():
    run_status.start("run-1", "a topic")
    snap = run_status.get("run-1")
    assert snap["state"] == "QUEUED"
    assert snap["topic"] == "a topic"
    assert snap["stages"]["retrieve"]["status"] == "pending"


def test_get_unknown_run_returns_none():
    assert run_status.get("does-not-exist") is None


def test_set_state_advances_and_bumps_seq():
    run_status.start("run-1", "topic")
    before = run_status.get("run-1")["seq"]
    run_status.set_state("run-1", "RETRIEVING")
    after = run_status.get("run-1")
    assert after["state"] == "RETRIEVING"
    assert after["seq"] > before


def test_set_state_on_unregistered_run_is_a_no_op():
    """Direct run_pipeline() callers (tests, the eval harness) never call
    start() — stage transitions must not raise for them."""
    run_status.set_state("never-started", "RETRIEVING")  # must not raise
    assert run_status.get("never-started") is None


def test_set_stage_updates_substage_fields():
    run_status.start("run-1", "topic")
    run_status.set_stage("run-1", "extract", status="running", done=3, total=10)
    stage = run_status.get("run-1")["stages"]["extract"]
    assert stage == {"status": "running", "done": 3, "total": 10}


def test_set_error_moves_to_failed():
    run_status.start("run-1", "topic")
    run_status.set_error("run-1", "boom")
    snap = run_status.get("run-1")
    assert snap["state"] == "FAILED"
    assert snap["error"] == "boom"


def test_terminal_state_ignores_further_state_changes():
    """Once FAILED/COMPLETE, later stage updates from a stray thread must not
    resurrect the run into a different state."""
    run_status.start("run-1", "topic")
    run_status.set_error("run-1", "boom")
    run_status.set_state("run-1", "RETRIEVING")
    assert run_status.get("run-1")["state"] == "FAILED"


def test_get_returns_an_isolated_copy():
    run_status.start("run-1", "topic")
    snap = run_status.get("run-1")
    snap["state"] = "TAMPERED"
    snap["stages"]["retrieve"]["status"] = "tampered"

    fresh = run_status.get("run-1")
    assert fresh["state"] == "QUEUED"
    assert fresh["stages"]["retrieve"]["status"] == "pending"


def test_sweep_drops_old_terminal_runs_but_keeps_active_ones(monkeypatch):
    import time

    run_status.start("old-done", "topic")
    run_status.set_state("old-done", "COMPLETE")
    run_status.start("active", "topic")  # QUEUED, never finished

    # Simulate the old run having finished long ago.
    run_status._runs["old-done"]["updated_at"] = time.time() - 999999

    # start() triggers a sweep as a side effect.
    run_status.start("new-run", "topic")

    assert run_status.get("old-done") is None
    assert run_status.get("active") is not None
    assert run_status.get("new-run") is not None
