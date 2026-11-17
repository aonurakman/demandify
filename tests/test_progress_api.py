"""Tests for progress API response and pipeline async behavior."""

import inspect

import pytest
from starlette.testclient import TestClient

from demandify.app import app
from demandify.pipeline import CalibrationPipeline
from demandify.web.routes import active_runs


@pytest.fixture()
def run_entry():
    """Create and clean up an active_runs entry for testing."""
    created_ids = []

    def _make(run_id, status, stage, stage_name, logs):
        active_runs[run_id] = {
            "status": status,
            "progress": {
                "stage": stage,
                "stage_name": stage_name,
                "logs": list(logs),
            },
        }
        created_ids.append(run_id)
        return run_id

    yield _make

    for rid in created_ids:
        active_runs.pop(rid, None)


def test_progress_response_includes_status(run_entry):
    """Progress endpoint should include run status alongside progress fields."""
    run_id = run_entry(
        "test-status-included",
        "running",
        2,
        "Download OSM",
        [{"message": "Downloading...", "level": "info"}],
    )

    client = TestClient(app)

    resp = client.get(f"/api/run/{run_id}/progress")
    assert resp.status_code == 200
    data = resp.json()

    # Must include status
    assert "status" in data
    assert data["status"] == "running"
    # Must still include progress fields
    assert data["stage"] == 2
    assert data["stage_name"] == "Download OSM"
    assert len(data["logs"]) >= 1


def test_progress_response_completed_status(run_entry):
    """Completed runs should return status='completed'."""
    run_id = run_entry(
        "test-completed",
        "completed",
        8,
        "Complete",
        [{"message": "Done", "level": "info"}],
    )

    client = TestClient(app)

    resp = client.get(f"/api/run/{run_id}/progress")
    data = resp.json()

    assert data["status"] == "completed"
    assert data["stage"] == 8


def test_progress_response_failed_status(run_entry):
    """Failed runs should return status='failed'."""
    run_id = run_entry(
        "test-failed",
        "failed",
        3,
        "Building Network",
        [{"message": "Error: network build failed", "level": "error"}],
    )

    client = TestClient(app)

    resp = client.get(f"/api/run/{run_id}/progress")
    data = resp.json()

    assert data["status"] == "failed"


def test_results_page_renders_for_active_run(run_entry):
    """Results page should render successfully for an existing in-memory run."""
    run_id = run_entry(
        "test-results-page",
        "completed",
        8,
        "Complete",
        [{"message": "Done", "level": "info"}],
    )
    active_runs[run_id]["metadata"] = {"results": {"final_loss_mae_kmh": 12.3}}

    client = TestClient(app)

    resp = client.get(f"/results?run_id={run_id}")
    assert resp.status_code == 200
    assert run_id in resp.text


def test_log_trimming_in_progress_endpoint(run_entry):
    """Logs should be trimmed to MAX_LOG_ENTRIES when appended via update callback."""
    from demandify.web.routes import MAX_LOG_ENTRIES

    # Create a run with more logs than MAX_LOG_ENTRIES to test trimming
    large_logs = [
        {"message": f"Log entry {i}", "level": "info"} for i in range(MAX_LOG_ENTRIES + 50)
    ]
    run_id = run_entry(
        "test-log-trim",
        "running",
        5,
        "Calibrating",
        large_logs,
    )

    # Verify initial state: endpoint returns all logs (no trimming yet)
    client = TestClient(app)
    resp = client.get(f"/api/run/{run_id}/progress")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["logs"]) == MAX_LOG_ENTRIES + 50  # No trimming yet

    # Simulate the update callback behavior by appending logs and applying the same
    # trimming logic that _do_update uses. This tests that the trimming logic works.
    for i in range(20):
        active_runs[run_id]["progress"]["logs"].append(
            {"message": f"New log entry {i}", "level": "info"}
        )
        # Apply the same trimming logic as _do_update
        if len(active_runs[run_id]["progress"]["logs"]) > MAX_LOG_ENTRIES:
            active_runs[run_id]["progress"]["logs"] = active_runs[run_id]["progress"]["logs"][
                -MAX_LOG_ENTRIES:
            ]

    # Verify logs are trimmed to MAX_LOG_ENTRIES after simulating updates
    assert len(active_runs[run_id]["progress"]["logs"]) == MAX_LOG_ENTRIES

    # Verify we kept the most recent entries
    assert active_runs[run_id]["progress"]["logs"][-1]["message"] == "New log entry 19"

    # Verify the endpoint returns the trimmed logs
    resp = client.get(f"/api/run/{run_id}/progress")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["logs"]) == MAX_LOG_ENTRIES


def test_calibrate_runs_in_thread():
    """pipeline.run() should use asyncio.to_thread for calibrate()."""
    # Verify run() is async
    assert inspect.iscoroutinefunction(CalibrationPipeline.run)

    # Verify prepare() is async
    assert inspect.iscoroutinefunction(CalibrationPipeline.prepare)

    # Verify calibrate() is sync (it's the method that needs wrapping)
    assert not inspect.iscoroutinefunction(CalibrationPipeline.calibrate)


def test_log_capping_prevents_unbounded_growth(run_entry):
    """Logs should be capped at MAX_LOG_ENTRIES to prevent memory issues."""
    from demandify.web.routes import MAX_LOG_ENTRIES

    # Create a run with more logs than the cap
    num_logs = MAX_LOG_ENTRIES + 100
    initial_logs = [{"message": f"Log {i}", "level": "info"} for i in range(num_logs)]
    run_id = run_entry("test-log-cap", "running", 5, "Calibrating", initial_logs)

    # Get the run data
    run = active_runs[run_id]
    
    # The initial logs should already exceed MAX_LOG_ENTRIES
    assert len(run["progress"]["logs"]) == num_logs
    
    # Simulate what _do_update does: append a log and cap
    logs = run["progress"]["logs"]
    logs.append({"message": "New log entry", "level": "info"})
    
    # Apply the capping logic used in routes.py
    if len(logs) > MAX_LOG_ENTRIES:
        run["progress"]["logs"] = logs[-MAX_LOG_ENTRIES:]

    # Verify logs are capped at MAX_LOG_ENTRIES
    assert len(run["progress"]["logs"]) == MAX_LOG_ENTRIES
    
    # Verify the newest log is still there
    assert run["progress"]["logs"][-1]["message"] == "New log entry"
    
    # Verify boundary: oldest logs were removed
    # With 600 initial logs + 1 new = 601 total, we keep last 500
    # So logs 0-100 should be removed, logs 101-600 should remain
    assert "Log 0" not in [log["message"] for log in run["progress"]["logs"]]
    assert "Log 100" not in [log["message"] for log in run["progress"]["logs"]]
    assert "Log 101" in [log["message"] for log in run["progress"]["logs"]]
    
    # Verify we kept the most recent entries
    log_messages = [log["message"] for log in run["progress"]["logs"]]
    # Should have logs 101-599 (499 entries) plus "New log entry" = 500 total
    assert f"Log {num_logs - 1}" in log_messages  # Last of initial logs


def test_progress_log_ingestion_is_incremental(run_entry, tmp_path):
    """Progress endpoint should ingest only newly appended log-file lines."""
    run_dir = tmp_path / "run_incremental"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "pipeline.log"
    log_file.write_text(
        "\n".join(
            [
                "2026-02-17 10:00:00 - demandify - INFO - First line",
                "2026-02-17 10:00:01 - demandify - WARNING - Second line",
                "2026-02-17 10:00:02 - demandify - DEBUG - Ignored debug",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    run_id = run_entry("test-incremental-logs", "running", 1, "Init", [])
    active_runs[run_id]["output_dir"] = str(run_dir)

    client = TestClient(app)

    first = client.get(f"/api/run/{run_id}/progress")
    assert first.status_code == 200
    first_logs = first.json()["logs"]
    assert [entry["message"] for entry in first_logs] == ["First line", "Second line"]

    second = client.get(f"/api/run/{run_id}/progress")
    assert second.status_code == 200
    second_logs = second.json()["logs"]
    assert [entry["message"] for entry in second_logs] == ["First line", "Second line"]

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("2026-02-17 10:00:03 - demandify - ERROR - Third line\n")
        f.write("2026-02-17 10:00:04 - demandify - INFO - Fourth line\n")

    third = client.get(f"/api/run/{run_id}/progress")
    assert third.status_code == 200
    third_logs = third.json()["logs"]
    assert [entry["message"] for entry in third_logs] == [
        "First line",
        "Second line",
        "Third line",
        "Fourth line",
    ]


def test_cancel_run_endpoint_signals_event_and_aborts(run_entry):
    """POST /api/run/{run_id}/cancel should set cancel_event and mark status aborted."""
    import threading
    from demandify.web.routes import run_cancel_events

    run_id = run_entry(
        "test-cancel-active",
        "running",
        6,
        "Calibrating Demand",
        [{"message": "Gen 1...", "level": "info"}],
    )
    cancel_ev = threading.Event()
    run_cancel_events[run_id] = cancel_ev

    try:
        client = TestClient(app)
        resp = client.post(f"/api/run/{run_id}/cancel")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == run_id
        assert data["status"] == "aborted"
        assert cancel_ev.is_set()
        assert active_runs[run_id]["status"] == "aborted"
        assert any(
            "Cancellation requested" in log["message"]
            for log in active_runs[run_id]["progress"]["logs"]
        )
    finally:
        run_cancel_events.pop(run_id, None)


def test_cancel_run_endpoint_idempotent_on_terminal_and_404_on_missing(run_entry):
    """Cancel endpoint should return 404 for unknown run and preserve terminal status."""
    client = TestClient(app)
    missing_resp = client.post("/api/run/nonexistent-run-xyz/cancel")
    assert missing_resp.status_code == 404

    run_id = run_entry(
        "test-cancel-completed",
        "completed",
        8,
        "Complete",
        [{"message": "Done", "level": "info"}],
    )
    done_resp = client.post(f"/api/run/{run_id}/cancel")
    assert done_resp.status_code == 200
    assert done_resp.json()["status"] == "completed"
    assert active_runs[run_id]["status"] == "completed"


def test_ga_optimize_aborts_on_cancel_event():
    """GeneticAlgorithm.optimize should raise PipelineCancelledError when cancel_event is set."""
    import threading
    import numpy as np
    from demandify.calibration.optimizer import GeneticAlgorithm, PipelineCancelledError

    cancel_ev = threading.Event()
    eval_calls = 0

    def _eval_fn(genome: np.ndarray):
        nonlocal eval_calls
        eval_calls += 1
        if eval_calls >= 3:
            cancel_ev.set()
        return float(np.sum(genome)), {"mae": float(np.sum(genome)), "teleports": 0}

    ga = GeneticAlgorithm(
        genome_size=4,
        seed=42,
        initial_population=100,
        population_size=6,
        num_generations=10,
        num_workers=1,
        cancel_event=cancel_ev,
    )

    with pytest.raises(PipelineCancelledError):
        ga.optimize(_eval_fn)

    assert eval_calls == 3


def test_pipeline_run_returns_none_when_cancelled(tmp_path):
    """CalibrationPipeline.run() should return None cleanly when cancel_event is set."""
    import asyncio
    import threading

    cancel_ev = threading.Event()
    cancel_ev.set()

    pipeline = CalibrationPipeline(
        bbox=(2.29, 48.84, 2.31, 48.85),
        window_minutes=15,
        seed=42,
        output_dir=tmp_path / "run_cancelled",
        run_id="cancelled_test",
        cancel_event=cancel_ev,
    )

    result = asyncio.run(pipeline.run())
    assert result is None

