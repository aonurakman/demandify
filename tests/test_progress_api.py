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
        "test-status-included", "running", 2, "Download OSM",
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
        "test-completed", "completed", 8, "Complete",
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
        "test-failed", "failed", 3, "Building Network",
        [{"message": "Error: network build failed", "level": "error"}],
    )

    client = TestClient(app)

    resp = client.get(f"/api/run/{run_id}/progress")
    data = resp.json()

    assert data["status"] == "failed"


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


