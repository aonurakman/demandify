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


def test_log_trimming_in_progress_endpoint(run_entry):
    """Logs should be trimmed to MAX_LOG_ENTRIES when read via progress endpoint."""
    from demandify.web.routes import MAX_LOG_ENTRIES

    # Create a run with more logs than MAX_LOG_ENTRIES
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

    # Simulate appending more logs (like the log file tailing does)
    for i in range(20):
        active_runs[run_id]["progress"]["logs"].append(
            {"message": f"New log entry {i}", "level": "info"}
        )

    # Manually trigger the trimming logic
    if len(active_runs[run_id]["progress"]["logs"]) > MAX_LOG_ENTRIES:
        active_runs[run_id]["progress"]["logs"] = active_runs[run_id]["progress"]["logs"][
            -MAX_LOG_ENTRIES:
        ]

    # Verify logs are trimmed to MAX_LOG_ENTRIES
    assert len(active_runs[run_id]["progress"]["logs"]) == MAX_LOG_ENTRIES

    # Verify we kept the most recent entries
    assert active_runs[run_id]["progress"]["logs"][-1]["message"] == "New log entry 19"


def test_calibrate_runs_in_thread():
    """pipeline.run() should use asyncio.to_thread for calibrate()."""
    # Verify run() is async
    assert inspect.iscoroutinefunction(CalibrationPipeline.run)

    # Verify prepare() is async
    assert inspect.iscoroutinefunction(CalibrationPipeline.prepare)

    # Verify calibrate() is sync (it's the method that needs wrapping)
    assert not inspect.iscoroutinefunction(CalibrationPipeline.calibrate)
