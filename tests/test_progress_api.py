"""Tests for progress API response and pipeline async behavior."""
import pytest
from unittest.mock import MagicMock

from demandify.web.routes import active_runs


def test_progress_response_includes_status():
    """Progress endpoint should include run status alongside progress fields."""
    run_id = "test-status-included"
    active_runs[run_id] = {
        "status": "running",
        "progress": {
            "stage": 2,
            "stage_name": "Download OSM",
            "logs": [{"message": "Downloading...", "level": "info"}]
        }
    }

    from starlette.testclient import TestClient
    from demandify.app import app

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

    # Cleanup
    del active_runs[run_id]


def test_progress_response_completed_status():
    """Completed runs should return status='completed'."""
    run_id = "test-completed"
    active_runs[run_id] = {
        "status": "completed",
        "progress": {
            "stage": 8,
            "stage_name": "Complete",
            "logs": [{"message": "Done", "level": "info"}]
        }
    }

    from starlette.testclient import TestClient
    from demandify.app import app

    client = TestClient(app)

    resp = client.get(f"/api/run/{run_id}/progress")
    data = resp.json()

    assert data["status"] == "completed"
    assert data["stage"] == 8

    del active_runs[run_id]


def test_progress_response_failed_status():
    """Failed runs should return status='failed'."""
    run_id = "test-failed"
    active_runs[run_id] = {
        "status": "failed",
        "progress": {
            "stage": 3,
            "stage_name": "Building Network",
            "logs": [{"message": "Error: network build failed", "level": "error"}]
        }
    }

    from starlette.testclient import TestClient
    from demandify.app import app

    client = TestClient(app)

    resp = client.get(f"/api/run/{run_id}/progress")
    data = resp.json()

    assert data["status"] == "failed"

    del active_runs[run_id]


def test_calibrate_runs_in_thread():
    """pipeline.run() should use asyncio.to_thread for calibrate()."""
    from demandify.pipeline import CalibrationPipeline
    import inspect

    # Verify run() is async
    assert inspect.iscoroutinefunction(CalibrationPipeline.run)

    # Verify the source code uses asyncio.to_thread for calibrate
    source = inspect.getsource(CalibrationPipeline.run)
    assert "asyncio.to_thread" in source
    assert "self.calibrate" in source
