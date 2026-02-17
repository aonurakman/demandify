"""Tests for import-mode calibration flows in web routes."""

from types import SimpleNamespace

import pandas as pd
from starlette.testclient import TestClient

from demandify.app import app
from demandify.web import routes


def test_check_feasibility_import_mode_works_without_api_key(monkeypatch):
    class FakePipeline:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def prepare(self):
            traffic_df = pd.DataFrame(
                {
                    "segment_id": ["s1", "s2", "s3"],
                    "current_speed": [20.0, 22.0, 18.0],
                    "freeflow_speed": [40.0, 40.0, 35.0],
                    "timestamp": ["2026-02-17 12:00:00"] * 3,
                }
            )
            observed_edges = pd.DataFrame(
                {
                    "edge_id": ["e1", "e2"],
                    "segment_id": ["s1", "s2"],
                    "current_speed": [20.0, 22.0],
                    "freeflow_speed": [40.0, 40.0],
                    "match_confidence": [0.99, 1.0],
                }
            )
            return {
                "traffic_df": traffic_df,
                "observed_edges": observed_edges,
                "total_edges": 120,
            }

    import demandify.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "CalibrationPipeline", FakePipeline)
    monkeypatch.setattr(
        routes,
        "resolve_offline_dataset",
        lambda _: SimpleNamespace(
            dataset_id="packaged:krakow_v1",
            bbox={"west": 20.0174, "south": 50.0702, "east": 20.0566, "north": 50.0875},
        ),
    )

    client = TestClient(app)
    resp = client.post(
        "/api/check_feasibility",
        data={"data_mode": "import", "offline_dataset": "krakow_v1"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "success"
    assert payload["data_mode"] == "import"
    assert payload["offline_dataset"] == "packaged:krakow_v1"
    assert payload["stats"]["fetched_segments"] == 3
    assert payload["stats"]["matched_edges"] == 2
    assert "quality" in payload


def test_start_run_import_mode_accepts_missing_bbox(monkeypatch):
    monkeypatch.setattr(
        routes,
        "resolve_offline_dataset",
        lambda _: SimpleNamespace(
            dataset_id="packaged:rennes_v1",
            bbox={"west": -1.69, "south": 48.1054, "east": -1.6576, "north": 48.1189},
        ),
    )
    monkeypatch.setattr(routes, "run_calibration_pipeline", lambda *args, **kwargs: None)

    client = TestClient(app)
    resp = client.post(
        "/api/run",
        data={
            "data_mode": "import",
            "offline_dataset": "rennes_v1",
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "started"
    assert "run_id" in payload


def test_start_run_create_mode_with_offline_save(monkeypatch, tmp_path):
    captured = {}

    def fake_run_calibration_pipeline(run_id, params):
        captured["run_id"] = run_id
        captured["params"] = params

    monkeypatch.setattr(routes, "run_calibration_pipeline", fake_run_calibration_pipeline)
    monkeypatch.setattr(
        routes,
        "get_config",
        lambda: SimpleNamespace(tomtom_api_key="abc123", max_bbox_area_km2=25.0),
    )
    monkeypatch.setattr(routes, "offline_dataset_name_exists", lambda _: False)
    monkeypatch.setattr(routes, "get_writable_offline_datasets_root", lambda: tmp_path / "offline_root")

    client = TestClient(app)
    resp = client.post(
        "/api/run",
        data={
            "data_mode": "create",
            "bbox_west": 2.2961,
            "bbox_south": 48.8469,
            "bbox_east": 2.3071,
            "bbox_north": 48.8532,
            "save_offline_dataset": "true",
            "save_offline_dataset_name": "paris_v1",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "started"
    assert captured["params"]["save_offline_dataset"] is True
    assert captured["params"]["save_offline_dataset_name"] == "paris_v1"
    assert captured["params"]["save_offline_dataset_root"] == str(tmp_path / "offline_root")


def test_start_run_create_mode_rejects_existing_offline_save_name(monkeypatch):
    monkeypatch.setattr(
        routes,
        "get_config",
        lambda: SimpleNamespace(tomtom_api_key="abc123", max_bbox_area_km2=25.0),
    )
    monkeypatch.setattr(routes, "offline_dataset_name_exists", lambda _: True)

    client = TestClient(app)
    resp = client.post(
        "/api/run",
        data={
            "data_mode": "create",
            "bbox_west": 2.2961,
            "bbox_south": 48.8469,
            "bbox_east": 2.3071,
            "bbox_north": 48.8532,
            "save_offline_dataset": "true",
            "save_offline_dataset_name": "krakow_v1",
        },
    )
    assert resp.status_code == 409
    assert "already exists" in resp.text


def test_start_run_import_mode_rejects_offline_save(monkeypatch):
    monkeypatch.setattr(
        routes,
        "resolve_offline_dataset",
        lambda _: SimpleNamespace(
            dataset_id="packaged:rennes_v1",
            bbox={"west": -1.69, "south": 48.1054, "east": -1.6576, "north": 48.1189},
        ),
    )

    client = TestClient(app)
    resp = client.post(
        "/api/run",
        data={
            "data_mode": "import",
            "offline_dataset": "rennes_v1",
            "save_offline_dataset": "true",
            "save_offline_dataset_name": "should_fail",
        },
    )
    assert resp.status_code == 400
    assert "create mode" in resp.text
