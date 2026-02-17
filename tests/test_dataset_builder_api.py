"""Tests for the dedicated offline dataset builder routes."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from starlette.testclient import TestClient

from demandify.app import app
from demandify.web import dataset_routes


def test_dataset_builder_page_available():
    client = TestClient(app)
    resp = client.get("/dataset-builder")
    assert resp.status_code == 200
    assert "Offline Dataset Builder" in resp.text


def test_known_datasets_catalog_merges_generated_and_packaged(monkeypatch, tmp_path):
    generated_root = tmp_path / "generated"
    packaged_root = tmp_path / "packaged"
    generated_root.mkdir()
    packaged_root.mkdir()

    gen_ds = generated_root / "gen_city_v1"
    gen_ds.mkdir()
    (gen_ds / "dataset_meta.json").write_text(
        json.dumps(
            {
                "bbox": {"west": 2.1, "south": 48.8, "east": 2.2, "north": 48.9},
                "quality": {"label": "good", "score": 75},
                "created_at": "2026-02-17T12:00:00",
            }
        ),
        encoding="utf-8",
    )

    pkg_ds = packaged_root / "pkg_city_v1"
    pkg_ds.mkdir()
    (pkg_ds / "dataset_meta.json").write_text(
        json.dumps(
            {
                "bbox": {"west": 20.0, "south": 50.0, "east": 20.1, "north": 50.1},
                "quality": {"label": "fair", "score": 58},
                "created_at": "2026-02-17T13:00:00",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(dataset_routes, "DATASETS_ROOT", generated_root)
    monkeypatch.setattr(dataset_routes, "PACKAGED_DATASETS_ROOT", packaged_root)

    catalog = dataset_routes._known_datasets_catalog()
    assert len(catalog) == 2

    by_name = {item["name"]: item for item in catalog}
    assert by_name["gen_city_v1"]["source"] == "generated"
    assert by_name["pkg_city_v1"]["source"] == "packaged"
    assert by_name["gen_city_v1"]["bbox"]["west"] == 2.1
    assert by_name["pkg_city_v1"]["bbox"]["north"] == 50.1


def test_dataset_builder_page_renders_known_datasets_payload(monkeypatch):
    monkeypatch.setattr(
        dataset_routes,
        "_known_datasets_catalog",
        lambda: [
            {
                "id": "generated:demo_v1",
                "name": "demo_v1",
                "source": "generated",
                "bbox": {"west": 2.1, "south": 48.8, "east": 2.2, "north": 48.9},
                "created_at": "2026-02-17T12:00:00",
                "quality_label": "good",
                "quality_score": 77,
            }
        ],
    )
    monkeypatch.setattr(
        dataset_routes,
        "get_config",
        lambda: SimpleNamespace(tomtom_api_key="abc", max_bbox_area_km2=25.0),
    )

    client = TestClient(app)
    resp = client.get("/dataset-builder")
    assert resp.status_code == 200
    html = resp.text
    assert "existing_dataset_select" in html
    assert "generated:demo_v1" in html
    assert "known-datasets-json" in html
    assert "demo_v1" in html


def test_dataset_build_requires_api_key(monkeypatch, tmp_path):
    monkeypatch.setattr(dataset_routes, "DATASETS_ROOT", tmp_path / "datasets")
    monkeypatch.setattr(
        dataset_routes,
        "get_config",
        lambda: SimpleNamespace(tomtom_api_key=None, max_bbox_area_km2=25.0),
    )

    client = TestClient(app)
    resp = client.post(
        "/api/datasets/build",
        data={
            "dataset_name": "sample_dataset",
            "bbox_west": 2.2961,
            "bbox_south": 48.8469,
            "bbox_east": 2.3071,
            "bbox_north": 48.8532,
            "traffic_tile_zoom": 12,
        },
    )
    assert resp.status_code == 400
    assert "TomTom API key not configured" in resp.text


def test_dataset_build_rejects_invalid_name(monkeypatch, tmp_path):
    monkeypatch.setattr(dataset_routes, "DATASETS_ROOT", tmp_path / "datasets")
    monkeypatch.setattr(
        dataset_routes,
        "get_config",
        lambda: SimpleNamespace(tomtom_api_key="abc", max_bbox_area_km2=25.0),
    )

    client = TestClient(app)
    resp = client.post(
        "/api/datasets/build",
        data={
            "dataset_name": "../bad/name",
            "bbox_west": 2.2961,
            "bbox_south": 48.8469,
            "bbox_east": 2.3071,
            "bbox_north": 48.8532,
            "traffic_tile_zoom": 12,
        },
    )
    assert resp.status_code == 400
    assert "Dataset name must use only letters" in resp.text


def test_dataset_build_start_creates_job(monkeypatch, tmp_path):
    monkeypatch.setattr(dataset_routes, "DATASETS_ROOT", tmp_path / "datasets")
    monkeypatch.setattr(
        dataset_routes,
        "get_config",
        lambda: SimpleNamespace(tomtom_api_key="abc", max_bbox_area_km2=25.0),
    )

    async def fake_run_dataset_builder(job_id, params):  # pragma: no cover - simple test hook
        dataset_routes.active_dataset_jobs[job_id]["status"] = "completed"
        dataset_routes.active_dataset_jobs[job_id]["output_dir"] = params["output_dir"]
        dataset_routes.active_dataset_jobs[job_id]["progress"]["stage"] = 5
        dataset_routes.active_dataset_jobs[job_id]["progress"]["stage_name"] = "Complete"

    monkeypatch.setattr(dataset_routes, "run_dataset_builder", fake_run_dataset_builder)

    client = TestClient(app)
    resp = client.post(
        "/api/datasets/build",
        data={
            "dataset_name": "sample_dataset",
            "bbox_west": 2.2961,
            "bbox_south": 48.8469,
            "bbox_east": 2.3071,
            "bbox_north": 48.8532,
            "traffic_tile_zoom": 12,
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "started"
    assert payload["dataset_name"] == "sample_dataset"

    job_id = payload["job_id"]
    progress = client.get(f"/api/datasets/{job_id}/progress")
    assert progress.status_code == 200
    assert progress.json()["status"] == "completed"

    dataset_routes.active_dataset_jobs.pop(job_id, None)


def test_dataset_progress_log_ingestion_is_incremental(tmp_path):
    """Dataset progress endpoint should ingest only newly appended log-file lines."""
    output_dir = tmp_path / "datasets" / "sample_dataset"
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "pipeline.log"
    log_file.write_text(
        "\n".join(
            [
                "2026-02-17 10:00:00 - demandify - INFO - Dataset first",
                "2026-02-17 10:00:01 - demandify - WARNING - Dataset second",
                "2026-02-17 10:00:02 - demandify - DEBUG - Ignored debug",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    job_id = "dataset_progress_incremental_test"
    dataset_routes.active_dataset_jobs[job_id] = {
        "status": "running",
        "dataset_name": "sample_dataset",
        "created_at": "now",
        "output_dir": str(output_dir),
        "progress": {"stage": 0, "stage_name": "Initializing", "logs": []},
    }

    client = TestClient(app)

    first = client.get(f"/api/datasets/{job_id}/progress")
    assert first.status_code == 200
    assert [entry["message"] for entry in first.json()["logs"]] == [
        "Dataset first",
        "Dataset second",
    ]

    second = client.get(f"/api/datasets/{job_id}/progress")
    assert second.status_code == 200
    assert [entry["message"] for entry in second.json()["logs"]] == [
        "Dataset first",
        "Dataset second",
    ]

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("2026-02-17 10:00:03 - demandify - ERROR - Dataset third\n")

    third = client.get(f"/api/datasets/{job_id}/progress")
    assert third.status_code == 200
    assert [entry["message"] for entry in third.json()["logs"]] == [
        "Dataset first",
        "Dataset second",
        "Dataset third",
    ]

    dataset_routes.active_dataset_jobs.pop(job_id, None)


def test_dataset_builder_writes_metadata_and_bundle(monkeypatch, tmp_path):
    class FakePipeline:
        def __init__(self, **kwargs):
            self.output_dir = Path(kwargs["output_dir"])
            self.provider_meta = {"provider": "tomtom", "tile_zoom": kwargs.get("traffic_tile_zoom", 12)}
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "data").mkdir(exist_ok=True)
            (self.output_dir / "sumo").mkdir(exist_ok=True)
            (self.output_dir / "plots").mkdir(exist_ok=True)
            (self.output_dir / "logs").mkdir(exist_ok=True)

        async def prepare(self):
            traffic = pd.DataFrame({"id": [1, 2], "speed_kph": [22.0, 18.0]})
            observed = pd.DataFrame({"edge_id": ["e1", "e2"]})
            traffic.to_csv(self.output_dir / "data" / "traffic_data_raw.csv", index=False)
            observed.to_csv(self.output_dir / "data" / "observed_edges.csv", index=False)
            (self.output_dir / "sumo" / "network.net.xml").write_text("<net/>", encoding="utf-8")
            (self.output_dir / "plots" / "network.png").write_bytes(b"png")
            (self.output_dir / "logs" / "pipeline.log").write_text("log", encoding="utf-8")
            return {
                "traffic_df": traffic,
                "observed_edges": observed,
                "total_edges": 11,
            }

        async def _fetch_osm_data(self):
            osm_cache = tmp_path / "osm_cache.osm"
            osm_cache.write_text("<osm/>", encoding="utf-8")
            return osm_cache

    import demandify.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "CalibrationPipeline", FakePipeline)

    output_dir = tmp_path / "datasets" / "sample_dataset"
    params = {
        "dataset_name": "sample_dataset",
        "bbox": [2.2961, 48.8469, 2.3071, 48.8532],
        "traffic_tile_zoom": 12,
        "output_dir": str(output_dir),
    }
    job_id = "sample_job_1"
    dataset_routes.active_dataset_jobs[job_id] = {
        "status": "running",
        "dataset_name": "sample_dataset",
        "created_at": "now",
        "progress": {"stage": 0, "stage_name": "Initializing", "logs": []},
    }

    asyncio.run(dataset_routes.run_dataset_builder(job_id, params))

    assert dataset_routes.active_dataset_jobs[job_id]["status"] == "completed"
    assert (output_dir / "data" / "map.osm").exists()
    assert (output_dir / "dataset_meta.json").exists()

    with open(output_dir / "dataset_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["dataset_name"] == "sample_dataset"
    assert meta["stats"]["fetched_segments"] == 2
    assert meta["stats"]["matched_edges"] == 2
    assert meta["stats"]["total_network_edges"] == 11
    assert "quality" in meta
    assert "label" in meta["quality"]
    assert "score" in meta["quality"]

    dataset_routes.active_dataset_jobs.pop(job_id, None)
