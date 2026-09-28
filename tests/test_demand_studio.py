"""
Tests for Demand Studio API endpoints, routing, multipliers, and scenario saving.
"""
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from demandify.app import app
from demandify.sumo.studio_service import (
    list_studio_sources,
    resolve_source_paths,
    StudioNetworkManager,
    apply_multiplier,
    parse_demand_dataframe,
    run_studio_test_simulation,
    save_studio_scenario,
)
import pandas as pd


@pytest.fixture
def client():
    return TestClient(app)


def test_navbar_present_on_all_pages(client):
    """Verify that the sleek navbar is rendered on all three primary views."""
    # 1. Calibration page
    res_calib = client.get("/")
    assert res_calib.status_code == 200
    assert "nav-pills-demandify" in res_calib.text
    assert "Demand Studio" in res_calib.text

    # 2. Dataset Builder page
    res_builder = client.get("/dataset-builder")
    assert res_builder.status_code == 200
    assert "nav-pills-demandify" in res_builder.text
    assert "Demand Studio" in res_builder.text

    # 3. Demand Studio page
    res_studio = client.get("/demand-studio")
    assert res_studio.status_code == 200
    assert "nav-pills-demandify" in res_studio.text
    assert "studio-workspace" in res_studio.text


def test_studio_sources_api(client):
    """Verify sources discovery returns cataloged datasets and runs."""
    res = client.get("/api/studio/sources")
    assert res.status_code == 200
    data = res.json()
    assert "datasets" in data
    assert "runs" in data
    assert "scenarios" in data
    assert len(data["datasets"]) > 0


def test_studio_load_dataset(client):
    """Verify loading a bundled offline dataset returns GeoJSON and routable edges."""
    res = client.get("/api/studio/load?source_type=dataset&source_id=krakow_v1")
    assert res.status_code == 200
    data = res.json()
    assert data["source_type"] == "dataset"
    assert "network" in data
    assert data["network"]["type"] == "FeatureCollection"
    assert len(data["network"]["features"]) > 500
    assert "bounds" in data["network"]
    assert len(data["routable_edges"]) > 500
    assert "edge_flows" in data
    assert isinstance(data["edge_flows"], dict)
    first_feat = data["network"]["features"][0]
    assert "flow" in first_feat["properties"]
    assert "flow_vehs_h" in first_feat["properties"]


def test_studio_route_computation(client):
    """Verify route calculation between two network edges."""
    # First get routable edges
    load_res = client.get("/api/studio/load?source_type=dataset&source_id=krakow_v1")
    edges = load_res.json()["routable_edges"]
    assert len(edges) >= 2

    orig = edges[0]
    dest = edges[1]

    route_res = client.post(
        "/api/studio/route",
        json={
            "source_type": "dataset",
            "source_id": "krakow_v1",
            "origin": orig,
            "destination": dest,
        },
    )
    assert route_res.status_code == 200
    route_data = route_res.json()
    assert "routable" in route_data
    assert "coordinates" in route_data


def test_studio_multiplier_logic():
    """Verify multiplier scaling and preserve-min-one behavior."""
    od_pairs = [
        {"id": "od_1", "origin": "A", "destination": "B", "vehs_per_hour": 100, "base_vehs_per_hour": 100},
        {"id": "od_2", "origin": "C", "destination": "D", "vehs_per_hour": 2, "base_vehs_per_hour": 2},
    ]

    # Test scaling up 1.5x
    scaled, summary = apply_multiplier(od_pairs, factor=1.5, preserve_min_one=True)
    assert scaled[0]["vehs_per_hour"] == 150
    assert scaled[1]["vehs_per_hour"] == 3
    assert summary["delta_pct"] == 50.0

    # Test scaling down 0.1x with preserve_min_one
    scaled_low, summary_low = apply_multiplier(od_pairs, factor=0.1, preserve_min_one=True)
    assert scaled_low[0]["vehs_per_hour"] == 10
    # 2 * 0.1 = 0.2 -> rounds to 0, but preserve_min_one forces 1
    assert scaled_low[1]["vehs_per_hour"] == 1


def test_studio_multiplier_api(client):
    """Verify POST /api/studio/apply-multiplier endpoint."""
    od_pairs = [
        {"id": "od_1", "origin": "e1", "destination": "e2", "vehs_per_hour": 60, "base_vehs_per_hour": 60}
    ]
    res = client.post(
        "/api/studio/apply-multiplier",
        json={
            "od_pairs": od_pairs,
            "factor": 1.25,
            "preserve_min_one": True,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["od_pairs"][0]["vehs_per_hour"] == 75
    assert data["summary"]["total_vehs_h_scaled"] == 75


def test_studio_save_scenario(client, tmp_path):
    """Verify saving a customized scenario creates all SUMO and metadata files."""
    net_file, _ = resolve_source_paths("dataset", "krakow_v1")
    mgr = StudioNetworkManager(net_file)
    routable = mgr.get_routable_edges()
    orig = routable[0]
    dest = routable[1]

    od_pairs = [
        {"id": "od_test_save", "origin": orig, "destination": dest, "vehs_per_hour": 120, "vehs_per_min": 2.0}
    ]

    res = client.post(
        "/api/studio/save-scenario",
        json={
            "source_type": "dataset",
            "source_id": "krakow_v1",
            "scenario_id": "test_saved_scenario_01",
            "name": "Test Saved Scenario",
            "od_pairs": od_pairs,
            "window_minutes": 15,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["scenario_id"] == "test_saved_scenario_01"
    scenario_path = Path(data["path"])
    assert (scenario_path / "sumo" / "network.net.xml").exists()
    assert (scenario_path / "sumo" / "trips.xml").exists()
    assert (scenario_path / "sumo" / "scenario.sumocfg").exists()
    assert (scenario_path / "data" / "demand.csv").exists()
    assert (scenario_path / "scenario_meta.json").exists()

    # Clean up test scenario folder
    import shutil
    shutil.rmtree(scenario_path, ignore_errors=True)


def test_studio_test_simulation_endpoint(client):
    """Verify background test simulation endpoint returns stats and edge congestion."""
    net_file, _ = resolve_source_paths("dataset", "krakow_v1")
    mgr = StudioNetworkManager(net_file)
    routable = mgr.get_routable_edges()
    orig = routable[0]
    dest = routable[1]

    od_pairs = [
        {"id": "od_sim_test", "origin": orig, "destination": dest, "vehs_per_hour": 60, "vehs_per_min": 1.0}
    ]

    res = client.post(
        "/api/studio/test-simulation",
        json={
            "source_type": "dataset",
            "source_id": "krakow_v1",
            "od_pairs": od_pairs,
            "window_minutes": 5,
            "mesosim": True,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "stats" in data
    assert "edge_congestion" in data
    assert data["stats"]["total_vehicles_inserted"] > 0
    assert "teleports" in data["stats"]
    assert "mean_edge_speed_kmh" in data["stats"]


def test_studio_disconnected_od_pair_detection(client):
    """Verify that disconnected OD pairs return routable=False and are rejected."""
    # In Brussels network, these two links are disconnected
    res = client.post(
        "/api/studio/route",
        json={
            "source_type": "dataset",
            "source_id": "brussel_v1",
            "origin": "662874055#0",
            "destination": "24387267#0",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["routable"] is False
    assert len(data["path_edges"]) == 0
    assert len(data["coordinates"]) == 0
