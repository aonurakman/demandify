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

    # Also verify vehicle_types.xml is copied and wired into scenario.sumocfg when present
    temp_net_dir = tmp_path / "sumo"
    temp_net_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(net_file, temp_net_dir / "network.net.xml")
    (temp_net_dir / "vehicle_types.xml").write_text(
        "<additional><vType id=\"DEFAULT_VEHTYPE\" tau=\"1.2718\"/></additional>",
        encoding="utf-8",
    )
    saved_vtypes = save_studio_scenario(
        scenario_id="test_saved_scenario_vtypes",
        network_file=temp_net_dir / "network.net.xml",
        od_pairs=od_pairs,
        window_minutes=15,
    )
    vtypes_scen_path = Path(saved_vtypes["path"])
    try:
        assert (vtypes_scen_path / "sumo" / "vehicle_types.xml").exists()
        sumocfg_text = (vtypes_scen_path / "sumo" / "scenario.sumocfg").read_text(encoding="utf-8")
        assert "vehicle_types.xml" in sumocfg_text
    finally:
        shutil.rmtree(vtypes_scen_path, ignore_errors=True)


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


def test_studio_fetch_osm_uses_fetch_osm_data_and_derates_network(monkeypatch, tmp_path):
    """Verify fetch_osm_and_build_studio_network calls fetch_osm_data, closes client, and applies speed caps."""
    import asyncio
    import demandify.sumo.studio_service as studio_service

    calls = {"fetched": None, "closed": False, "converted": False, "derated": None}

    class FakeFetcher:
        def __init__(self, timeout=180):
            self.timeout = timeout

        async def fetch_osm_data(self, bbox, output_file):
            calls["fetched"] = (bbox, Path(output_file))
            Path(output_file).write_text("<osm/>", encoding="utf-8")
            return output_file

        async def close(self):
            calls["closed"] = True

    def fake_convert(osm_file, net_file, car_only=True):
        calls["converted"] = True
        Path(net_file).write_text("<net/>", encoding="utf-8")
        return net_file, {}

    def fake_derate(net_file, observed_edges=None):
        calls["derated"] = Path(net_file)
        return 0

    monkeypatch.setattr(studio_service, "STUDIO_SCENARIOS_DIR", tmp_path / "scenarios")
    monkeypatch.setattr(studio_service, "OSMFetcher", FakeFetcher)
    monkeypatch.setattr(studio_service, "convert_osm_to_sumo", fake_convert)
    monkeypatch.setattr(studio_service, "apply_road_hierarchy_speed_derating", fake_derate)

    res = asyncio.run(
        studio_service.fetch_osm_and_build_studio_network(
            bbox=(4.28, 52.06, 4.33, 52.08),
            name="Test City",
        )
    )
    assert res["scenario_id"] == "test_city"
    assert calls["fetched"] is not None
    assert calls["closed"] is True
    assert calls["converted"] is True
    assert calls["derated"] == Path(res["net_file"])


def test_studio_load_run_uses_actual_run_duration(client, monkeypatch, tmp_path):
    """Verify Demand Studio computes hourly flow rates using run_meta.json bin_minutes (e.g. warmup + window)."""
    import json
    import shutil
    import demandify.sumo.studio_service as studio_service

    monkeypatch.chdir(tmp_path)
    net_src, _ = resolve_source_paths("dataset", "krakow_v1")
    mgr = StudioNetworkManager(net_src)
    routable = mgr.get_routable_edges()
    orig, dest = routable[0], routable[1]

    run_dir = tmp_path / "demandify_runs" / "run_demo_20min"
    (run_dir / "sumo").mkdir(parents=True)
    (run_dir / "data").mkdir(parents=True)
    shutil.copy2(net_src, run_dir / "sumo" / "network.net.xml")

    # 20 trips across a 20-minute bin (15 window + 5 warmup) -> 60 veh/h, NOT 80 veh/h
    pd.DataFrame(
        [
            {
                "ID": f"t_{i}",
                "origin link id": orig,
                "destination link id": dest,
                "departure timestep": float(i * 60 + 10),
            }
            for i in range(20)
        ]
    ).to_csv(run_dir / "data" / "demand.csv", index=False)

    (run_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "simulation_config": {"window_minutes": 15, "warmup_minutes": 5},
                "demand_config": {"bin_minutes": 20.0},
            }
        ),
        encoding="utf-8",
    )

    # Clear route network cache interference just in case
    assert studio_service.resolve_source_duration_minutes("run", "demo_20min") == 20

    res = client.get("/api/studio/load?source_type=run&source_id=demo_20min")
    assert res.status_code == 200
    payload = res.json()
    assert payload["summary"]["duration_minutes"] == 20
    assert payload["summary"]["total_vehs_h"] == 60
    assert payload["od_pairs"][0]["vehs_per_hour"] == 60
