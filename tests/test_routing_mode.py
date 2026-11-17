import xml.etree.ElementTree as ET
from pathlib import Path
import pytest

from demandify.sumo.simulation import SUMOSimulation
from demandify.export.exporter import write_sumocfg
from demandify.cli import build_parser


def test_simulation_routing_mode_static(tmp_path):
    sim = SUMOSimulation(
        network_file=Path("net.net.xml"),
        vehicle_file=Path("trips.xml"),
        routing_mode="static",
    )
    assert sim.routing_mode == "static"
    assert sim.use_dynamic_routing is False

    cfg_file = tmp_path / "sim.sumocfg"
    sim._create_config(cfg_file, tmp_path / "edge.xml", tmp_path / "tripinfo.xml", tmp_path / "stats.xml")
    root = ET.parse(cfg_file).getroot()
    assert root.find("routing") is None


def test_simulation_routing_mode_dynamic(tmp_path):
    sim = SUMOSimulation(
        network_file=Path("net.net.xml"),
        vehicle_file=Path("trips.xml"),
        routing_mode="dynamic",
        seed=42,
    )
    assert sim.routing_mode == "dynamic"
    assert sim.use_dynamic_routing is True

    cfg_file = tmp_path / "sim.sumocfg"
    sim._create_config(cfg_file, tmp_path / "edge.xml", tmp_path / "tripinfo.xml", tmp_path / "stats.xml")
    root = ET.parse(cfg_file).getroot()
    routing = root.find("routing")
    assert routing is not None
    assert routing.find("device.rerouting.probability").get("value") == "0.50"
    assert routing.find("device.rerouting.period").get("value") == "300"


def test_simulation_invalid_routing_mode():
    for bad_mode in ("invalid_mode", "pretrip"):
        with pytest.raises(ValueError, match="Unknown routing_mode"):
            SUMOSimulation(
                network_file=Path("net.net.xml"),
                vehicle_file=Path("trips.xml"),
                routing_mode=bad_mode,
            )


def test_write_sumocfg_routing_modes(tmp_path):
    # Static
    static_cfg = tmp_path / "static.sumocfg"
    write_sumocfg(Path("net.net.xml"), Path("trips.xml"), static_cfg, 900, routing_mode="static")
    root = ET.parse(static_cfg).getroot()
    assert root.find("routing") is None

    # Dynamic
    dynamic_cfg = tmp_path / "dynamic.sumocfg"
    write_sumocfg(Path("net.net.xml"), Path("trips.xml"), dynamic_cfg, 900, routing_mode="dynamic")
    root = ET.parse(dynamic_cfg).getroot()
    routing = root.find("routing")
    assert routing is not None
    assert routing.find("device.rerouting.probability").get("value") == "0.50"
    assert routing.find("device.rerouting.period").get("value") == "300"


def test_cli_routing_mode_default_and_flags():
    parser = build_parser()

    # Default is static
    args = parser.parse_args(["run", "4.28,52.06,4.33,52.08"])
    assert args.routing_mode == "static"

    # Explicit dynamic
    args = parser.parse_args(["run", "4.28,52.06,4.33,52.08", "--routing-mode", "dynamic"])
    assert args.routing_mode == "dynamic"

    # Removed pretrip is rejected
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "4.28,52.06,4.33,52.08", "--routing-mode", "pretrip"])


def test_web_start_run_forwards_routing_mode_and_lefthand(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from starlette.testclient import TestClient
    from demandify.app import app
    from demandify.web import routes

    captured = {}

    def fake_run_calibration_pipeline(run_id, params):
        captured["run_id"] = run_id
        captured["params"] = params

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(routes, "run_calibration_pipeline", fake_run_calibration_pipeline)
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
            "routing_mode": "dynamic",
            "lefthand": "true",
        },
    )
    assert resp.status_code == 200
    assert captured["params"]["routing_mode"] == "dynamic"
    assert captured["params"]["lefthand"] is True

    for rejected_mode in ("invalid", "pretrip"):
        bad_resp = client.post(
            "/api/run",
            data={
                "data_mode": "import",
                "offline_dataset": "rennes_v1",
                "routing_mode": rejected_mode,
            },
        )
        assert bad_resp.status_code == 400
