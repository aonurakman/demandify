from pathlib import Path
import xml.etree.ElementTree as ET

from demandify.export.exporter import ScenarioExporter


def _write(path: Path, text: str = "x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_exporter_prefers_sumo_seed_over_run_seed(tmp_path):
    network = tmp_path / "inputs" / "network.net.xml"
    demand = tmp_path / "inputs" / "demand.csv"
    trips = tmp_path / "inputs" / "trips.xml"
    observed = tmp_path / "inputs" / "observed_edges.csv"
    _write(network)
    _write(demand, "id\n")
    _write(trips, "<routes/>")
    _write(observed, "edge_id\n")

    metadata = {
        "run_info": {"seed": 42, "sumo_seed": 123456789},
        "simulation_config": {"window_minutes": 15, "warmup_minutes": 5, "step_length_seconds": 1.0},
    }

    out_dir = tmp_path / "exported"
    exporter = ScenarioExporter(out_dir)
    exporter.export(network, demand, trips, observed, metadata)

    cfg = ET.parse(out_dir / "scenario.sumocfg").getroot()
    seed_node = cfg.find("./random/seed")
    assert seed_node is not None
    assert seed_node.get("value") == "123456789"


def test_exporter_falls_back_to_run_seed(tmp_path):
    network = tmp_path / "inputs" / "network.net.xml"
    demand = tmp_path / "inputs" / "demand.csv"
    trips = tmp_path / "inputs" / "trips.xml"
    observed = tmp_path / "inputs" / "observed_edges.csv"
    _write(network)
    _write(demand, "id\n")
    _write(trips, "<routes/>")
    _write(observed, "edge_id\n")

    metadata = {
        "run_info": {"seed": 7},
        "simulation_config": {"window_minutes": 15, "warmup_minutes": 5, "step_length_seconds": 1.0},
    }

    out_dir = tmp_path / "exported"
    exporter = ScenarioExporter(out_dir)
    exporter.export(network, demand, trips, observed, metadata)

    cfg = ET.parse(out_dir / "scenario.sumocfg").getroot()
    seed_node = cfg.find("./random/seed")
    assert seed_node is not None
    assert seed_node.get("value") == "7"
