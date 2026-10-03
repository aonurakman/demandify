"""Tests for effective capacity factor calculation, XML generation, and scenario export."""

import xml.etree.ElementTree as ET
from pathlib import Path
import pytest

from demandify.sumo.network import tau_from_capacity_factor, write_vehicle_types_xml
from demandify.export.exporter import write_sumocfg, ScenarioExporter


def test_tau_from_capacity_factor():
    """Verify conversion of capacity factor to SUMO tau."""
    # 1.0 (default) -> 1.0 s
    assert tau_from_capacity_factor(1.0) == pytest.approx(1.0, abs=1e-4)

    # 0.85 -> ~1.2718 s
    tau_085 = tau_from_capacity_factor(0.85)
    assert tau_085 == pytest.approx(1.2718, abs=1e-3)
    assert tau_085 > 1.0

    # Bounds validation
    with pytest.raises(ValueError):
        tau_from_capacity_factor(0.0)
    with pytest.raises(ValueError):
        tau_from_capacity_factor(-0.5)
    with pytest.raises(ValueError):
        tau_from_capacity_factor(1.1)


def test_write_vehicle_types_xml(tmp_path: Path):
    """Verify vehicle_types.xml writes both DEFAULT_VEHTYPE and passenger types."""
    vtypes_file = tmp_path / "vehicle_types.xml"
    write_vehicle_types_xml(1.2718, vtypes_file)

    assert vtypes_file.exists()
    tree = ET.parse(vtypes_file)
    root = tree.getroot()
    assert root.tag == "additional"

    vtypes = {v.get("id"): v.get("tau") for v in root.findall("vType")}
    # Crucial: DEFAULT_VEHTYPE must be present so untyped trips inherit the derated tau
    assert "DEFAULT_VEHTYPE" in vtypes
    assert vtypes["DEFAULT_VEHTYPE"] == "1.2718"

    # passenger type must also be present for explicitly typed trips
    assert "passenger" in vtypes
    assert vtypes["passenger"] == "1.2718"


def test_write_sumocfg_with_vehicle_types(tmp_path: Path):
    """Verify write_sumocfg auto-detects and adds vehicle_types.xml when present."""
    net_file = tmp_path / "network.net.xml"
    trips_file = tmp_path / "trips.xml"
    vtypes_file = tmp_path / "vehicle_types.xml"
    cfg_file = tmp_path / "scenario.sumocfg"

    net_file.write_text("<net/>", encoding="utf-8")
    trips_file.write_text("<routes/>", encoding="utf-8")
    vtypes_file.write_text("<additional/>", encoding="utf-8")

    # Without explicit additional_files, auto-detects sibling vehicle_types.xml
    write_sumocfg(net_file, trips_file, cfg_file, simulation_time=600)

    tree = ET.parse(cfg_file)
    root = tree.getroot()
    add_elem = root.find("input/additional-files")
    assert add_elem is not None
    assert "vehicle_types.xml" in add_elem.get("value")


def test_scenario_exporter_tracks_vehicle_types(tmp_path: Path):
    """Verify ScenarioExporter copies vehicle_types.xml and references it in sumocfg."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    net_file = src_dir / "network.net.xml"
    trips_file = src_dir / "trips.xml"
    demand_file = src_dir / "demand.csv"
    observed_file = src_dir / "observed_edges.csv"
    vtypes_file = src_dir / "vehicle_types.xml"

    net_file.write_text("<net/>", encoding="utf-8")
    trips_file.write_text("<routes/>", encoding="utf-8")
    demand_file.write_text("ID,origin link id,destination link id,departure timestep\n", encoding="utf-8")
    observed_file.write_text("edge_id\n", encoding="utf-8")
    write_vehicle_types_xml(1.2718, vtypes_file)

    out_dir = tmp_path / "export"
    exporter = ScenarioExporter(out_dir)
    meta = {
        "run_info": {"seed": 42},
        "simulation_config": {"window_minutes": 15, "warmup_minutes": 5, "step_length_seconds": 1.0},
    }
    exporter.export(net_file, demand_file, trips_file, observed_file, meta)

    exported_vtypes = out_dir / "vehicle_types.xml"
    assert exported_vtypes.exists()

    exported_cfg = out_dir / "scenario.sumocfg"
    assert exported_cfg.exists()
    tree = ET.parse(exported_cfg)
    add_elem = tree.getroot().find("input/additional-files")
    assert add_elem is not None
    assert "vehicle_types.xml" in add_elem.get("value")

    # Also test when files are in the standard pipeline structure (sumo/ subfolder)
    out_dir2 = tmp_path / "export2"
    sumo_dir = out_dir2 / "sumo"
    sumo_dir.mkdir(parents=True)
    data_dir = out_dir2 / "data"
    data_dir.mkdir(parents=True)

    net2 = sumo_dir / "network.net.xml"
    net2.write_text("<net/>", encoding="utf-8")
    trips2 = sumo_dir / "trips.xml"
    trips2.write_text("<routes/>", encoding="utf-8")
    demand2 = data_dir / "demand.csv"
    demand2.write_text("ID,origin link id,destination link id,departure timestep\n", encoding="utf-8")
    obs2 = data_dir / "observed_edges.csv"
    obs2.write_text("edge_id\n", encoding="utf-8")
    vtypes2 = sumo_dir / "vehicle_types.xml"
    write_vehicle_types_xml(1.2718, vtypes2)

    exporter2 = ScenarioExporter(out_dir2)
    exporter2.export(net2, demand2, trips2, obs2, meta)

    exported_vtypes2 = sumo_dir / "vehicle_types.xml"
    assert exported_vtypes2.exists()
    exported_cfg2 = sumo_dir / "scenario.sumocfg"
    assert exported_cfg2.exists()
    tree2 = ET.parse(exported_cfg2)
    add_elem2 = tree2.getroot().find("input/additional-files")
    assert add_elem2 is not None
    assert "vehicle_types.xml" in add_elem2.get("value")
