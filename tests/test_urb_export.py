"""Tests for URB-style export compatibility."""

import csv
import logging
import xml.etree.ElementTree as ET

from demandify.export.custom_formats import URBDataExporter


def _write_network(tmp_path, xml_text: str):
    path = tmp_path / "network.net.xml"
    path.write_text(xml_text, encoding="utf-8")
    return path


def _write_trips(tmp_path, xml_text: str):
    path = tmp_path / "trips.xml"
    path.write_text(xml_text, encoding="utf-8")
    return path


def test_urb_export_writes_net_like_rou_xml(tmp_path, monkeypatch):
    network_file = _write_network(
        tmp_path,
        """<?xml version="1.0" encoding="UTF-8"?>
<net>
  <edge id="A" from="n0" to="n1" priority="1" type="highway.residential">
    <lane id="A_0" index="0" speed="13.89" length="10" shape="0,0 10,0" />
  </edge>
  <edge id="B" from="n1" to="n2" priority="1" type="highway.residential">
    <lane id="B_0" index="0" speed="13.89" length="10" shape="10,0 20,0" />
  </edge>
  <connection from="A" to="B" fromLane="0" toLane="0" />
</net>
""",
    )
    trips_file = _write_trips(
        tmp_path,
        """<?xml version="1.0" encoding="UTF-8"?>
<routes>
  <trip id="t0" depart="0" from="A" to="B" />
</routes>
""",
    )

    exporter = URBDataExporter("scenario_ok", tmp_path / "export_root")
    monkeypatch.setattr(exporter, "_generate_plain_xml", lambda *_args, **_kwargs: None)

    exporter.export(network_file, trips_file, min_connection_paths=1)

    rou_file = tmp_path / "export_root" / "scenario_ok" / "scenario_ok.rou.xml"
    assert rou_file.exists()
    assert ET.parse(rou_file).getroot().tag == "net"


def test_urb_export_writes_sequential_integer_agent_ids(tmp_path, monkeypatch):
    network_file = _write_network(
        tmp_path,
        """<?xml version="1.0" encoding="UTF-8"?>
<net>
  <edge id="A" from="n0" to="n1" priority="1" type="highway.residential">
    <lane id="A_0" index="0" speed="13.89" length="10" shape="0,0 10,0" />
  </edge>
  <edge id="B" from="n1" to="n2" priority="1" type="highway.residential">
    <lane id="B_0" index="0" speed="13.89" length="10" shape="10,0 20,0" />
  </edge>
  <connection from="A" to="B" fromLane="0" toLane="0" />
</net>
""",
    )
    trips_file = _write_trips(
        tmp_path,
        """<?xml version="1.0" encoding="UTF-8"?>
<routes>
  <trip id="trip_950" depart="0" from="A" to="B" />
  <trip id="trip_12" depart="5" from="A" to="B" />
  <trip id="abc" depart="9" from="A" to="B" />
</routes>
""",
    )

    exporter = URBDataExporter("scenario_ids", tmp_path / "export_root")
    monkeypatch.setattr(exporter, "_generate_plain_xml", lambda *_args, **_kwargs: None)

    exporter.export(network_file, trips_file, min_connection_paths=1)

    agents_file = tmp_path / "export_root" / "scenario_ids" / "agents.csv"
    with agents_file.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["id"] for row in rows] == ["0", "1", "2"]


def test_urb_export_fails_cleanly_when_min_connection_paths_not_met(tmp_path, monkeypatch, caplog):
    network_file = _write_network(
        tmp_path,
        """<?xml version="1.0" encoding="UTF-8"?>
<net>
  <edge id="A" from="n0" to="n1" priority="1" type="highway.residential">
    <lane id="A_0" index="0" speed="13.89" length="10" shape="0,0 10,0" />
  </edge>
  <edge id="B" from="n1" to="n2" priority="1" type="highway.residential">
    <lane id="B_0" index="0" speed="13.89" length="10" shape="10,0 20,0" />
  </edge>
  <edge id="C" from="n2" to="n3" priority="1" type="highway.residential">
    <lane id="C_0" index="0" speed="13.89" length="10" shape="20,0 30,0" />
  </edge>
  <connection from="A" to="B" fromLane="0" toLane="0" />
  <connection from="B" to="C" fromLane="0" toLane="0" />
</net>
""",
    )
    trips_file = _write_trips(
        tmp_path,
        """<?xml version="1.0" encoding="UTF-8"?>
<routes>
  <trip id="t0" depart="0" from="A" to="C" />
</routes>
""",
    )

    exporter = URBDataExporter("scenario_bad", tmp_path / "export_root")
    monkeypatch.setattr(exporter, "_generate_plain_xml", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(logging.getLogger("demandify"), "propagate", True)

    exporter.export(network_file, trips_file, min_connection_paths=2)

    target_dir = tmp_path / "export_root" / "scenario_bad"
    assert not (target_dir / "agents.csv").exists()
    assert "min_connection_paths=2" in caplog.text


def test_urb_export_writes_demand_routes_csv_from_vehroutes_and_fallback(tmp_path, monkeypatch):
    import pytest

    network_file = _write_network(
        tmp_path,
        """<?xml version="1.0" encoding="UTF-8"?>
<net>
  <edge id="A" from="n0" to="n1" priority="1" type="highway.residential">
    <lane id="A_0" index="0" speed="10.0" length="100.0" shape="0,0 10,0" />
  </edge>
  <edge id="B1" from="n1" to="n2" priority="1" type="highway.residential">
    <lane id="B1_0" index="0" speed="20.0" length="100.0" shape="10,0 20,5" />
  </edge>
  <edge id="B2" from="n1" to="n2" priority="1" type="highway.residential">
    <lane id="B2_0" index="0" speed="5.0" length="100.0" shape="10,0 20,-5" />
  </edge>
  <edge id="C" from="n2" to="n3" priority="1" type="highway.residential">
    <lane id="C_0" index="0" speed="10.0" length="50.0" shape="20,0 30,0" />
  </edge>
  <connection from="A" to="B1" fromLane="0" toLane="0" />
  <connection from="A" to="B2" fromLane="0" toLane="0" />
  <connection from="B1" to="C" fromLane="0" toLane="0" />
  <connection from="B2" to="C" fromLane="0" toLane="0" />
</net>
""",
    )
    trips_file = _write_trips(
        tmp_path,
        """<?xml version="1.0" encoding="UTF-8"?>
<routes>
  <trip id="trip_10" depart="0" from="A" to="C" />
  <trip id="trip_20" depart="5" from="A" to="C" />
</routes>
""",
    )
    # trip_10 took the slower congestion-adapted route A -> B2 -> C in SUMO;
    # trip_20 is omitted from vehroutes.xml (simulating an uninserted backlog vehicle)
    # and should fall back to the free-flow shortest path A -> B1 -> C.
    vehroutes_file = tmp_path / "vehroutes.xml"
    vehroutes_file.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<routes>
  <vehicle id="trip_10" depart="0.00">
    <route edges="A B2 C" />
  </vehicle>
</routes>
""",
        encoding="utf-8",
    )

    exporter = URBDataExporter("scenario_routes", tmp_path / "export_root")
    monkeypatch.setattr(exporter, "_generate_plain_xml", lambda *_args, **_kwargs: None)

    exporter.export(network_file, trips_file, min_connection_paths=1, vehroutes_file=vehroutes_file)

    demand_routes_file = tmp_path / "export_root" / "scenario_routes" / "demand_routes.csv"
    assert demand_routes_file.exists()

    with demand_routes_file.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert rows[0]["agent_id"] == "0"
    assert rows[0]["source_id"] == "trip_10"
    assert rows[0]["route_id"] == "0"
    assert float(rows[0]["probability"]) == pytest.approx(1.0)
    assert rows[0]["edges"] == "A B2 C"
    # A (100/10=10s) + B2 (100/5=20s) + C (50/10=5s) = 35.0s
    assert float(rows[0]["free_flow_time"]) == pytest.approx(35.0)

    assert rows[1]["agent_id"] == "1"
    assert rows[1]["source_id"] == "trip_20"
    assert rows[1]["route_id"] == "0"
    assert float(rows[1]["probability"]) == pytest.approx(1.0)
    assert rows[1]["edges"] == "A B1 C"
    # A (100/10=10s) + B1 (100/20=5s) + C (50/10=5s) = 20.0s
    assert float(rows[1]["free_flow_time"]) == pytest.approx(20.0)
