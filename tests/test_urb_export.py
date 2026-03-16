"""Tests for URB-style export compatibility."""

import csv
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

    with caplog.at_level("ERROR"):
        exporter.export(network_file, trips_file, min_connection_paths=2)

    target_dir = tmp_path / "export_root" / "scenario_bad"
    assert not (target_dir / "agents.csv").exists()
    assert "min_connection_paths=2" in caplog.text
