from pathlib import Path

from demandify.sumo.simulation import SUMOSimulation


def _sim_for_parse() -> SUMOSimulation:
    return SUMOSimulation(
        network_file=Path("network.net.xml"),
        vehicle_file=Path("trips.xml"),
        use_dynamic_routing=False,
    )


def test_parse_statistics_reads_dedicated_teleports_tag(tmp_path):
    stats_xml = tmp_path / "statistics.xml"
    stats_xml.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<statistics>
  <vehicles loaded="100" inserted="90" running="10" waiting="5" />
  <teleports total="7" jam="3" yield="4" wrongLane="0" />
</statistics>
""",
        encoding="utf-8",
    )

    stats = _sim_for_parse()._parse_statistic_output(stats_xml)

    assert stats["loaded"] == 100
    assert stats["inserted"] == 90
    assert stats["running"] == 10
    assert stats["waiting"] == 5
    assert stats["teleports"] == 7


def test_parse_statistics_keeps_legacy_vehicle_teleports(tmp_path):
    stats_xml = tmp_path / "statistics.xml"
    stats_xml.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<statistics>
  <vehicles loaded="20" inserted="20" running="0" waiting="0" teleports="2" />
</statistics>
""",
        encoding="utf-8",
    )

    stats = _sim_for_parse()._parse_statistic_output(stats_xml)

    assert stats["teleports"] == 2


def test_parse_statistics_prefers_dedicated_teleports_when_both_exist(tmp_path):
    stats_xml = tmp_path / "statistics.xml"
    stats_xml.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<statistics>
  <vehicles loaded="20" inserted="20" running="0" waiting="0" teleports="2" />
  <teleports total="9" jam="1" yield="8" wrongLane="0" />
</statistics>
""",
        encoding="utf-8",
    )

    stats = _sim_for_parse()._parse_statistic_output(stats_xml)

    assert stats["teleports"] == 9


def test_resolve_routing_failures_uses_expected_vehicles_when_available():
    failures, total = SUMOSimulation._resolve_routing_failures(
        expected_vehicles=100,
        loaded=80,
        inserted=75,
    )

    assert failures == 25
    assert total == 100


def test_resolve_routing_failures_falls_back_to_loaded_minus_inserted():
    failures, total = SUMOSimulation._resolve_routing_failures(
        expected_vehicles=None,
        loaded=80,
        inserted=75,
    )

    assert failures == 5
    assert total == 80


def test_parse_edge_data_keeps_interval_traces_for_objective(tmp_path):
    edge_xml = tmp_path / "edge_data.xml"
    edge_xml.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<meandata>
  <interval begin="0.00" end="60.00">
    <edge id="e1" speed="10.00" />
    <edge id="e2" speed="-1.00" />
  </interval>
  <interval begin="60.00" end="120.00">
    <edge id="e1" speed="-1.00" />
    <edge id="e2" speed="5.00" />
  </interval>
</meandata>
""",
        encoding="utf-8",
    )

    sim = SUMOSimulation(
        network_file=Path("network.net.xml"),
        vehicle_file=Path("trips.xml"),
        use_dynamic_routing=False,
        warmup_time=0,
        simulation_time=120,
    )

    stats = sim._parse_edge_data(edge_xml)

    assert stats["e1"] == 36.0
    assert stats["e2"] == 18.0
    assert stats.measurement_intervals == 2
    assert stats.interval_speeds["e1"] == {0: 36.0}
    assert stats.interval_speeds["e2"] == {1: 18.0}
