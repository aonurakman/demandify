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
