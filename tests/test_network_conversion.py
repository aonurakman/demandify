import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

from demandify.sumo.network import convert_osm_to_sumo


def test_convert_osm_to_sumo_netconvert_args(tmp_path: Path):
    osm_file = tmp_path / "test.osm"
    osm_file.write_text("<osm></osm>", encoding="utf-8")
    output_net = tmp_path / "test.net.xml"

    with patch("demandify.sumo.network.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)

        out_file, metadata = convert_osm_to_sumo(
            osm_file=osm_file,
            output_net_file=output_net,
            car_only=True,
            seed=42,
        )

        assert out_file == output_net
        cmd = mock_run.call_args[0][0]
        assert "--tls.join" in cmd
        assert "--tls.default-type" in cmd
        assert cmd[cmd.index("--tls.default-type") + 1] == "actuated"
        assert "--junctions.limit-turn-speed" in cmd
        assert cmd[cmd.index("--junctions.limit-turn-speed") + 1] == "5.50"
        assert "--no-turnarounds" in cmd
        assert cmd[cmd.index("--no-turnarounds") + 1] == "true"
        assert "--crossings.guess" in cmd
        assert "--crossings.guess.roundabout-priority" in cmd
        assert "--keep-edges.by-vclass" in cmd
        assert cmd[cmd.index("--keep-edges.by-vclass") + 1] == "passenger"
        assert "--lefthand" not in cmd
        assert metadata["lefthand"] is False

        assert "--tls.join" in metadata["netconvert_args"]
        assert "actuated" in metadata["netconvert_args"]
        assert "5.50" in metadata["netconvert_args"]
        assert "true" in metadata["netconvert_args"]


def test_convert_osm_to_sumo_lefthand_autodetection_and_override(tmp_path: Path):
    uk_osm = tmp_path / "uk.osm"
    uk_osm.write_text(
        '<osm version="0.6"><bounds minlat="53.79" minlon="-1.57" maxlat="53.81" maxlon="-1.53"/></osm>',
        encoding="utf-8",
    )
    eu_osm = tmp_path / "eu.osm"
    eu_osm.write_text(
        '<osm version="0.6"><bounds minlat="50.05" minlon="19.90" maxlat="50.09" maxlon="19.98"/></osm>',
        encoding="utf-8",
    )
    output_net = tmp_path / "out.net.xml"

    with patch("demandify.sumo.network.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)

        # 1. Auto-detect UK -> lefthand True
        _, meta_uk = convert_osm_to_sumo(uk_osm, output_net, seed=42)
        cmd_uk = mock_run.call_args[0][0]
        assert "--lefthand" in cmd_uk
        assert meta_uk["lefthand"] is True

        # 2. Auto-detect EU -> lefthand False
        _, meta_eu = convert_osm_to_sumo(eu_osm, output_net, seed=42)
        cmd_eu = mock_run.call_args[0][0]
        assert "--lefthand" not in cmd_eu
        assert meta_eu["lefthand"] is False

        # 3. Explicit override lefthand=True on EU
        _, meta_forced_lh = convert_osm_to_sumo(eu_osm, output_net, seed=42, lefthand=True)
        cmd_forced_lh = mock_run.call_args[0][0]
        assert "--lefthand" in cmd_forced_lh
        assert meta_forced_lh["lefthand"] is True

        # 4. Explicit override lefthand=False on UK
        _, meta_forced_rh = convert_osm_to_sumo(uk_osm, output_net, seed=42, lefthand=False)
        cmd_forced_rh = mock_run.call_args[0][0]
        assert "--lefthand" not in cmd_forced_rh
        assert meta_forced_rh["lefthand"] is False


def test_network_key_lefthand_cache_separation():
    from demandify.cache.keys import network_key

    k_rh = network_key("osm123", car_only=True, seed=42, lefthand=False)
    k_lh = network_key("osm123", car_only=True, seed=42, lefthand=True)
    assert k_rh != k_lh


def test_apply_road_hierarchy_speed_derating(tmp_path: Path):
    from demandify.sumo.network import apply_road_hierarchy_speed_derating
    import xml.etree.ElementTree as ET
    import pandas as pd
    import pytest

    net_xml = """<net>
    <edge id="res1" type="highway.residential">
        <lane id="res1_0" speed="13.89"/>
    </edge>
    <edge id="living1" type="highway.living_street">
        <lane id="living1_0" speed="13.89"/>
    </edge>
    <edge id="primary1" type="highway.primary">
        <lane id="primary1_0" speed="20.00"/>
    </edge>
    <edge id=":internal1" type="highway.residential">
        <lane id=":internal1_0" speed="13.89"/>
    </edge>
</net>"""
    net_file = tmp_path / "network.net.xml"
    net_file.write_text(net_xml, encoding="utf-8")

    # res1 observed at 20 km/h (below 24 km/h cap)
    observed = pd.DataFrame({"edge_id": ["res1"], "current_speed": [20.0]})
    updated = apply_road_hierarchy_speed_derating(net_file, observed)

    # res1, living1, and primary1 should be updated (all exceed their respective caps)
    # internal edge remains untouched
    assert updated == 3

    tree = ET.parse(net_file)
    root = tree.getroot()

    res_speed = float(root.find(".//edge[@id='res1']/lane").get("speed"))
    living_speed = float(root.find(".//edge[@id='living1']/lane").get("speed"))
    primary_speed = float(root.find(".//edge[@id='primary1']/lane").get("speed"))
    internal_speed = float(root.find(".//edge[@id=':internal1']/lane").get("speed"))

    # residential capped at 24 km/h = 6.67 m/s
    assert res_speed == pytest.approx(24.0 / 3.6, abs=0.01)
    # living street capped at 10 km/h = 2.78 m/s
    assert living_speed == pytest.approx(10.0 / 3.6, abs=0.01)
    # primary capped at 50 km/h = 13.89 m/s
    assert primary_speed == pytest.approx(50.0 / 3.6, abs=0.01)
    assert internal_speed == pytest.approx(13.89, abs=0.01)
    # internal unchanged
    assert internal_speed == pytest.approx(13.89, abs=0.01)

