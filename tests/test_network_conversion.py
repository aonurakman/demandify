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
        assert "--crossings.guess" in cmd
        assert "--crossings.guess.roundabout-priority" in cmd
        assert "--keep-edges.by-vclass" in cmd
        assert cmd[cmd.index("--keep-edges.by-vclass") + 1] == "passenger"

        assert "--tls.join" in metadata["netconvert_args"]
        assert "actuated" in metadata["netconvert_args"]
