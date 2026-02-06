from pathlib import Path

from demandify.sumo.demand import DemandGenerator
from demandify.sumo.network import SUMONetwork


def _write_net(tmp_path: Path, xml: str) -> Path:
    net = tmp_path / "network.net.xml"
    net.write_text(xml, encoding="utf-8")
    return net


def test_has_route_respects_lane_permissions(tmp_path):
    # Edge-level connectivity exists, but the only connection is from a bus-only lane.
    net = _write_net(
        tmp_path,
        """<?xml version="1.0" encoding="UTF-8"?>
<net>
  <edge id="A" from="n0" to="n1" priority="1" type="highway.residential">
    <lane id="A_0" index="0" allow="bus" speed="13.89" length="10" shape="0,0 10,0" />
    <lane id="A_1" index="1" disallow="bus" speed="13.89" length="10" shape="0,0 10,0" />
  </edge>
  <edge id="B" from="n1" to="n2" priority="1" type="highway.residential">
    <lane id="B_0" index="0" speed="13.89" length="10" shape="10,0 20,0" />
  </edge>
  <connection from="A" to="B" fromLane="0" toLane="0" />
</net>
""",
    )

    network = SUMONetwork(net)
    demand_gen = DemandGenerator(network, seed=1)

    assert demand_gen._has_route("A", "B") is False


def test_has_route_allows_passenger_lane_connection(tmp_path):
    # Same as above, but the connection uses the passenger-allowed lane.
    net = _write_net(
        tmp_path,
        """<?xml version="1.0" encoding="UTF-8"?>
<net>
  <edge id="A" from="n0" to="n1" priority="1" type="highway.residential">
    <lane id="A_0" index="0" allow="bus" speed="13.89" length="10" shape="0,0 10,0" />
    <lane id="A_1" index="1" disallow="bus" speed="13.89" length="10" shape="0,0 10,0" />
  </edge>
  <edge id="B" from="n1" to="n2" priority="1" type="highway.residential">
    <lane id="B_0" index="0" speed="13.89" length="10" shape="10,0 20,0" />
  </edge>
  <connection from="A" to="B" fromLane="1" toLane="0" />
</net>
""",
    )

    network = SUMONetwork(net)
    demand_gen = DemandGenerator(network, seed=1)

    assert demand_gen._has_route("A", "B") is True

