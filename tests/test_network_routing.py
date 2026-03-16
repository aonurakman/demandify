from pathlib import Path

import demandify.sumo.demand as demand_module
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


def test_has_route_handles_paths_longer_than_legacy_depth_limit(tmp_path):
    chain_len = 1105
    edge_xml = []
    connection_xml = []

    for idx in range(chain_len):
        edge_xml.append(
            f"""
  <edge id="e{idx}" from="n{idx}" to="n{idx + 1}" priority="1" type="highway.residential">
    <lane id="e{idx}_0" index="0" speed="13.89" length="10" shape="{idx},0 {idx + 1},0" />
  </edge>"""
        )
        if idx < chain_len - 1:
            connection_xml.append(
                f'  <connection from="e{idx}" to="e{idx + 1}" fromLane="0" toLane="0" />'
            )

    net_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n<net>\n'
        + "\n".join(edge_xml)
        + "\n"
        + "\n".join(connection_xml)
        + "\n</net>\n"
    )
    net = _write_net(tmp_path, net_xml)

    network = SUMONetwork(net)
    demand_gen = DemandGenerator(network, seed=1)

    assert demand_gen._has_route("e0", f"e{chain_len - 1}") is True


def test_has_at_least_k_paths_counts_distinct_simple_routes(tmp_path):
    net = _write_net(
        tmp_path,
        """<?xml version="1.0" encoding="UTF-8"?>
<net>
  <edge id="start" from="n0" to="n1" priority="1" type="highway.residential">
    <lane id="start_0" index="0" speed="13.89" length="10" shape="0,0 10,0" />
  </edge>
  <edge id="upper" from="n1" to="n2" priority="1" type="highway.residential">
    <lane id="upper_0" index="0" speed="13.89" length="10" shape="10,0 20,10" />
  </edge>
  <edge id="lower" from="n1" to="n3" priority="1" type="highway.residential">
    <lane id="lower_0" index="0" speed="13.89" length="10" shape="10,0 20,-10" />
  </edge>
  <edge id="dest" from="n2" to="n4" priority="1" type="highway.residential">
    <lane id="dest_0" index="0" speed="13.89" length="10" shape="20,10 30,0" />
  </edge>
  <edge id="dest_alt" from="n3" to="n4" priority="1" type="highway.residential">
    <lane id="dest_alt_0" index="0" speed="13.89" length="10" shape="20,-10 30,0" />
  </edge>
  <edge id="finish" from="n4" to="n5" priority="1" type="highway.residential">
    <lane id="finish_0" index="0" speed="13.89" length="10" shape="30,0 40,0" />
  </edge>
  <connection from="start" to="upper" fromLane="0" toLane="0" />
  <connection from="start" to="lower" fromLane="0" toLane="0" />
  <connection from="upper" to="dest" fromLane="0" toLane="0" />
  <connection from="lower" to="dest_alt" fromLane="0" toLane="0" />
  <connection from="dest" to="finish" fromLane="0" toLane="0" />
  <connection from="dest_alt" to="finish" fromLane="0" toLane="0" />
</net>
""",
    )

    network = SUMONetwork(net)
    demand_gen = DemandGenerator(network, seed=1)

    assert demand_gen.has_at_least_k_paths("start", "finish", 2) is True
    assert demand_gen.has_at_least_k_paths("start", "finish", 3) is False
    assert demand_gen._k_path_cache[("start", "finish", 2)] is True
    assert demand_gen._k_path_cache[("start", "finish", 3)] is False


def test_select_od_pairs_with_min_connection_paths_preserves_reachability_mode(tmp_path):
    net = _write_net(
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

    network = SUMONetwork(net)
    demand_gen = DemandGenerator(network, seed=7)

    od_pairs = demand_gen.select_od_pairs(
        max_od_pairs=1,
        max_consecutive_failures=100,
        min_trip_distance=0.0,
        min_connection_paths=1,
    )

    assert len(od_pairs) == 1
    origin, destination = od_pairs[0]
    assert demand_gen._has_route(origin, destination) is True


def test_select_od_pairs_emits_progress_logs_for_small_runs(tmp_path, caplog):
    net = _write_net(
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

    network = SUMONetwork(net)
    demand_gen = DemandGenerator(network, seed=7)

    with caplog.at_level("INFO"):
        demand_gen.select_od_pairs(
            max_od_pairs=1,
            max_consecutive_failures=100,
            min_trip_distance=0.0,
            min_connection_paths=1,
        )

    assert "OD selection [" in caplog.text


def test_preferred_mp_start_method_uses_spawn_on_macos(monkeypatch):
    monkeypatch.setattr(demand_module.sys, "platform", "darwin")
    monkeypatch.setattr(demand_module, "get_all_start_methods", lambda: ["fork", "spawn"])

    assert DemandGenerator._preferred_mp_start_method() == "spawn"


def test_preferred_mp_start_method_prefers_fork_off_macos(monkeypatch):
    monkeypatch.setattr(demand_module.sys, "platform", "linux")
    monkeypatch.setattr(demand_module, "get_all_start_methods", lambda: ["spawn", "fork"])

    assert DemandGenerator._preferred_mp_start_method() == "fork"


def test_has_at_least_k_paths_respects_search_cap(tmp_path, monkeypatch):
    net = _write_net(
        tmp_path,
        """<?xml version="1.0" encoding="UTF-8"?>
<net>
  <edge id="start" from="n0" to="n1" priority="1" type="highway.residential">
    <lane id="start_0" index="0" speed="13.89" length="10" shape="0,0 10,0" />
  </edge>
  <edge id="upper" from="n1" to="n2" priority="1" type="highway.residential">
    <lane id="upper_0" index="0" speed="13.89" length="10" shape="10,0 20,10" />
  </edge>
  <edge id="lower" from="n1" to="n3" priority="1" type="highway.residential">
    <lane id="lower_0" index="0" speed="13.89" length="10" shape="10,0 20,-10" />
  </edge>
  <edge id="dest" from="n2" to="n4" priority="1" type="highway.residential">
    <lane id="dest_0" index="0" speed="13.89" length="10" shape="20,10 30,0" />
  </edge>
  <edge id="dest_alt" from="n3" to="n4" priority="1" type="highway.residential">
    <lane id="dest_alt_0" index="0" speed="13.89" length="10" shape="20,-10 30,0" />
  </edge>
  <edge id="finish" from="n4" to="n5" priority="1" type="highway.residential">
    <lane id="finish_0" index="0" speed="13.89" length="10" shape="30,0 40,0" />
  </edge>
  <connection from="start" to="upper" fromLane="0" toLane="0" />
  <connection from="start" to="lower" fromLane="0" toLane="0" />
  <connection from="upper" to="dest" fromLane="0" toLane="0" />
  <connection from="lower" to="dest_alt" fromLane="0" toLane="0" />
  <connection from="dest" to="finish" fromLane="0" toLane="0" />
  <connection from="dest_alt" to="finish" fromLane="0" toLane="0" />
</net>
""",
    )

    network = SUMONetwork(net)
    demand_gen = DemandGenerator(network, seed=1)
    monkeypatch.setattr(demand_gen, "K_PATH_SEARCH_STATE_LIMIT", 1)

    assert demand_gen.has_at_least_k_paths("start", "finish", 2) is False


def test_select_od_pairs_rejects_pairs_without_enough_connection_paths(tmp_path):
    net = _write_net(
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

    network = SUMONetwork(net)
    demand_gen = DemandGenerator(network, seed=7)

    try:
        demand_gen.select_od_pairs(
            max_od_pairs=1,
            max_consecutive_failures=50,
            min_trip_distance=0.0,
            min_connection_paths=2,
        )
    except ValueError as exc:
        assert "min_connection_paths=2" in str(exc)
    else:
        raise AssertionError("Expected select_od_pairs to reject OD sampling when no pair has 2 paths")
