"""Tests for bbox-boundary-biased OD selection."""

from collections import Counter

from demandify.sumo.demand import DemandGenerator
from demandify.sumo.network import SUMONetwork


def _write_boundary_biased_network(tmp_path):
    network_file = tmp_path / "boundary_bias.net.xml"
    network_file.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<net version="1.16">
    <location netOffset="0.00,0.00" convBoundary="0.00,0.00,100.00,100.00" origBoundary="0.0,0.0,1.0,1.0" projParameter="+proj=utm +zone=34 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"/>
    <edge id="in_w" from="n0" to="n1" priority="5" type="highway.primary">
        <lane id="in_w_0" index="0" speed="25.0" length="30.0" shape="0.0,50.0 30.0,50.0"/>
        <lane id="in_w_1" index="1" speed="25.0" length="30.0" shape="0.0,50.0 30.0,50.0"/>
    </edge>
    <edge id="in_s" from="n2" to="n3" priority="5" type="highway.primary">
        <lane id="in_s_0" index="0" speed="25.0" length="30.0" shape="50.0,0.0 50.0,30.0"/>
        <lane id="in_s_1" index="1" speed="25.0" length="30.0" shape="50.0,0.0 50.0,30.0"/>
    </edge>
    <edge id="hub_w" from="n1" to="n4" priority="3" type="highway.secondary">
        <lane id="hub_w_0" index="0" speed="15.0" length="20.0" shape="30.0,50.0 50.0,50.0"/>
    </edge>
    <edge id="hub_s" from="n3" to="n4" priority="3" type="highway.secondary">
        <lane id="hub_s_0" index="0" speed="15.0" length="20.0" shape="50.0,30.0 50.0,50.0"/>
    </edge>
    <edge id="inner_in" from="n4" to="n5" priority="3" type="highway.secondary">
        <lane id="inner_in_0" index="0" speed="15.0" length="21.2" shape="50.0,50.0 65.0,65.0"/>
    </edge>
    <edge id="inner_out" from="n5" to="n4" priority="3" type="highway.secondary">
        <lane id="inner_out_0" index="0" speed="15.0" length="21.2" shape="65.0,65.0 50.0,50.0"/>
    </edge>
    <edge id="to_e" from="n4" to="n6" priority="3" type="highway.secondary">
        <lane id="to_e_0" index="0" speed="15.0" length="20.0" shape="50.0,50.0 70.0,50.0"/>
    </edge>
    <edge id="to_n" from="n4" to="n7" priority="3" type="highway.secondary">
        <lane id="to_n_0" index="0" speed="15.0" length="20.0" shape="50.0,50.0 50.0,70.0"/>
    </edge>
    <edge id="out_e" from="n6" to="n8" priority="5" type="highway.primary">
        <lane id="out_e_0" index="0" speed="25.0" length="30.0" shape="70.0,50.0 100.0,50.0"/>
        <lane id="out_e_1" index="1" speed="25.0" length="30.0" shape="70.0,50.0 100.0,50.0"/>
    </edge>
    <edge id="out_n" from="n7" to="n9" priority="5" type="highway.primary">
        <lane id="out_n_0" index="0" speed="25.0" length="30.0" shape="50.0,70.0 50.0,100.0"/>
        <lane id="out_n_1" index="1" speed="25.0" length="30.0" shape="50.0,70.0 50.0,100.0"/>
    </edge>
    <connection from="in_w" to="hub_w" fromLane="0" toLane="0"/>
    <connection from="in_s" to="hub_s" fromLane="0" toLane="0"/>
    <connection from="hub_w" to="inner_in" fromLane="0" toLane="0"/>
    <connection from="hub_w" to="to_e" fromLane="0" toLane="0"/>
    <connection from="hub_w" to="to_n" fromLane="0" toLane="0"/>
    <connection from="hub_s" to="inner_in" fromLane="0" toLane="0"/>
    <connection from="hub_s" to="to_e" fromLane="0" toLane="0"/>
    <connection from="hub_s" to="to_n" fromLane="0" toLane="0"/>
    <connection from="inner_in" to="inner_out" fromLane="0" toLane="0"/>
    <connection from="inner_out" to="to_e" fromLane="0" toLane="0"/>
    <connection from="inner_out" to="to_n" fromLane="0" toLane="0"/>
    <connection from="to_e" to="out_e" fromLane="0" toLane="0"/>
    <connection from="to_n" to="out_n" fromLane="0" toLane="0"/>
</net>
""",
        encoding="utf-8",
    )
    return network_file


def test_od_sampling_profiles_bias_boundary_edges_without_eliminating_internal_trips(tmp_path):
    network = SUMONetwork(_write_boundary_biased_network(tmp_path))
    demand_gen = DemandGenerator(network, seed=2026)

    assert network.get_network_boundary() == (0.0, 0.0, 100.0, 100.0)

    edges = network.get_all_edges()
    profiles = demand_gen._build_od_sampling_profiles(edges)
    edge_index = profiles["edge_index"]
    edge_roles = profiles["edge_roles"]

    assert edge_roles["in_w"] == "incoming"
    assert edge_roles["in_s"] == "incoming"
    assert edge_roles["out_e"] == "outgoing"
    assert edge_roles["out_n"] == "outgoing"
    assert edge_roles["inner_in"] == "internal"
    assert edge_roles["inner_out"] == "internal"

    incoming_origin_share = sum(
        profiles["effective_origin_probs"][edge_index[edge_id]]
        for edge_id, role in edge_roles.items()
        if role == "incoming"
    )
    internal_origin_share = sum(
        profiles["effective_origin_probs"][edge_index[edge_id]]
        for edge_id, role in edge_roles.items()
        if role == "internal"
    )
    outgoing_origin_share = sum(
        profiles["effective_origin_probs"][edge_index[edge_id]]
        for edge_id, role in edge_roles.items()
        if role == "outgoing"
    )
    outgoing_destination_share = sum(
        profiles["effective_destination_probs"][edge_index[edge_id]]
        for edge_id, role in edge_roles.items()
        if role == "outgoing"
    )
    internal_destination_share = sum(
        profiles["effective_destination_probs"][edge_index[edge_id]]
        for edge_id, role in edge_roles.items()
        if role == "internal"
    )
    incoming_destination_share = sum(
        profiles["effective_destination_probs"][edge_index[edge_id]]
        for edge_id, role in edge_roles.items()
        if role == "incoming"
    )

    assert incoming_origin_share > 0.5
    assert outgoing_destination_share > 0.5
    assert incoming_origin_share > outgoing_origin_share
    assert outgoing_destination_share > incoming_destination_share
    assert internal_origin_share > 0.10
    assert internal_destination_share > 0.10

    od_pairs = demand_gen.select_od_pairs(max_od_pairs=20, min_trip_distance=0.0)
    origin_role_counts = Counter(edge_roles[origin] for origin, _destination in od_pairs)
    destination_role_counts = Counter(edge_roles[destination] for _origin, destination in od_pairs)

    assert origin_role_counts["incoming"] > origin_role_counts["internal"]
    assert destination_role_counts["outgoing"] > destination_role_counts["internal"]
    assert origin_role_counts["internal"] > 0 or destination_role_counts["internal"] > 0
