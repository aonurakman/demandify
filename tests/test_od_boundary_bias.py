"""Tests for bbox-boundary-biased OD selection."""

from collections import Counter

import pytest

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


def test_adaptive_boundary_gain_handles_sparse_balanced_and_extreme_cases():
    balanced_gain = DemandGenerator._compute_adaptive_boundary_gain(preferred_mass=500.0, total_mass=1000.0)
    sparse_gain = DemandGenerator._compute_adaptive_boundary_gain(preferred_mass=20.0, total_mass=1000.0)
    extreme_gain = DemandGenerator._compute_adaptive_boundary_gain(preferred_mass=1e-6, total_mass=1000.0)

    assert balanced_gain == pytest.approx(1.0, abs=1e-9)
    assert sparse_gain > 1.0
    assert sparse_gain < DemandGenerator.BOUNDARY_ADAPTIVE_GAIN_MAX
    assert extreme_gain == pytest.approx(DemandGenerator.BOUNDARY_ADAPTIVE_GAIN_MAX, abs=1e-9)

    # Robust fallback for near-zero preferred/total mass.
    assert DemandGenerator._compute_adaptive_boundary_gain(preferred_mass=0.0, total_mass=1000.0) == 1.0
    assert DemandGenerator._compute_adaptive_boundary_gain(preferred_mass=10.0, total_mass=0.0) == 1.0


def test_adaptive_boundary_bias_lifts_sparse_boundary_roles_without_collapsing_internal(tmp_path):
    network = SUMONetwork(_write_boundary_biased_network(tmp_path))
    demand_gen = DemandGenerator(network, seed=2026)

    # Synthetic skewed role distribution: few boundary edges and many internal edges.
    edges = ["in_sparse", "out_sparse"] + [f"int_{idx}" for idx in range(12)]
    edge_roles = {"in_sparse": "incoming", "out_sparse": "outgoing"}
    edge_roles.update({edge_id: "internal" for edge_id in edges[2:]})
    base_weights = [15.0, 15.0] + [120.0] * 12

    adaptive_origin = demand_gen._normalize_weights(
        demand_gen._apply_boundary_role_bias(
            edges,
            base_weights,
            edge_roles,
            preferred_role="incoming",
        )
    )
    adaptive_destination = demand_gen._normalize_weights(
        demand_gen._apply_boundary_role_bias(
            edges,
            base_weights,
            edge_roles,
            preferred_role="outgoing",
        )
    )

    def _static_probs(preferred_role: str):
        static_weights = []
        for edge_id, base_weight in zip(edges, base_weights):
            role = edge_roles[edge_id]
            factor = 1.0
            if role == preferred_role:
                factor = demand_gen.PREFERRED_BOUNDARY_ROLE_BOOST
            elif role in {"incoming", "outgoing"}:
                factor = demand_gen.OPPOSITE_BOUNDARY_ROLE_FACTOR
            static_weights.append(base_weight * factor)
        return demand_gen._normalize_weights(static_weights)

    static_origin = _static_probs(preferred_role="incoming")
    static_destination = _static_probs(preferred_role="outgoing")

    def _role_share(probabilities, role_name: str) -> float:
        return sum(
            probability
            for edge_id, probability in zip(edges, probabilities)
            if edge_roles[edge_id] == role_name
        )

    adaptive_origin_incoming = _role_share(adaptive_origin, "incoming")
    adaptive_destination_outgoing = _role_share(adaptive_destination, "outgoing")
    static_origin_incoming = _role_share(static_origin, "incoming")
    static_destination_outgoing = _role_share(static_destination, "outgoing")

    assert adaptive_origin_incoming > static_origin_incoming + 0.05
    assert adaptive_destination_outgoing > static_destination_outgoing + 0.05
    assert _role_share(adaptive_origin, "internal") > 0.50
    assert _role_share(adaptive_destination, "internal") > 0.50
