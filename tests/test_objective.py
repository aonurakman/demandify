"""Tests for objective free-flow fallback behavior."""

import inspect

import pandas as pd
import pytest

from demandify.calibration.objective import EdgeSpeedObjective
from demandify.pipeline import CalibrationPipeline
from demandify.sumo.simulation import EdgeSpeedSnapshot


def test_missing_edge_uses_sumo_freeflow_speed_kmh():
    observed_edges = pd.DataFrame(
        {
            "edge_id": ["e1"],
            "current_speed": [20.0],
            "sumo_freeflow_speed_kmh": [50.0],
            "match_confidence": [1.0],
        }
    )

    objective = EdgeSpeedObjective(observed_edges)
    components = objective.calculate_loss_components(simulated_speeds={})

    # Missing edge: sim falls back to freeflow (50). Error = (50 - 20) / 50 = 0.60
    assert components["mae"] == pytest.approx(0.60)
    assert components["missing_edges"] == 1


def test_present_edge_keeps_measured_simulated_speed():
    observed_edges = pd.DataFrame(
        {
            "edge_id": ["e1"],
            "current_speed": [20.0],
            "sumo_freeflow_speed_kmh": [50.0],
            "match_confidence": [1.0],
        }
    )

    objective = EdgeSpeedObjective(observed_edges)
    components = objective.calculate_loss_components(simulated_speeds={"e1": 18.0})

    # Error = |18 - 20| / 50 = 0.04
    assert components["mae"] == pytest.approx(0.04)
    assert components["missing_edges"] == 0


def test_intervalwise_mae_does_not_allow_temporal_cancellation():
    observed_edges = pd.DataFrame(
        {
            "edge_id": ["e1"],
            "current_speed": [20.0],
            "sumo_freeflow_speed_kmh": [50.0],
            "match_confidence": [1.0],
        }
    )

    objective = EdgeSpeedObjective(observed_edges)
    snapshot = EdgeSpeedSnapshot(
        mean_speeds={"e1": 20.0},
        interval_speeds={"e1": {0: 35.0, 1: 5.0}},
        measurement_intervals=2,
    )

    components = objective.calculate_loss_components(simulated_speeds=snapshot)

    # Interval 0: |35 - 20| / 50 = 0.30; interval 1: |5 - 20| / 50 = 0.30; mean = 0.30
    # (Errors do not cancel because we take absolute values per interval.)
    assert components["mae"] == pytest.approx(0.30)
    assert components["missing_edges"] == 0


def test_observed_edges_freeflow_is_enriched_from_sumo_network():
    class FakeNetwork:
        def get_edge_attributes(self, edge_id):
            return {"speed": 13.89 if edge_id == "e1" else 10.0}

    observed_edges = pd.DataFrame(
        {
            "edge_id": ["e1", "e2"],
            "current_speed": [20.0, 30.0],
            "match_confidence": [1.0, 1.0],
        }
    )

    enriched = CalibrationPipeline._ensure_observed_edges_sumo_freeflow(
        observed_edges,
        FakeNetwork(),
    )

    assert "sumo_freeflow_speed_kmh" in enriched.columns
    assert enriched["sumo_freeflow_speed_kmh"].tolist() == pytest.approx([50.004, 36.0])


def test_objective_constructor_removes_unused_weight_flag():
    params = inspect.signature(EdgeSpeedObjective).parameters
    assert "weight_by_confidence" not in params


def test_compute_effective_freeflow_kmh_hierarchy():
    from demandify.sumo.network import compute_effective_freeflow_kmh

    # Tier 1: Empirical free-flow takes precedence when >= 1.0
    assert compute_effective_freeflow_kmh(
        edge_attrs={"speed": 13.89, "type": "highway.residential"},
        obs_speed=20.0,
        empirical_freeflow=42.0,
    ) == pytest.approx(42.0)

    # Tier 2: Road hierarchy derating when empirical is 0.0 or None
    # Residential (50 km/h raw) derated to 35 km/h
    assert compute_effective_freeflow_kmh(
        edge_attrs={"speed": 13.89, "type": "highway.residential"},
        obs_speed=20.0,
        empirical_freeflow=0.0,
    ) == pytest.approx(35.0)

    # Living street (50 km/h raw) derated to 20 km/h
    assert compute_effective_freeflow_kmh(
        edge_attrs={"speed": 13.89, "type": "highway.living_street"},
        obs_speed=10.0,
        empirical_freeflow=None,
    ) == pytest.approx(20.0)

    # Primary arterial retains raw speed
    assert compute_effective_freeflow_kmh(
        edge_attrs={"speed": 20.0, "type": "highway.primary"},
        obs_speed=40.0,
        empirical_freeflow=None,
    ) == pytest.approx(72.0)

    # Lower bound: Freeflow is never lower than observed speed
    assert compute_effective_freeflow_kmh(
        edge_attrs={"speed": 13.89, "type": "highway.residential"},
        obs_speed=45.0,
        empirical_freeflow=0.0,
    ) == pytest.approx(45.0)


def test_observed_edges_freeflow_with_road_types_and_empirical():
    class TypedNetwork:
        def get_edge_attributes(self, edge_id):
            if edge_id == "res":
                return {"speed": 13.89, "type": "highway.residential"}
            elif edge_id == "primary":
                return {"speed": 13.89, "type": "highway.primary"}
            return {"speed": 13.89}

    observed_edges = pd.DataFrame(
        {
            "edge_id": ["res", "primary", "with_empirical"],
            "current_speed": [20.0, 40.0, 30.0],
            "freeflow_speed": [0.0, 0.0, 48.0],
            "match_confidence": [1.0, 1.0, 1.0],
        }
    )

    enriched = CalibrationPipeline._ensure_observed_edges_sumo_freeflow(
        observed_edges,
        TypedNetwork(),
    )

    assert "sumo_freeflow_speed_kmh" in enriched.columns
    # res -> 35.0 (derated), primary -> 50.004, with_empirical -> 48.0
    assert enriched["sumo_freeflow_speed_kmh"].tolist() == pytest.approx(
        [35.0, 50.004, 48.0]
    )


def test_objective_prefers_empirical_freeflow():
    observed_edges = pd.DataFrame(
        {
            "edge_id": ["e1"],
            "current_speed": [20.0],
            "freeflow_speed": [40.0],
            "sumo_freeflow_speed_kmh": [50.0],
            "match_confidence": [1.0],
        }
    )

    objective = EdgeSpeedObjective(observed_edges)
    components = objective.calculate_loss_components(simulated_speeds={})

    # Missing edge fallback: uses empirical freeflow (40.0). Error = (40 - 20) / 40 = 0.50
    assert components["mae"] == pytest.approx(0.50)

