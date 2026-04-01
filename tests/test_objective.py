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

    assert components["mae"] == 30.0
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

    assert components["mae"] == 2.0
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

    assert components["mae"] == 15.0
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
