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
    # Residential (50 km/h raw) derated to 24 km/h (traffic-calmed residential standard)
    assert compute_effective_freeflow_kmh(
        edge_attrs={"speed": 13.89, "type": "highway.residential"},
        obs_speed=20.0,
        empirical_freeflow=0.0,
    ) == pytest.approx(24.0)

    # Living street (50 km/h raw) derated to 10 km/h (InTAS standard)
    assert compute_effective_freeflow_kmh(
        edge_attrs={"speed": 13.89, "type": "highway.living_street"},
        obs_speed=5.0,
        empirical_freeflow=None,
    ) == pytest.approx(10.0)

    # Primary arterial capped at 50 km/h (InTAS/LuST standard)
    assert compute_effective_freeflow_kmh(
        edge_attrs={"speed": 20.0, "type": "highway.primary"},
        obs_speed=40.0,
        empirical_freeflow=None,
    ) == pytest.approx(50.0)

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
    # res -> 24.0 (derated to residential baseline), primary -> 50.0 (capped), with_empirical -> 48.0
    assert enriched["sumo_freeflow_speed_kmh"].tolist() == pytest.approx(
        [24.0, 50.0, 48.0]
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


def test_objective_decouples_active_mae_and_coverage():
    # e1 is matched and observed at 20 km/h, simulated at 24 km/h (freeflow 40 km/h)
    # active error = |24 - 20| / 40 = 0.10
    # e2 is missing (no simulated traffic), observed at 20 km/h, freeflow 50 km/h
    # missing fallback error = |50 - 20| / 50 = 0.60
    # total MAE = (0.10 + 0.60) / 2 = 0.35
    # active MAE = 0.10
    # sensor coverage = 1 / 2 = 0.50
    observed_edges = pd.DataFrame(
        {
            "edge_id": ["e1", "e2"],
            "current_speed": [20.0, 20.0],
            "freeflow_speed": [40.0, 50.0],
            "sumo_freeflow_speed_kmh": [40.0, 50.0],
            "match_confidence": [1.0, 1.0],
        }
    )

    objective = EdgeSpeedObjective(observed_edges)

    # 1. Loss components (interval-less fallback)
    components = objective.calculate_loss_components(simulated_speeds={"e1": 24.0})
    assert components["mae"] == pytest.approx(0.35)
    assert components["active_mae"] == pytest.approx(0.10)
    assert components["sensor_coverage"] == pytest.approx(0.50)
    assert components["missing_edges"] == 1

    # 2. Detailed metrics
    metrics = objective.calculate_metrics(simulated_speeds={"e1": 24.0})
    assert metrics["mae"] == pytest.approx(0.35)
    assert metrics["active_mae"] == pytest.approx(0.10)
    assert metrics["sensor_coverage"] == pytest.approx(0.50)
    assert metrics["matched_edges"] == 1
    assert metrics["missing_edges"] == 1


def test_objective_clamps_simulated_overspeeding():
    # If edge freeflow is 50, observed is 50, and simulated car travels at 55 due to speedFactor,
    # the error should be clamped to freeflow (50), resulting in zero error (|50 - 50| = 0).
    observed_edges = pd.DataFrame({
        "edge_id": ["e1"],
        "current_speed": [50.0],
        "sumo_freeflow_speed_kmh": [50.0],
        "match_confidence": [1.0],
    })
    objective = EdgeSpeedObjective(observed_edges)
    loss = objective.calculate_loss(simulated_speeds={"e1": 55.0})
    assert loss == pytest.approx(0.0)


def test_objective_auto_deduplicates_duplicate_edge_rows():
    observed_edges = pd.DataFrame({
        "edge_id": ["e1", "e1"],
        "current_speed": [20.0, 40.0],
        "sumo_freeflow_speed_kmh": [50.0, 50.0],
        "match_confidence": [0.5, 0.5],
    })
    objective = EdgeSpeedObjective(observed_edges)
    assert len(objective.observed_edges) == 1
    assert objective.observed_edges.loc["e1", "current_speed"] == pytest.approx(30.0)


def test_objective_upper_band_penalizes_error_variance():
    """Verify that upper-band loss (MAE + std) naturally penalizes bipolar error splits."""
    # Scenario A: Uniform error across 2 edges: both have |sim - obs| / freeflow = 0.30
    # Scenario B: Bipolar error across 2 edges: one has error 0.0, one has error 0.60 (same MAE = 0.30)
    obs = pd.DataFrame({
        "edge_id": ["e1", "e2"],
        "current_speed": [50.0, 50.0],
        "sumo_freeflow_speed_kmh": [100.0, 100.0],
        "match_confidence": [1.0, 1.0],
    })
    objective = EdgeSpeedObjective(obs)

    # Uniform candidate: errors are |80 - 50|/100 = 0.30 and |80 - 50|/100 = 0.30 -> MAE=0.30, std=0.0 -> loss=0.30
    comp_uniform = objective.calculate_loss_components({"e1": 80.0, "e2": 80.0})
    assert comp_uniform["mae"] == pytest.approx(0.30)
    assert comp_uniform["std"] == pytest.approx(0.0)
    assert comp_uniform["loss"] == pytest.approx(0.30)

    # Bipolar candidate: e1 error is |50 - 50|/100 = 0.0, e2 error is |110 - 50|/100 = 0.60 (clamped to 100 -> |100-50|/100 = 0.50)
    # Let's use sim_speed e2 = 10 -> |10 - 50|/100 = 0.40 -> MAE = 0.20, std = 0.20 -> loss = 0.40
    comp_bipolar = objective.calculate_loss_components({"e1": 50.0, "e2": 10.0})
    assert comp_bipolar["mae"] == pytest.approx(0.20)
    assert comp_bipolar["std"] == pytest.approx(0.20)
    assert comp_bipolar["loss"] == pytest.approx(0.40)
    # Even though bipolar has lower MAE (0.20 < 0.30), its high variance drives loss higher (0.40 > 0.30)!
    assert comp_bipolar["loss"] > comp_uniform["loss"]



