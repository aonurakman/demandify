"""Tests for GA early stopping on sustained stagnation after mutation boost."""

import argparse
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from demandify.calibration.optimizer import GeneticAlgorithm
import demandify.pipeline as pipeline_module
from demandify.cli import parse_args


def test_early_stopping_triggers_after_boosted_stagnation():
    """Verify that with early_stopping=True, stagnation persisting through boosted phase stops GA."""
    ga = GeneticAlgorithm(
        genome_size=4,
        seed=42,
        bounds=(0, 50),
        population_size=6,
        num_generations=15,
        stagnation_patience=2,
        stagnation_boost=1.5,
        early_stopping=True,
        num_workers=1,
    )

    # Constant evaluator: zero improvement across all generations.
    def evaluate_constant(individual):
        return 12.0, {"mae": 12.0, "teleports": 0, "routing_failures": 0}

    best_genome, best_loss, loss_history, gen_stats = ga.optimize(evaluate_constant)

    # Gen 1: init loss 12.0 (stagnation=0)
    # Gen 2: no improvement (stagnation=1)
    # Gen 3: no improvement (stagnation=2 >= patience -> boost triggered!)
    # Gen 4: no improvement (stagnation=3, boosted=True)
    # Gen 5: no improvement (stagnation=4 >= 2*patience, boosted=True -> early stop!)
    assert ga.early_stopped is True
    assert ga.early_stop_generation == 5
    assert len(loss_history) == 5
    assert len(gen_stats) == 5
    assert best_loss == pytest.approx(12.0)
    assert len(best_genome) == 4


def test_early_stopping_disabled_by_default():
    """Verify that when early_stopping=False, GA continues for all requested generations."""
    ga = GeneticAlgorithm(
        genome_size=4,
        seed=42,
        bounds=(0, 50),
        population_size=6,
        num_generations=8,
        stagnation_patience=2,
        stagnation_boost=1.5,
        early_stopping=False,
        num_workers=1,
    )

    def evaluate_constant(individual):
        return 15.0, {"mae": 15.0, "teleports": 0, "routing_failures": 0}

    best_genome, best_loss, loss_history, gen_stats = ga.optimize(evaluate_constant)

    assert ga.early_stopped is False
    assert ga.early_stop_generation is None
    assert len(loss_history) == 8
    assert len(gen_stats) == 8


def test_early_stopping_resets_if_improvement_found_during_boost():
    """Verify that an improvement during the boosted phase resets stagnation and delays early stopping."""
    ga = GeneticAlgorithm(
        genome_size=4,
        seed=42,
        bounds=(0, 50),
        population_size=6,
        num_generations=15,
        stagnation_patience=2,
        stagnation_boost=1.5,
        early_stopping=True,
        num_workers=1,
    )

    def evaluate_improving(individual):
        # Improve loss when boosted
        if ga._mutation_boosted:
            return 5.0, {"mae": 5.0, "teleports": 0, "routing_failures": 0}
        return 20.0, {"mae": 20.0, "teleports": 0, "routing_failures": 0}

    best_genome, best_loss, loss_history, gen_stats = ga.optimize(evaluate_improving)

    # Because improvement occurred during boost, stagnation counter was reset,
    # pushing early stopping well past the initial gen 5 cutoff.
    assert len(loss_history) > 5
    assert best_loss == pytest.approx(5.0)
    assert ga.early_stopped is True


def test_pipeline_passes_early_stopping_and_exports_metadata(monkeypatch, tmp_path):
    """Verify pipeline passes ga_early_stopping to GA and exports it in metadata."""
    captured = {}

    class FakeGA:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.early_stopped = True
            self.early_stop_generation = 4
            self.last_best_selection_mode = "mae_elite_pareto"
            self.last_best_mae = 8.5
            self.last_best_mae_candidate_mae = 8.5
            self.last_best_mae_candidate_teleports = 0
            self.last_best_mae_candidate_failure_rate = 0.0
            self.last_best_mae_candidate_fail_total = 0
            self.last_best_mae_candidate_magnitude = 100.0
            self.last_best_selected_mae = 8.5
            self.last_best_selected_teleports = 0
            self.last_best_selected_failure_rate = 0.0
            self.last_best_selected_fail_total = 0
            self.last_best_selected_missing_edges = 0
            self.last_best_selected_magnitude = 100.0

        def optimize(self, evaluate_func, **kwargs):
            return np.array([1, 2, 3]), 8.5, [10.0, 9.5, 9.0, 8.5], []

    monkeypatch.setattr(pipeline_module, "GeneticAlgorithm", FakeGA)
    monkeypatch.setattr(
        pipeline_module,
        "get_config",
        lambda: SimpleNamespace(cache_dir=tmp_path / "cache", default_parallel_workers=1),
    )
    monkeypatch.setattr(pipeline_module.CalibrationPipeline, "_setup_run_logging", lambda self: None)

    pipeline = pipeline_module.CalibrationPipeline(
        bbox=(20.0, 50.0, 20.1, 50.1),
        window_minutes=15,
        seed=42,
        ga_early_stopping=True,
        output_dir=tmp_path / "run_es",
        run_id="es_test",
    )

    assert pipeline.ga_early_stopping is True

    # Calibrate demand
    observed_edges = pd.DataFrame(
        {
            "edge_id": ["e1"],
            "current_speed": [30.0],
            "freeflow_speed": [50.0],
            "match_confidence": [1.0],
        }
    )
    od_pairs = [("o1", "d1")]
    departure_bins = [(0, 300)]
    network_file = tmp_path / "network.net.xml"
    network_file.write_text("<net/>")

    best_genome, best_loss, loss_history, gen_stats = pipeline._calibrate_demand(
        demand_gen=SimpleNamespace(),
        od_pairs=od_pairs,
        departure_bins=departure_bins,
        observed_edges=observed_edges,
        network_file=network_file,
    )

    assert captured.get("early_stopping") is True
    assert pipeline._last_optimization_result["early_stopped"] is True
    assert pipeline._last_optimization_result["early_stop_generation"] == 4

    # Prepare dummy files needed for export
    demand_csv = tmp_path / "demand.csv"
    demand_csv.write_text("ID,origin link id,destination link id,departure timestep\n")
    trips_file = tmp_path / "trips.xml"
    trips_file.write_text("<routes/>")
    observed_edges_file = tmp_path / "observed.csv"
    observed_edges_file.write_text("edge_id,current_speed,freeflow_speed,match_confidence\n")
    traffic_data_file = tmp_path / "traffic.csv"
    traffic_data_file.write_text("segment,speed\n")

    # Export results metadata
    metadata = pipeline._export_results(
        network_file=network_file,
        demand_csv=demand_csv,
        trips_file=trips_file,
        observed_edges_file=observed_edges_file,
        traffic_data_file=traffic_data_file,
        observed_edges=observed_edges,
        simulated_speeds={"e1": 30.0},
        best_loss=8.5,
        loss_history=[10.0, 9.5, 9.0, 8.5],
        quality_metrics={"mae": 8.5, "mse": 72.25, "matched_edges": 1, "missing_edges": 0, "total_edges": 1},
        generation_stats=[],
        final_sim_seed=42,
    )

    assert metadata["calibration_config"]["ga_early_stopping"] is True
    assert metadata["results"]["optimization_result"]["early_stopped"] is True
    assert metadata["results"]["optimization_result"]["early_stop_generation"] == 4
    assert metadata["user_inputs"]["ga_early_stopping"] is True
    assert "--early-stopping" in metadata["reproducibility"]["rerun_cli_command"]


def test_cli_argument_parsing_early_stopping():
    """Verify CLI parses --early-stopping and --no-early-stopping flags."""
    # Default is False
    args_default = parse_args(["run", "20.0,50.0,20.1,50.1"])
    assert args_default.ga_early_stopping is False

    # Explicit flag
    args_enabled = parse_args(["run", "20.0,50.0,20.1,50.1", "--early-stopping"])
    assert args_enabled.ga_early_stopping is True

    # Explicit negation
    args_disabled = parse_args(["run", "20.0,50.0,20.1,50.1", "--no-early-stopping"])
    assert args_disabled.ga_early_stopping is False
