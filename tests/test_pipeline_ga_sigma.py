"""Tests for GA mutation sigma wiring in the pipeline."""

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd

import demandify.pipeline as pipeline_module


def test_pipeline_passes_user_sigma_to_ga(monkeypatch, tmp_path):
    captured = {}

    class FakeGA:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.last_best_selection_mode = "elite_slice_secondary"
            self.last_best_selection_value = 0.25
            self.last_best_raw_loss = 1.0
            self.last_best_selected_raw_loss = 1.0
            self.last_best_selected_e_loss = 1.0
            self.last_best_selected_fail_total = 0
            self.last_best_selected_magnitude = 0.0

        def optimize(self, evaluate_func, **kwargs):
            return np.array([0]), 1.0, [1.0], []

    monkeypatch.setattr(pipeline_module, "GeneticAlgorithm", FakeGA)
    monkeypatch.setattr(
        pipeline_module,
        "get_config",
        lambda: SimpleNamespace(cache_dir=tmp_path / "cache", default_parallel_workers=1),
    )

    pipeline = pipeline_module.CalibrationPipeline(
        bbox=(20.0, 50.0, 20.1, 50.1),
        window_minutes=15,
        seed=42,
        ga_mutation_sigma=37,
        output_dir=tmp_path / "run_sigma_passthrough",
        run_id="sigma_passthrough",
    )

    observed_edges = pd.DataFrame(
        {
            "edge_id": ["e1"],
            "current_speed": [30.0],
            "freeflow_speed": [50.0],
            "match_confidence": [1.0],
        }
    )
    od_pairs = [("o1", "d1")]
    departure_bins = [(0, 60)]

    network_file = Path(tmp_path / "network.net.xml")
    pipeline._calibrate_demand(
        demand_gen=None,
        od_pairs=od_pairs,
        departure_bins=departure_bins,
        observed_edges=observed_edges,
        network_file=network_file,
    )

    assert captured["mutation_sigma"] == 37


def test_pipeline_always_checkpoints_first_generation(monkeypatch, tmp_path):
    saved_generations = []

    class FakeGA:
        def __init__(self, **kwargs):
            self.last_best_selection_mode = "elite_slice_secondary"
            self.last_best_selection_value = 0.25
            self.last_best_raw_loss = 1.0
            self.last_best_selected_raw_loss = 1.0
            self.last_best_selected_e_loss = 1.0
            self.last_best_selected_fail_total = 0
            self.last_best_selected_magnitude = 1.0

        def optimize(self, evaluate_func, **kwargs):
            generation_callback = kwargs["generation_callback"]
            for generation in (1, 2, 10, 11):
                generation_callback(
                    generation,
                    np.array([generation], dtype=int),
                    float(generation),
                    {"generation": generation},
                )
            return np.array([1]), 1.0, [1.0], []

    monkeypatch.setattr(pipeline_module, "GeneticAlgorithm", FakeGA)
    monkeypatch.setattr(
        pipeline_module,
        "get_config",
        lambda: SimpleNamespace(cache_dir=tmp_path / "cache", default_parallel_workers=1),
    )

    pipeline = pipeline_module.CalibrationPipeline(
        bbox=(20.0, 50.0, 20.1, 50.1),
        window_minutes=15,
        seed=42,
        ga_checkpoint_interval=10,
        output_dir=tmp_path / "run_first_checkpoint",
        run_id="first_checkpoint",
    )

    monkeypatch.setattr(
        pipeline,
        "_save_generation_checkpoint",
        lambda **kwargs: saved_generations.append(kwargs["generation"]),
    )

    observed_edges = pd.DataFrame(
        {
            "edge_id": ["e1"],
            "current_speed": [30.0],
            "freeflow_speed": [50.0],
            "match_confidence": [1.0],
        }
    )

    pipeline._calibrate_demand(
        demand_gen=object(),
        od_pairs=[("o1", "d1")],
        departure_bins=[(0, 60)],
        observed_edges=observed_edges,
        network_file=Path(tmp_path / "network.net.xml"),
    )

    assert saved_generations == [1, 10]
