"""Tests for GA mutation sigma wiring in the pipeline."""

from types import SimpleNamespace
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

import demandify.pipeline as pipeline_module


def test_pipeline_passes_user_sigma_to_ga(monkeypatch, tmp_path):
    captured = {}

    class FakeGA:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.last_best_selection_mode = "mae_elite_pareto"
            self.last_best_mae = 1.0
            self.last_best_mae_candidate_mae = 1.0
            self.last_best_mae_candidate_teleports = 1
            self.last_best_mae_candidate_failure_rate = 0.5
            self.last_best_mae_candidate_fail_total = 2
            self.last_best_mae_candidate_magnitude = 5.0
            self.last_best_selected_mae = 1.0
            self.last_best_selected_teleports = 0
            self.last_best_selected_failure_rate = 0.25
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


def test_pipeline_rejects_zero_ga_generations(tmp_path):
    with pytest.raises(ValueError, match="ga_generations must be at least 1"):
        pipeline_module.CalibrationPipeline(
            bbox=(20.0, 50.0, 20.1, 50.1),
            window_minutes=15,
            seed=42,
            ga_generations=0,
            output_dir=tmp_path / "run_zero_generations",
            run_id="zero_generations",
        )


def test_pipeline_always_checkpoints_first_generation(monkeypatch, tmp_path):
    saved_generations = []

    class FakeGA:
        def __init__(self, **kwargs):
            self.last_best_selection_mode = "mae_elite_pareto"
            self.last_best_mae = 1.0
            self.last_best_mae_candidate_mae = 1.0
            self.last_best_mae_candidate_teleports = 1
            self.last_best_mae_candidate_failure_rate = 0.5
            self.last_best_mae_candidate_fail_total = 2
            self.last_best_mae_candidate_magnitude = 5.0
            self.last_best_selected_mae = 1.0
            self.last_best_selected_teleports = 0
            self.last_best_selected_failure_rate = 0.25
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


def test_latest_selected_export_tracks_latest_generation(monkeypatch, tmp_path):
    class FakeGA:
        def __init__(self, **kwargs):
            self.last_best_selection_mode = "mae_elite_pareto"
            self.last_best_mae = 1.0
            self.last_best_mae_candidate_mae = 1.0
            self.last_best_mae_candidate_teleports = 1
            self.last_best_mae_candidate_failure_rate = 0.5
            self.last_best_mae_candidate_fail_total = 2
            self.last_best_mae_candidate_magnitude = 5.0
            self.last_best_selected_mae = 1.0
            self.last_best_selected_teleports = 0
            self.last_best_selected_failure_rate = 0.25
            self.last_best_selected_fail_total = 0
            self.last_best_selected_magnitude = 1.0

        def optimize(self, evaluate_func, **kwargs):
            generation_callback = kwargs["generation_callback"]
            generation_callback(
                1,
                np.array([1], dtype=int),
                3.0,
                {
                    "teleports": 2,
                    "failure_rate": 0.4,
                    "fail_total": 2,
                    "magnitude": 1.0,
                    "selection_mode": "mae_elite_pareto",
                },
            )
            generation_callback(
                2,
                np.array([2], dtype=int),
                2.0,
                {
                    "teleports": 0,
                    "failure_rate": 0.1,
                    "fail_total": 1,
                    "magnitude": 2.0,
                    "selection_mode": "mae_elite_pareto",
                },
            )
            return np.array([2]), 2.0, [3.0, 2.0], []

    class FakeDemandGen:
        def genome_to_demand_csv(self, genome, od_pairs, departure_bins, output_path):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                "ID,origin link id,destination link id,departure timestep\n"
                f"trip_0,{od_pairs[0][0]},{od_pairs[0][1]},{int(genome.sum())}\n",
                encoding="utf-8",
            )

        def demand_csv_to_trips_xml(self, demand_csv, trips_file):
            trips_file.parent.mkdir(parents=True, exist_ok=True)
            trips_file.write_text(
                '<routes><trip id="0" depart="0" from="o1" to="d1" /></routes>',
                encoding="utf-8",
            )

    monkeypatch.setattr(pipeline_module, "GeneticAlgorithm", FakeGA)
    monkeypatch.setattr(
        pipeline_module,
        "get_config",
        lambda: SimpleNamespace(cache_dir=tmp_path / "cache", default_parallel_workers=1),
    )
    from demandify.export import exporter as exporter_module

    sumocfg_calls = []

    def record_write_sumocfg(**kwargs):
        sumocfg_calls.append(dict(kwargs))
        exporter_module.write_sumocfg(**kwargs)

    monkeypatch.setattr(pipeline_module, "write_sumocfg", record_write_sumocfg)

    pipeline = pipeline_module.CalibrationPipeline(
        bbox=(20.0, 50.0, 20.1, 50.1),
        window_minutes=15,
        seed=42,
        ga_checkpoint_interval=10,
        output_dir=tmp_path / "run_latest_selected",
        run_id="latest_selected",
    )

    network_file = tmp_path / "network.net.xml"
    network_file.write_text("<net></net>", encoding="utf-8")
    observed_edges = pd.DataFrame(
        {
            "edge_id": ["e1"],
            "current_speed": [30.0],
            "sumo_freeflow_speed_kmh": [50.0],
            "match_confidence": [1.0],
        }
    )

    pipeline._calibrate_demand(
        demand_gen=FakeDemandGen(),
        od_pairs=[("o1", "d1")],
        departure_bins=[(0, 60)],
        observed_edges=observed_edges,
        network_file=network_file,
    )

    latest_dir = pipeline.output_dir / "latest_selected"
    assert (latest_dir / "data" / "demand.csv").exists()
    assert (latest_dir / "sumo" / "trips.xml").exists()
    assert (latest_dir / "sumo" / "network.net.xml").exists()
    assert (latest_dir / "sumo" / "scenario.sumocfg").exists()
    assert len(sumocfg_calls) == 2
    assert sumocfg_calls[-1]["output_file"] == latest_dir / "sumo" / "scenario.sumocfg"

    latest_meta = json.loads((latest_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert latest_meta["latest_selected"]["generation"] == 2
    assert latest_meta["latest_selected"]["selected_mae"] == 2.0
    assert latest_meta["latest_selected"]["selected_teleports"] == 0


def test_initialize_demand_passes_worker_count_to_od_selection(monkeypatch, tmp_path):
    captured = {}

    class FakeDemandGenerator:
        def __init__(self, network, seed):
            captured["seed"] = seed

        def select_od_pairs(self, **kwargs):
            captured.update(kwargs)
            return [("o1", "d1")]

    monkeypatch.setattr(pipeline_module, "DemandGenerator", FakeDemandGenerator)
    monkeypatch.setattr(
        pipeline_module,
        "SUMONetwork",
        lambda _path: object(),
    )
    monkeypatch.setattr(
        pipeline_module,
        "get_config",
        lambda: SimpleNamespace(cache_dir=tmp_path / "cache", default_parallel_workers=3),
    )

    pipeline = pipeline_module.CalibrationPipeline(
        bbox=(20.0, 50.0, 20.1, 50.1),
        window_minutes=15,
        seed=42,
        output_dir=tmp_path / "run_od_workers",
        run_id="od_workers",
    )

    network_file = tmp_path / "network.net.xml"
    network_file.write_text("<net></net>", encoding="utf-8")

    _demand_gen, od_pairs, departure_bins = pipeline._initialize_demand(network_file)

    assert od_pairs == [("o1", "d1")]
    assert departure_bins
    assert captured["min_connection_paths"] == 1
    assert captured["num_workers"] == 3
