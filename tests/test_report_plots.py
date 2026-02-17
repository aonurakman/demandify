"""Tests for enhanced report generation and plot outputs."""
import numpy as np
import pandas as pd
import pytest

from demandify.export.report import ReportGenerator


@pytest.fixture
def observed_edges():
    """Sample observed edges DataFrame."""
    return pd.DataFrame({
        'edge_id': ['e1', 'e2', 'e3', 'e4', 'e5'],
        'current_speed': [30.0, 50.0, 20.0, 60.0, 40.0],
        'freeflow_speed': [50.0, 60.0, 50.0, 70.0, 50.0],
        'match_confidence': [0.9, 0.8, 0.7, 0.95, 0.85],
    })


@pytest.fixture
def simulated_speeds():
    """Sample simulated speeds dict."""
    return {
        'e1': 35.0,
        'e2': 45.0,
        'e3': 25.0,
        # e4 missing -> zero-flow
        'e5': 38.0,
    }


@pytest.fixture
def generation_stats():
    """Sample generation statistics."""
    rng = np.random.RandomState(42)
    stats = []
    for gen in range(1, 6):
        stats.append({
            'generation': gen,
            'best_loss': 50.0 - gen * 5 + rng.normal(0, 1),
            'mean_loss': 80.0 - gen * 4 + rng.normal(0, 2),
            'std_loss': 15.0 - gen * 1.5 + abs(rng.normal(0, 1)),
            'best_magnitude': 100 + gen * 10,
            'mean_magnitude': 200 + gen * 5,
            'best_zero_flow': max(0, 10 - gen * 2),
            'mean_zero_flow': max(0.0, 12 - gen * 1.5),
            'best_routing_failures': max(0, 8 - gen),
            'mean_routing_failures': max(0.0, 10 - gen * 0.8),
        })
    return stats


@pytest.fixture
def metadata():
    """Sample metadata dict."""
    return {
        'run_info': {
            'timestamp': '2026-01-01T00:00:00',
            'bbox_coordinates': {'west': 0, 'south': 0, 'east': 1, 'north': 1},
            'seed': 42,
        },
        'simulation_config': {'window_minutes': 10},
        'calibration_config': {'ga_population': 20, 'ga_generations': 5},
        'results': {
            'final_loss_mae_kmh': 25.0,
            'quality_metrics': {
                'matched_edges': 4,
                'total_observed_edges': 5,
            },
        },
    }


def test_report_generates_with_generation_stats(
    tmp_path, observed_edges, simulated_speeds, metadata, generation_stats
):
    """Report generates all plots when generation_stats are provided."""
    loss_history = [s['best_loss'] for s in generation_stats]

    gen = ReportGenerator(tmp_path)
    report_path = gen.generate(
        observed_edges, simulated_speeds, loss_history, metadata, generation_stats
    )

    assert report_path.exists()

    # Check all expected plots exist
    assert (tmp_path / "plots" / "loss_plot.png").exists()
    assert (tmp_path / "plots" / "speed_comparison.png").exists()
    assert (tmp_path / "plots" / "failures_plot.png").exists()
    assert (tmp_path / "plots" / "magnitude_plot.png").exists()

    # Check CSV data export
    assert (tmp_path / "data" / "speed_comparison.csv").exists()

    # Check HTML contains new sections
    html = report_path.read_text()
    assert "GA Population Statistics" in html
    assert "Failures" in html
    assert "Genome Magnitude" in html


def test_report_generates_without_generation_stats(
    tmp_path, observed_edges, simulated_speeds, metadata
):
    """Report still works when generation_stats is None (backward compat)."""
    loss_history = [50.0, 40.0, 30.0]

    gen = ReportGenerator(tmp_path)
    report_path = gen.generate(
        observed_edges, simulated_speeds, loss_history, metadata
    )

    assert report_path.exists()
    assert (tmp_path / "plots" / "loss_plot.png").exists()
    assert (tmp_path / "plots" / "speed_comparison.png").exists()

    # Extra plots should NOT exist
    assert not (tmp_path / "plots" / "failures_plot.png").exists()
    assert not (tmp_path / "plots" / "magnitude_plot.png").exists()


def test_speed_comparison_includes_stats(
    tmp_path, observed_edges, simulated_speeds
):
    """Speed comparison CSV contains R² and RMSE data in the plot."""
    gen = ReportGenerator(tmp_path)
    plot_path = gen._create_speed_comparison(observed_edges, simulated_speeds)

    assert plot_path == "plots/speed_comparison.png"
    assert (tmp_path / plot_path).exists()

    # Check CSV was saved
    csv_path = tmp_path / "data" / "speed_comparison.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert 'status' in df.columns
    assert (df['status'] == 'missing_in_sim').sum() == 1  # e4 missing


def test_loss_plot_with_mean_stddev(tmp_path, generation_stats):
    """Loss plot renders with mean ± stddev when stats provided."""
    loss_history = [s['best_loss'] for s in generation_stats]

    gen = ReportGenerator(tmp_path)
    plot_path = gen._create_loss_plot(loss_history, generation_stats)

    assert plot_path == "plots/loss_plot.png"
    assert (tmp_path / plot_path).exists()


def test_failures_plot_with_no_data(tmp_path):
    """Failures plot returns None when no failure data available."""
    stats = [
        {'generation': 1, 'best_zero_flow': None, 'mean_zero_flow': None,
         'best_routing_failures': None, 'mean_routing_failures': None}
    ]
    gen = ReportGenerator(tmp_path)
    result = gen._create_failures_plot(stats)
    assert result is None


def test_magnitude_plot_with_no_data(tmp_path):
    """Magnitude plot returns None when no magnitude data available."""
    stats = [{'generation': 1, 'best_magnitude': None, 'mean_magnitude': None}]
    gen = ReportGenerator(tmp_path)
    result = gen._create_magnitude_plot(stats)
    assert result is None


def test_pipeline_metadata_separates_final_mae_and_optimization_result(tmp_path, monkeypatch):
    """final_loss_mae_kmh should reflect final simulation MAE, not GA optimization score."""
    from demandify.pipeline import CalibrationPipeline

    class _DummyScenarioExporter:
        def __init__(self, *_args, **_kwargs):
            pass

        def export(self, *_args, **_kwargs):
            return None

    class _DummyURBExporter:
        def __init__(self, *_args, **_kwargs):
            pass

        def export(self, *_args, **_kwargs):
            return None

    class _DummyReportGenerator:
        def __init__(self, *_args, **_kwargs):
            pass

        def generate(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr("demandify.pipeline.ScenarioExporter", _DummyScenarioExporter)
    monkeypatch.setattr("demandify.pipeline.URBDataExporter", _DummyURBExporter)
    monkeypatch.setattr("demandify.pipeline.ReportGenerator", _DummyReportGenerator)
    monkeypatch.setattr("demandify.export.visualize.visualize_network", lambda *_args, **_kwargs: None)

    pipeline = CalibrationPipeline(
        bbox=(0.0, 0.0, 1.0, 1.0),
        window_minutes=10,
        seed=42,
        output_dir=tmp_path / "run_semantics",
        run_id="semantics",
    )

    network_file = pipeline.output_dir / "sumo" / "network.net.xml"
    network_file.write_text("<net></net>", encoding="utf-8")
    demand_csv = pipeline.output_dir / "data" / "demand.csv"
    demand_csv.write_text("ID,origin link id,destination link id,departure timestep\n", encoding="utf-8")
    trips_file = pipeline.output_dir / "sumo" / "trips.xml"
    trips_file.write_text("<routes></routes>", encoding="utf-8")
    observed_edges_file = pipeline.output_dir / "data" / "observed_edges.csv"
    observed_edges_file.write_text("edge_id,current_speed,freeflow_speed,match_confidence\n", encoding="utf-8")
    traffic_data_file = pipeline.output_dir / "data" / "traffic_data_raw.csv"
    pd.DataFrame({"segment": [1]}).to_csv(traffic_data_file, index=False)

    observed_edges = pd.DataFrame(
        {
            "edge_id": ["e1"],
            "current_speed": [30.0],
            "freeflow_speed": [50.0],
            "match_confidence": [0.9],
        }
    )
    simulated_speeds = {"e1": 35.0}
    quality_metrics = {
        "mae": 12.34,
        "mse": 200.0,
        "matched_edges": 1,
        "missing_edges": 0,
        "total_edges": 1,
    }
    pipeline._last_optimization_result = {
        "selected_mode": "feasible",
        "selected_value": 5.55,
        "selected_value_label": "feasible flow-fit error E",
        "best_raw_loss": 9.99,
        "best_feasible_e_loss": 5.55,
        "loss_history_metric": "best_loss (raw objective per generation)",
    }

    metadata = pipeline._export_results(
        network_file=network_file,
        demand_csv=demand_csv,
        trips_file=trips_file,
        observed_edges_file=observed_edges_file,
        traffic_data_file=traffic_data_file,
        observed_edges=observed_edges,
        simulated_speeds=simulated_speeds,
        best_loss=9.99,
        loss_history=[9.99, 8.88],
        quality_metrics=quality_metrics,
        generation_stats=[],
        final_sim_seed=123,
    )

    results = metadata["results"]
    assert results["final_loss_mae_kmh"] == 12.34
    assert results["loss_history"] == [9.99, 8.88]
    assert results["optimization_result"]["selected_mode"] == "feasible"
    assert results["optimization_result"]["selected_value"] == 5.55
    assert results["optimization_result"]["best_raw_loss"] == 9.99

    user_inputs = metadata["user_inputs"]
    assert user_inputs["run_id"] == "semantics"
    assert user_inputs["window_minutes"] == 10
    assert user_inputs["traffic_tile_zoom"] == 12
    assert user_inputs["initial_population"] == 1000
    assert user_inputs["parallel_workers"] is None

    assert metadata["simulation_config"]["traffic_tile_zoom"] == 12
    assert metadata["calibration_config"]["requested_parallel_workers"] is None

    rerun_cmd = metadata["reproducibility"]["rerun_cli_command"]
    assert rerun_cmd.startswith("demandify run ")
    assert "--window 10" in rerun_cmd
    assert "--warmup 5" in rerun_cmd
    assert "--seed 42" in rerun_cmd
    assert "--tile-zoom 12" in rerun_cmd
    assert "--initial-population 1000" in rerun_cmd
    assert "--name semantics" in rerun_cmd
    assert "--workers" not in rerun_cmd
