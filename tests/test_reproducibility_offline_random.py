"""Integration test: offline import-mode reproducibility on a random dataset."""

import asyncio
import random
import shutil
from pathlib import Path
from typing import Optional

import pytest

from demandify.offline_data import get_offline_dataset_catalog, resolve_offline_dataset
from demandify.pipeline import CalibrationPipeline


def _sumo_available() -> bool:
    """Return True when SUMO binary is available in PATH."""
    return shutil.which("sumo") is not None


def _pick_random_dataset_id_with_observed_edges() -> Optional[str]:
    """Pick a random offline dataset id that has at least one observed edge row."""
    catalog = get_offline_dataset_catalog(include_generated=True, include_packaged=True)
    if not catalog:
        return None

    candidates = [item["id"] for item in catalog if item.get("id")]
    random.shuffle(candidates)

    for dataset_id in candidates:
        resolved = resolve_offline_dataset(dataset_id)
        observed_edges_path = resolved.root.joinpath("data/observed_edges.csv")
        try:
            with observed_edges_path.open("r", encoding="utf-8") as f:
                # Header + at least one row
                line_count = sum(1 for _ in f)
            if line_count > 1:
                return dataset_id
        except Exception:
            continue
    return None


async def _run_tiny_import_calibration(dataset_id: str, output_dir: Path, run_id: str) -> dict:
    """Run a very small import-mode calibration and capture reproducibility artifacts."""
    pipeline = CalibrationPipeline(
        bbox=None,
        offline_dataset=dataset_id,
        window_minutes=5,
        warmup_minutes=1,
        step_length=1.0,
        seed=2026,
        parallel_workers=1,
        ga_population=4,
        ga_generations=1,
        ga_mutation_rate=0.5,
        ga_crossover_rate=0.7,
        ga_elitism=1,
        ga_mutation_sigma=2,
        ga_mutation_indpb=0.3,
        ga_immigrant_rate=0.0,
        ga_elite_top_pct=0.1,
        ga_magnitude_penalty_weight=0.01,
        ga_stagnation_patience=20,
        ga_stagnation_boost=1.2,
        ga_checkpoint_interval=9999,  # keep checkpointing out of this tiny test
        num_origins=3,
        num_destinations=3,
        max_od_pairs=10,
        bin_minutes=3,
        initial_population=50,
        output_dir=output_dir,
        run_id=run_id,
    )

    metadata = await pipeline.run(confirm_callback=lambda _stats: True)
    demand_csv_path = output_dir / "data" / "demand.csv"
    trips_xml_path = output_dir / "sumo" / "trips.xml"
    with demand_csv_path.open("r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f)
    # CSV has a header line; never return negative counts.
    trip_count = max(0, line_count - 1)

    return {
        "metadata": metadata,
        "demand_csv_text": demand_csv_path.read_text(encoding="utf-8"),
        "trips_xml_text": trips_xml_path.read_text(encoding="utf-8"),
        "trip_count": int(trip_count),
        "loss_history": metadata["results"]["loss_history"],
        "selected_mode": metadata["results"]["optimization_result"]["selected_mode"],
        "selected_value": metadata["results"]["optimization_result"]["selected_value"],
        "best_raw_loss": metadata["results"]["optimization_result"]["best_raw_loss"],
        "best_feasible_e_loss": metadata["results"]["optimization_result"]["best_feasible_e_loss"],
    }


def test_random_offline_dataset_reproducibility(tmp_path):
    """Two tiny runs with identical params on a random offline dataset should match byte-for-byte."""
    if not _sumo_available():
        pytest.skip("SUMO is not available in PATH")

    dataset_id = _pick_random_dataset_id_with_observed_edges()
    if not dataset_id:
        pytest.skip("No offline datasets with observed edges are available")
    else:
        print(f"Selected random dataset for reproducibility test: {dataset_id}")

    run1 = asyncio.run(
        _run_tiny_import_calibration(
            dataset_id=dataset_id,
            output_dir=tmp_path / "run_repro_1",
            run_id="repro_1",
        )
    )
    run2 = asyncio.run(
        _run_tiny_import_calibration(
            dataset_id=dataset_id,
            output_dir=tmp_path / "run_repro_2",
            run_id="repro_2",
        )
    )

    assert run1["trip_count"] == run2["trip_count"]
    assert run1["demand_csv_text"] == run2["demand_csv_text"]
    assert run1["trips_xml_text"] == run2["trips_xml_text"]
    assert run1["loss_history"] == run2["loss_history"]
    assert run1["selected_mode"] == run2["selected_mode"]
    assert run1["selected_value"] == run2["selected_value"]
    assert run1["best_raw_loss"] == run2["best_raw_loss"]
    assert run1["best_feasible_e_loss"] == run2["best_feasible_e_loss"]
