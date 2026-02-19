import errno
from pathlib import Path

import numpy as np
import pandas as pd

from demandify.calibration import worker


def test_stable_seed_reproducible():
    genome = np.array([1, 2, 3])
    seed_a = worker._stable_seed(genome, base_seed=42)
    seed_b = worker._stable_seed(genome.copy(), base_seed=42)
    assert seed_a == seed_b


def test_stable_seed_changes_with_genome_or_base():
    genome = np.array([1, 2, 3])
    genome2 = np.array([1, 2, 4])

    seed_a = worker._stable_seed(genome, base_seed=42)
    seed_b = worker._stable_seed(genome2, base_seed=42)
    seed_c = worker._stable_seed(genome, base_seed=7)

    assert seed_a != seed_b
    assert seed_a != seed_c


def test_worker_error_metrics_are_explicitly_infeasible():
    metrics = worker.build_worker_error_metrics("boom", worker_idx=7)

    assert metrics["worker_error"] is True
    assert metrics["error"] == "boom"
    assert metrics["fail_total"] >= 1
    assert metrics["routing_failures"] >= 1
    assert metrics["e_loss"] == float("inf")
    assert metrics["loss"] == float("inf")
    assert metrics["worker_id"] == 7


def test_create_worker_temp_dir_prefers_configured_root(tmp_path):
    preferred_root = tmp_path / "temp_eval"

    temp_dir = worker._create_worker_temp_dir(
        preferred_root=preferred_root,
        worker_idx=3,
        run_id="run_test",
    )
    try:
        assert temp_dir.exists()
        assert temp_dir.parent == preferred_root
    finally:
        temp_dir.rmdir()


def test_create_worker_temp_dir_falls_back_when_preferred_root_is_stale(tmp_path, monkeypatch):
    preferred_root = tmp_path / "stale_root"
    fallback_root = tmp_path / "fallback_root"
    original_mkdir = Path.mkdir

    def patched_mkdir(self, *args, **kwargs):
        if self == preferred_root:
            raise OSError(errno.ESTALE, "Stale file handle")
        return original_mkdir(self, *args, **kwargs)

    def patched_mkdtemp(prefix, dir=None):
        if dir is None:
            target = fallback_root / f"{prefix}tmp"
        else:
            target = Path(dir) / f"{prefix}tmp"
        target.mkdir(parents=True, exist_ok=False)
        return str(target)

    monkeypatch.setattr(Path, "mkdir", patched_mkdir)
    monkeypatch.setattr(worker.tempfile, "mkdtemp", patched_mkdtemp)

    worker._UNUSABLE_TEMP_ROOTS.clear()
    temp_dirs = []
    try:
        temp_dir = worker._create_worker_temp_dir(
            preferred_root=preferred_root,
            worker_idx=7,
            run_id="run_test",
        )
        temp_dirs.append(temp_dir)
        assert temp_dir.exists()
        assert temp_dir.parent == fallback_root
        assert str(preferred_root) in worker._UNUSABLE_TEMP_ROOTS
    finally:
        worker._UNUSABLE_TEMP_ROOTS.clear()
        for d in temp_dirs:
            d.rmdir()


def test_stale_preferred_root_is_not_retried_every_evaluation(tmp_path, monkeypatch):
    preferred_root = tmp_path / "stale_root"
    fallback_root = tmp_path / "fallback_root"
    original_mkdir = Path.mkdir
    mkdir_attempts = {"count": 0}
    mkdtemp_counter = {"count": 0}

    def patched_mkdir(self, *args, **kwargs):
        if self == preferred_root:
            mkdir_attempts["count"] += 1
            raise OSError(errno.ESTALE, "Stale file handle")
        return original_mkdir(self, *args, **kwargs)

    def patched_mkdtemp(prefix, dir=None):
        mkdtemp_counter["count"] += 1
        target = fallback_root / f"{prefix}{mkdtemp_counter['count']}"
        target.mkdir(parents=True, exist_ok=False)
        return str(target)

    monkeypatch.setattr(Path, "mkdir", patched_mkdir)
    monkeypatch.setattr(worker.tempfile, "mkdtemp", patched_mkdtemp)

    worker._UNUSABLE_TEMP_ROOTS.clear()
    temp_dirs = []
    try:
        temp_dirs.append(
            worker._create_worker_temp_dir(preferred_root=preferred_root, worker_idx=1, run_id="r")
        )
        temp_dirs.append(
            worker._create_worker_temp_dir(preferred_root=preferred_root, worker_idx=1, run_id="r")
        )
        assert mkdir_attempts["count"] == 1
        assert mkdtemp_counter["count"] == 2
    finally:
        worker._UNUSABLE_TEMP_ROOTS.clear()
        for d in temp_dirs:
            d.rmdir()


def test_run_simulation_worker_returns_infeasible_on_temp_dir_failure(monkeypatch):
    def raise_stale(*args, **kwargs):
        raise OSError(errno.ESTALE, "Stale file handle")

    monkeypatch.setattr(worker, "_create_worker_temp_dir", raise_stale)

    config = worker.SimulationConfig(
        run_id="run_test",
        network_file=Path("network.net.xml"),
        od_pairs=[],
        departure_bins=[],
        observed_edges=pd.DataFrame(),
        warmup_time=0,
        simulation_time=0,
        output_base_dir=Path("temp_eval"),
    )
    loss, metrics = worker.run_simulation_worker(
        genome=np.array([], dtype=int),
        config=config,
        worker_idx=12,
    )

    assert loss == float("inf")
    assert metrics["worker_error"] is True
    assert metrics["worker_id"] == 12
    assert metrics["fail_total"] >= 1
