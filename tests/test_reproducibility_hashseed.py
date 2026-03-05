"""Regression test: OD selection should be stable across PYTHONHASHSEED values."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _od_pairs_digest_for_hashseed(hash_seed: str) -> str:
    script = r"""
import hashlib
import json
from pathlib import Path

from demandify.sumo.demand import DemandGenerator
from demandify.sumo.network import SUMONetwork

bbox = (20.0174, 50.0702, 20.0566, 50.0875)
w, s, e, n = bbox
dx = (e - w) * 75000.0
dy = (n - s) * 111000.0
min_trip = min(1000.0, max(200.0, ((dx * dx + dy * dy) ** 0.5) * 0.10))

network_file = Path("demandify/offline_datasets/krakow_v1/sumo/network.net.xml")
network = SUMONetwork(network_file)
demand_gen = DemandGenerator(network, seed=42)
od_pairs = demand_gen.select_od_pairs(max_od_pairs=120, min_trip_distance=min_trip)
digest = hashlib.sha256(json.dumps(od_pairs, separators=(",", ":")).encode()).hexdigest()
print(digest)
"""
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hash_seed
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        cwd=Path(__file__).resolve().parent.parent,
    )
    return result.stdout.strip()


def test_od_selection_digest_is_stable_across_python_hashseed():
    digest_seed0 = _od_pairs_digest_for_hashseed("0")
    digest_seed1 = _od_pairs_digest_for_hashseed("1")
    assert digest_seed0
    assert digest_seed0 == digest_seed1
