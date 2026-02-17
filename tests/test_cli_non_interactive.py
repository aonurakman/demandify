"""Tests for non-interactive headless CLI behavior."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from demandify import cli as cli_module


def _build_run_args(**overrides):
    base = {
        "bbox": "20.0174,50.0702,20.0566,50.0875",
        "name": "test_non_interactive",
        "non_interactive": False,
        "window": 15,
        "warmup": 5,
        "seed": 42,
        "step_length": 1.0,
        "workers": 1,
        "tile_zoom": 12,
        "pop": 10,
        "gen": 2,
        "mutation": 0.5,
        "crossover": 0.7,
        "elitism": 2,
        "sigma": 5,
        "indpb": 0.3,
        "immigrant_rate": 0.03,
        "elite_top_pct": 0.1,
        "magnitude_penalty": 0.01,
        "stagnation_patience": 20,
        "stagnation_boost": 1.2,
        "ga_assortative_mating": True,
        "ga_deterministic_crowding": True,
        "origins": 10,
        "destinations": 10,
        "max_ods": 50,
        "bin_size": 5,
        "initial_population": 1000,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_cmd_run_non_interactive_skips_all_prompts(monkeypatch):
    class FakePipeline:
        confirm_result = None

        def __init__(self, **kwargs):
            self.output_dir = Path("/tmp/demandify_non_interactive")
            self.run_id = kwargs.get("run_id") or "fake_run"

        async def run(self, confirm_callback):
            stats = {
                "fetched_segments": 44,
                "matched_edges": 41,
                "total_network_edges": 1234,
            }
            FakePipeline.confirm_result = confirm_callback(stats)
            return {"run_dir": str(self.output_dir)}

    import demandify.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "CalibrationPipeline", FakePipeline)

    def fail_restart():
        raise AssertionError("_prompt_restart should not be called in non-interactive mode")

    monkeypatch.setattr(cli_module, "_prompt_restart", fail_restart)
    args = _build_run_args(non_interactive=True)

    with patch("builtins.input", side_effect=AssertionError("input() should not be called")):
        asyncio.run(cli_module.cmd_run(args))

    assert FakePipeline.confirm_result is True
