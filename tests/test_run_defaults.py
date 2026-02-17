"""Tests for shared run defaults used by UI/API/CLI."""

import inspect

from fastapi.params import Form
from starlette.testclient import TestClient

from demandify.app import app
from demandify.config import get_run_defaults
from demandify.web.routes import check_feasibility, start_run


def test_run_defaults_have_required_keys():
    defaults = get_run_defaults()

    required_keys = {
        "window_options_minutes",
        "window_minutes",
        "seed",
        "warmup_minutes",
        "step_length",
        "traffic_tile_zoom",
        "ga_population",
        "ga_generations",
        "parallel_workers",
        "ga_elitism",
        "ga_mutation_rate",
        "ga_crossover_rate",
        "initial_population",
        "num_origins",
        "num_destinations",
        "max_od_pairs",
        "bin_minutes",
        "ga_mutation_sigma",
        "ga_mutation_indpb",
        "ga_immigrant_rate",
        "ga_elite_top_pct",
        "ga_magnitude_penalty_weight",
        "ga_stagnation_patience",
        "ga_stagnation_boost",
        "ga_assortative_mating",
        "ga_deterministic_crowding",
    }
    assert required_keys.issubset(defaults.keys())
    assert defaults["window_minutes"] in defaults["window_options_minutes"]
    assert defaults["parallel_workers"] >= 1


def test_api_form_defaults_match_run_defaults():
    defaults = get_run_defaults()

    start_sig = inspect.signature(start_run)
    check_sig = inspect.signature(check_feasibility)

    def _form_default(sig, param_name):
        param = sig.parameters[param_name]
        assert isinstance(param.default, Form)
        return param.default.default

    assert _form_default(check_sig, "traffic_tile_zoom") == defaults["traffic_tile_zoom"]

    assert _form_default(start_sig, "window_minutes") == defaults["window_minutes"]
    assert _form_default(start_sig, "seed") == defaults["seed"]
    assert _form_default(start_sig, "warmup_minutes") == defaults["warmup_minutes"]
    assert _form_default(start_sig, "step_length") == defaults["step_length"]
    assert _form_default(start_sig, "ga_population") == defaults["ga_population"]
    assert _form_default(start_sig, "ga_generations") == defaults["ga_generations"]
    assert _form_default(start_sig, "parallel_workers") == defaults["parallel_workers"]
    assert _form_default(start_sig, "max_od_pairs") == defaults["max_od_pairs"]
    assert _form_default(start_sig, "bin_minutes") == defaults["bin_minutes"]


def test_index_template_uses_shared_defaults():
    defaults = get_run_defaults()
    client = TestClient(app)

    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text

    assert f'name="seed" value="{defaults["seed"]}"' in html
    assert f'name="ga_population"' in html
    assert f'value="{defaults["ga_population"]}"' in html
    assert f'name="max_od_pairs" value="{defaults["max_od_pairs"]}"' in html
