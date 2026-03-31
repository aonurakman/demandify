"""Tests for SUMO edge matching diagnostics behavior."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from demandify.sumo.matching import EdgeMatcher


def _make_matcher(monkeypatch):
    monkeypatch.setattr(EdgeMatcher, "_build_spatial_index", lambda self: None)
    monkeypatch.setattr(EdgeMatcher, "_setup_transformer", lambda self: None)
    matcher = EdgeMatcher(network=SimpleNamespace())
    monkeypatch.setattr(matcher, "match_segment", lambda *_args, **_kwargs: (None, 0.0))
    return matcher


def _sample_traffic_df():
    return pd.DataFrame(
        {
            "segment_id": ["s1"],
            "geometry": [[(20.0, 50.0), (20.001, 50.001)]],
            "current_speed": [30.0],
            "freeflow_speed": [50.0],
            "timestamp": ["2026-02-17 12:00:00"],
        }
    )


def test_match_traffic_data_no_global_home_debug_log_by_default(monkeypatch, tmp_path):
    matcher = _make_matcher(monkeypatch)
    traffic_df = _sample_traffic_df()

    fake_home = tmp_path / "home"
    fake_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    result = matcher.match_traffic_data(traffic_df, min_confidence=0.1)

    assert result.empty
    assert not (fake_home / ".demandify" / "matching_debug.log").exists()


def test_match_traffic_data_debug_log_is_opt_in_and_run_local(monkeypatch, tmp_path):
    matcher = _make_matcher(monkeypatch)
    traffic_df = _sample_traffic_df()
    debug_log_path = tmp_path / "run_debug" / "logs" / "matching_debug.log"

    matcher.match_traffic_data(
        traffic_df,
        min_confidence=0.1,
        debug_log_path=debug_log_path,
    )

    assert debug_log_path.exists()
    content = debug_log_path.read_text(encoding="utf-8")
    assert "MATCHING SESSION STARTED" in content
