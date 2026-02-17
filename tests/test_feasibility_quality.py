"""Tests for feasibility endpoint quality reporting."""

from types import SimpleNamespace

import pandas as pd
from starlette.testclient import TestClient

from demandify.app import app


def test_check_feasibility_returns_quality(monkeypatch):
    class FakePipeline:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def prepare(self):
            traffic_df = pd.DataFrame(
                {
                    "segment_id": [f"s{i}" for i in range(12)],
                    "current_speed": [25.0] * 12,
                    "freeflow_speed": [40.0] * 12,
                    "timestamp": ["2026-02-17 12:00:00"] * 12,
                }
            )
            observed_edges = pd.DataFrame(
                {
                    "edge_id": [f"e{i}" for i in range(10)],
                    "segment_id": [f"s{i}" for i in range(10)],
                    "current_speed": [24.0] * 10,
                    "freeflow_speed": [40.0] * 10,
                    "match_confidence": [0.99] * 10,
                    "timestamp": ["2026-02-17 12:00:00"] * 10,
                }
            )
            return {
                "traffic_df": traffic_df,
                "observed_edges": observed_edges,
                "total_edges": 500,
            }

    import demandify.config as config_module
    import demandify.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "CalibrationPipeline", FakePipeline)
    monkeypatch.setattr(
        config_module,
        "get_config",
        lambda: SimpleNamespace(tomtom_api_key="abc123", max_bbox_area_km2=25.0),
    )

    client = TestClient(app)
    resp = client.post(
        "/api/check_feasibility",
        data={
            "bbox_west": 2.2961,
            "bbox_south": 48.8469,
            "bbox_east": 2.3071,
            "bbox_north": 48.8532,
            "traffic_tile_zoom": 12,
        },
    )
    assert resp.status_code == 200
    payload = resp.json()

    assert payload["status"] == "success"
    assert payload["stats"]["fetched_segments"] == 12
    assert payload["stats"]["matched_edges"] == 10
    assert "quality" in payload
    assert payload["quality"]["label"] in {"fair", "good", "excellent", "weak", "poor"}
    assert payload["quality"]["metrics"]["matched_edges"] == 10
