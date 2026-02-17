"""Tests for preparation data quality scoring."""

import pandas as pd

from demandify.utils.data_quality import assess_data_quality


def test_assess_data_quality_high_quality_case():
    traffic_df = pd.DataFrame(
        {
            "segment_id": [f"s{i}" for i in range(50)],
            "current_speed": [30.0] * 50,
        }
    )
    observed_edges = pd.DataFrame(
        {
            "edge_id": [f"e{i}" for i in range(45)],
            "segment_id": [f"s{i}" for i in range(45)],
            "match_confidence": [0.99] * 45,
            "current_speed": [30.0] * 45,
            "freeflow_speed": [40.0] * 45,
        }
    )

    quality = assess_data_quality(
        traffic_df=traffic_df,
        observed_edges=observed_edges,
        total_network_edges=1200,
        bbox=(2.2961, 48.8469, 2.3071, 48.8532),
    )

    assert quality["score"] >= 70
    assert quality["label"] in {"good", "excellent"}
    assert quality["recommendation"] == "proceed"
    assert quality["metrics"]["matched_edges"] == 45
    assert quality["metrics"]["fetched_segments"] == 50


def test_assess_data_quality_no_match_is_do_not_proceed():
    traffic_df = pd.DataFrame(
        {
            "segment_id": [f"s{i}" for i in range(8)],
            "current_speed": [25.0] * 8,
        }
    )
    observed_edges = pd.DataFrame(
        columns=["edge_id", "segment_id", "match_confidence", "current_speed", "freeflow_speed"]
    )

    quality = assess_data_quality(
        traffic_df=traffic_df,
        observed_edges=observed_edges,
        total_network_edges=800,
        bbox=(20.0174, 50.0702, 20.0566, 50.0875),
    )

    assert quality["label"] == "poor"
    assert quality["recommendation"] == "do_not_proceed"
    assert "no_matched_edges" in quality["warnings"]
    assert quality["metrics"]["match_rate"] == 0.0
