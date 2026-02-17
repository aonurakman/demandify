"""
Data quality assessment helpers for calibration preparation outputs.
"""

from typing import Any, Dict, Optional, Tuple

import pandas as pd

from demandify.utils.validation import calculate_bbox_area_km2


def _score_matched_edges(matched_edges: int) -> int:
    if matched_edges >= 40:
        return 35
    if matched_edges >= 25:
        return 30
    if matched_edges >= 15:
        return 24
    if matched_edges >= 8:
        return 16
    if matched_edges >= 5:
        return 10
    if matched_edges >= 1:
        return 4
    return 0


def _score_match_rate(match_rate: float) -> int:
    if match_rate >= 0.95:
        return 25
    if match_rate >= 0.9:
        return 22
    if match_rate >= 0.8:
        return 18
    if match_rate >= 0.7:
        return 14
    if match_rate >= 0.5:
        return 8
    if match_rate > 0.0:
        return 3
    return 0


def _score_confidence(median_confidence: Optional[float], low_conf_share: Optional[float]) -> int:
    # Neutral score when confidence is unavailable.
    if median_confidence is None or low_conf_share is None:
        return 10

    if median_confidence >= 0.99 and low_conf_share <= 0.02:
        return 15
    if median_confidence >= 0.97 and low_conf_share <= 0.1:
        return 13
    if median_confidence >= 0.94 and low_conf_share <= 0.2:
        return 10
    if median_confidence >= 0.9:
        return 6
    return 2


def _score_density(edges_per_km2: Optional[float]) -> int:
    # Neutral score when bbox area is unknown.
    if edges_per_km2 is None:
        return 8

    if edges_per_km2 >= 10.0:
        return 15
    if edges_per_km2 >= 6.0:
        return 12
    if edges_per_km2 >= 3.0:
        return 8
    if edges_per_km2 >= 1.5:
        return 5
    if edges_per_km2 > 0.0:
        return 2
    return 0


def _score_network_coverage(observed_to_total_ratio: Optional[float]) -> int:
    # Neutral score when network edge count is unknown.
    if observed_to_total_ratio is None:
        return 5

    if observed_to_total_ratio >= 0.03:
        return 10
    if observed_to_total_ratio >= 0.02:
        return 8
    if observed_to_total_ratio >= 0.01:
        return 6
    if observed_to_total_ratio >= 0.005:
        return 4
    if observed_to_total_ratio > 0.0:
        return 2
    return 0


def _label_from_score(score: int) -> str:
    if score >= 85:
        return "excellent"
    if score >= 70:
        return "good"
    if score >= 55:
        return "fair"
    if score >= 40:
        return "weak"
    return "poor"


def _recommendation(score: int, matched_edges: int) -> str:
    if matched_edges == 0:
        return "do_not_proceed"
    if matched_edges < 5 or score < 40:
        return "high_risk"
    if score < 55:
        return "caution"
    return "proceed"


def assess_data_quality(
    traffic_df: pd.DataFrame,
    observed_edges: pd.DataFrame,
    total_network_edges: int,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Dict[str, Any]:
    """
    Assess preparation-stage data quality and produce a label for calibration feasibility.

    Args:
        traffic_df: Raw traffic segments dataframe.
        observed_edges: Matched observed edges dataframe.
        total_network_edges: Total edge count in SUMO network.
        bbox: Optional (west, south, east, north) for density normalization.

    Returns:
        Dictionary with score/label/recommendation, metrics, and human-readable summary.
    """
    fetched_segments = int(len(traffic_df)) if traffic_df is not None else 0
    matched_edges = int(len(observed_edges)) if observed_edges is not None else 0
    total_network_edges = int(total_network_edges or 0)

    match_rate = (
        float(matched_edges / fetched_segments) if fetched_segments > 0 else 0.0
    )
    observed_to_total_ratio = (
        float(matched_edges / total_network_edges) if total_network_edges > 0 else None
    )

    area_km2: Optional[float] = None
    edges_per_km2: Optional[float] = None
    if bbox is not None:
        area_km2 = float(calculate_bbox_area_km2(*bbox))
        if area_km2 > 0:
            edges_per_km2 = float(matched_edges / area_km2)

    median_confidence: Optional[float] = None
    low_conf_share: Optional[float] = None
    if observed_edges is not None and "match_confidence" in observed_edges.columns:
        conf = pd.to_numeric(observed_edges["match_confidence"], errors="coerce").dropna()
        if not conf.empty:
            median_confidence = float(conf.median())
            low_conf_share = float((conf < 0.85).mean())

    score_components = {
        "availability": _score_matched_edges(matched_edges),
        "match_rate": _score_match_rate(match_rate),
        "confidence": _score_confidence(median_confidence, low_conf_share),
        "density": _score_density(edges_per_km2),
        "network_coverage": _score_network_coverage(observed_to_total_ratio),
    }
    score = int(sum(score_components.values()))
    label = _label_from_score(score)
    recommendation = _recommendation(score, matched_edges)

    warnings = []
    if fetched_segments == 0:
        warnings.append("no_traffic_segments")
    if matched_edges == 0:
        warnings.append("no_matched_edges")
    elif matched_edges < 5:
        warnings.append("very_few_matched_edges")

    if match_rate < 0.7:
        warnings.append("low_match_rate")
    if edges_per_km2 is not None and edges_per_km2 < 1.5:
        warnings.append("sparse_observed_edges")
    if observed_to_total_ratio is not None and observed_to_total_ratio < 0.005:
        warnings.append("low_network_coverage")
    if median_confidence is not None and median_confidence < 0.94:
        warnings.append("low_match_confidence")

    summary = (
        f"{label.upper()} quality ({score}/100): "
        f"{matched_edges} matched edges from {fetched_segments} traffic segments "
        f"(match rate {match_rate * 100:.1f}%)."
    )

    return {
        "score": score,
        "label": label,
        "recommendation": recommendation,
        "summary": summary,
        "warnings": warnings,
        "components": score_components,
        "metrics": {
            "fetched_segments": fetched_segments,
            "matched_edges": matched_edges,
            "total_network_edges": total_network_edges,
            "match_rate": match_rate,
            "observed_to_total_ratio": observed_to_total_ratio,
            "area_km2": area_km2,
            "edges_per_km2": edges_per_km2,
            "median_match_confidence": median_confidence,
            "low_confidence_share": low_conf_share,
        },
    }
