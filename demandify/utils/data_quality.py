"""
Data quality assessment helpers for calibration preparation outputs.

Evaluates datasets across 5 core transport engineering dimensions:
1. Critical Hierarchy Coverage (25 pts): Motorway, trunk, primary, secondary road coverage.
2. Spatial Dispersion & Anti-Clustering (25 pts): 4x4 grid occupancy and Gini inequality.
3. Congestion Dynamic Signal (20 pts): Speed variance and bottleneck slowdown ratios.
4. Map-Matching Fidelity (15 pts): Match success rate and geometric snap confidence.
5. Sample Volume & Statistical Scale (15 pts): Absolute unique link count for OD calibration.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from demandify.utils.validation import calculate_bbox_area_km2

try:
    from demandify.sumo.network import SUMONetwork
except ImportError:
    SUMONetwork = None

logger = logging.getLogger(__name__)


def _safe_float(val: Any, round_digits: Optional[int] = None) -> Optional[float]:
    """Convert value to float, ensuring NaN, Inf, and invalid types become None (JSON null)."""
    if val is None:
        return None
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, round_digits) if round_digits is not None else f
    except (TypeError, ValueError):
        return None


def _calculate_gini(values: List[float]) -> float:
    """
    Calculate Gini inequality coefficient across a distribution of counts.

    0.0 = perfect equality (uniform distribution).
    1.0 = maximum inequality (all observations concentrated in a single cell).
    """
    arr = np.array(values, dtype=float)
    if len(arr) == 0 or np.sum(arr) == 0:
        return 1.0
    arr = np.sort(arr)
    n = len(arr)
    index = np.arange(1, n + 1)
    return float(np.sum((2 * index - n - 1) * arr) / (n * np.sum(arr)))


def _score_critical_hierarchy(
    critical_coverage_ratio: Optional[float],
    observed_to_total_ratio: Optional[float],
) -> int:
    """
    Pillar 1: Critical Hierarchy Coverage (25 points).
    Evaluates observation coverage across primary carrying corridors.
    """
    if critical_coverage_ratio is not None:
        if critical_coverage_ratio >= 0.25:
            return 25
        if critical_coverage_ratio >= 0.18:
            return 22
        if critical_coverage_ratio >= 0.12:
            return 18
        if critical_coverage_ratio >= 0.08:
            return 14
        if critical_coverage_ratio >= 0.04:
            return 9
        if critical_coverage_ratio > 0.0:
            return 4
        return 0

    # Fallback to general network coverage ratio when hierarchy cannot be extracted
    if observed_to_total_ratio is not None:
        if observed_to_total_ratio >= 0.15:
            return 25
        if observed_to_total_ratio >= 0.10:
            return 21
        if observed_to_total_ratio >= 0.06:
            return 17
        if observed_to_total_ratio >= 0.03:
            return 13
        if observed_to_total_ratio >= 0.01:
            return 8
        if observed_to_total_ratio > 0.0:
            return 4
        return 0

    return 10


def _score_spatial_dispersion(
    occupancy_ratio: Optional[float],
    gini: Optional[float],
    edges_per_km2: Optional[float],
) -> Tuple[int, int, int]:
    """
    Pillar 2: Spatial Dispersion & Anti-Clustering (25 points).
    Grid occupancy (15 pts) + Gini equality (10 pts).
    Penalizes single-corridor clustering while rewarding uniform footprint coverage.
    """
    if occupancy_ratio is not None and gini is not None:
        if occupancy_ratio >= 0.85:
            p_occ = 15
        elif occupancy_ratio >= 0.70:
            p_occ = 12
        elif occupancy_ratio >= 0.50:
            p_occ = 9
        elif occupancy_ratio >= 0.35:
            p_occ = 5
        elif occupancy_ratio > 0.0:
            p_occ = 2
        else:
            p_occ = 0

        if gini <= 0.30:
            p_eq = 10
        elif gini <= 0.40:
            p_eq = 8
        elif gini <= 0.50:
            p_eq = 6
        elif gini <= 0.60:
            p_eq = 3
        else:
            p_eq = 0

        return p_occ, p_eq, p_occ + p_eq

    # Fallback when spatial network grid is unavailable
    if edges_per_km2 is not None:
        if edges_per_km2 >= 10.0:
            p = 18
        elif edges_per_km2 >= 5.0:
            p = 15
        elif edges_per_km2 >= 2.0:
            p = 12
        elif edges_per_km2 >= 1.0:
            p = 8
        elif edges_per_km2 > 0.0:
            p = 4
        else:
            p = 0
        return p // 2, p - (p // 2), p

    return 6, 6, 12


def _score_congestion_dynamics(
    congestion_share: Optional[float],
    speed_cv: Optional[float],
) -> int:
    """
    Pillar 3: Congestion Dynamic Signal (20 points).
    Rewards presence of bottleneck slowdowns (up to 16 pts) and speed variance (up to 4 pts)
    to guarantee informative optimization gradients for calibration.
    """
    if congestion_share is not None:
        if congestion_share >= 0.65:
            p_share = 16
        elif congestion_share >= 0.45:
            p_share = 12
        elif congestion_share >= 0.25:
            p_share = 8
        elif congestion_share >= 0.10:
            p_share = 4
        elif congestion_share > 0.0:
            p_share = 2
        else:
            p_share = 0

        if speed_cv is not None:
            if speed_cv >= 0.35:
                p_cv = 4
            elif speed_cv >= 0.20:
                p_cv = 3
            elif speed_cv >= 0.10:
                p_cv = 2
            elif congestion_share >= 0.80:
                p_cv = 1
            else:
                p_cv = 0
        else:
            p_cv = 3

        return p_share + p_cv

    if speed_cv is not None:
        if speed_cv >= 0.45:
            return 20
        if speed_cv >= 0.35:
            return 16
        if speed_cv >= 0.25:
            return 12
        if speed_cv >= 0.15:
            return 8
        if speed_cv >= 0.05:
            return 4
        if speed_cv > 0.0:
            return 2
        return 0

    return 10


def _score_match_fidelity(
    match_rate: float,
    median_confidence: Optional[float],
    low_conf_share: Optional[float],
) -> Tuple[int, int, int]:
    """
    Pillar 4: Map-Matching Fidelity (15 points).
    Match rate (10 pts) + GPS snap confidence (5 pts).
    """
    if match_rate >= 0.95:
        p_rate = 10
    elif match_rate >= 0.90:
        p_rate = 9
    elif match_rate >= 0.80:
        p_rate = 7
    elif match_rate >= 0.70:
        p_rate = 5
    elif match_rate >= 0.50:
        p_rate = 3
    elif match_rate > 0.0:
        p_rate = 1
    else:
        p_rate = 0

    if median_confidence is not None and low_conf_share is not None:
        if median_confidence >= 0.98 and low_conf_share <= 0.05:
            p_conf = 5
        elif median_confidence >= 0.95 and low_conf_share <= 0.15:
            p_conf = 4
        elif median_confidence >= 0.90:
            p_conf = 3
        elif median_confidence >= 0.80:
            p_conf = 2
        else:
            p_conf = 1
    else:
        p_conf = 3

    return p_rate, p_conf, p_rate + p_conf


def _score_sample_volume(unique_edges: int) -> int:
    """
    Pillar 5: Sample Volume & Statistical Scale (15 points).
    """
    if unique_edges >= 100:
        return 15
    if unique_edges >= 60:
        return 12
    if unique_edges >= 35:
        return 9
    if unique_edges >= 20:
        return 6
    if unique_edges >= 10:
        return 4
    if unique_edges >= 3:
        return 2
    return 0


def _label_from_score(score: int, warnings: Optional[List[str]] = None) -> str:
    """
    Categorizes score into 5 quality tiers.
    Applies a structural safety cap: datasets with severe spatial clustering
    or critically low arterial coverage cannot be labeled 'excellent'.
    """
    if score >= 85:
        if warnings and any(w in warnings for w in ("low_critical_coverage", "spatial_clustering")):
            return "good"
        return "excellent"
    if score >= 70:
        return "good"
    if score >= 55:
        return "fair"
    if score >= 40:
        return "weak"
    return "poor"


def _recommendation(
    score: int,
    matched_edges: int,
    unique_edges: int,
    warnings: Optional[List[str]] = None,
) -> str:
    if matched_edges == 0:
        return "do_not_proceed"
    if unique_edges < 5 or score < 40:
        return "high_risk"
    if score < 70 or (warnings and "spatial_clustering" in warnings):
        return "caution"
    return "proceed"


def assess_data_quality(
    traffic_df: Optional[pd.DataFrame],
    observed_edges: Optional[pd.DataFrame],
    total_network_edges: int = 0,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    network_file: Optional[Union[str, Path]] = None,
    network: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Assess traffic data quality across critical hierarchy, spatial dispersion,
    congestion signal, match fidelity, and sample volume.

    Args:
        traffic_df: Raw fetched traffic segments.
        observed_edges: Map-matched observed edges dataframe.
        total_network_edges: Total edge count in SUMO network.
        bbox: Optional (west, south, east, north) for density calculations.
        network_file: Optional path to SUMO network.net.xml.
        network: Optional pre-loaded SUMONetwork instance.

    Returns:
        Dictionary with score, label, recommendation, warnings, score components,
        and fine-grained metrics.
    """
    fetched_segments = int(len(traffic_df)) if traffic_df is not None else 0
    matched_edges = int(len(observed_edges)) if observed_edges is not None else 0
    total_network_edges = int(total_network_edges or 0)

    observed_edge_ids: set = set()
    if observed_edges is not None and "edge_id" in observed_edges.columns:
        observed_edge_ids = set(observed_edges["edge_id"].dropna().astype(str))
    unique_matched_edges = len(observed_edge_ids)

    match_rate = (
        float(matched_edges / fetched_segments) if fetched_segments > 0 else 0.0
    )

    # ---------------------------------------------------------
    # Parse SUMO Network if available
    # ---------------------------------------------------------
    net_obj = network
    if net_obj is None and network_file is not None and SUMONetwork is not None:
        try:
            npath = Path(network_file)
            if npath.exists():
                net_obj = SUMONetwork(npath)
        except Exception as e:
            logger.debug(f"Failed to load SUMO network for data quality: {e}")

    if total_network_edges == 0 and net_obj is not None and hasattr(net_obj, "edges"):
        total_network_edges = len(net_obj.edges)

    observed_to_total_ratio = (
        float(unique_matched_edges / total_network_edges)
        if total_network_edges > 0
        else None
    )

    # ---------------------------------------------------------
    # Pillar 1: Critical Hierarchy Coverage
    # ---------------------------------------------------------
    critical_total = 0
    critical_matched = 0
    critical_coverage_ratio: Optional[float] = None

    if net_obj is not None and hasattr(net_obj, "edge_attributes"):
        crit_eids = [
            eid
            for eid, attr in net_obj.edge_attributes.items()
            if any(
                t in str(attr.get("type", "")).lower()
                for t in ["motorway", "trunk", "primary", "secondary"]
            )
        ]
        critical_total = len(crit_eids)
        if critical_total > 0:
            critical_matched = len(observed_edge_ids.intersection(crit_eids))
            critical_coverage_ratio = float(critical_matched / critical_total)

    score_critical = _score_critical_hierarchy(
        critical_coverage_ratio, observed_to_total_ratio
    )

    # ---------------------------------------------------------
    # Pillar 2: Spatial Dispersion & Anti-Clustering
    # ---------------------------------------------------------
    area_km2: Optional[float] = None
    edges_per_km2: Optional[float] = None
    if bbox is not None:
        area_km2 = float(calculate_bbox_area_km2(*bbox))
        if area_km2 > 0:
            edges_per_km2 = float(unique_matched_edges / area_km2)

    active_cells_count: Optional[int] = None
    occupied_cells_count: Optional[int] = None
    spatial_occupancy_ratio: Optional[float] = None
    spatial_gini: Optional[float] = None

    if (
        net_obj is not None
        and getattr(net_obj, "conv_boundary", None) is not None
        and hasattr(net_obj, "edge_geometries")
    ):
        try:
            min_x, min_y, max_x, max_y = net_obj.conv_boundary
            w = max_x - min_x
            h = max_y - min_y
            if w > 0 and h > 0:
                net_cell_counts = {i: 0 for i in range(16)}
                edge_to_cell = {}
                for eid in net_obj.edges:
                    geom = net_obj.edge_geometries.get(eid)
                    if geom is not None and not geom.is_empty:
                        c = geom.centroid
                        col = min(3, max(0, int((c.x - min_x) / (w + 1e-9) * 4)))
                        row = min(3, max(0, int((c.y - min_y) / (h + 1e-9) * 4)))
                        cid = row * 4 + col
                        edge_to_cell[eid] = cid
                        net_cell_counts[cid] += 1

                active_cells = [cid for cid, count in net_cell_counts.items() if count > 0]
                active_cells_count = len(active_cells)

                if active_cells_count > 0:
                    obs_cell_counts = {cid: 0 for cid in active_cells}
                    for eid in observed_edge_ids:
                        cid = edge_to_cell.get(eid)
                        if cid in obs_cell_counts:
                            obs_cell_counts[cid] += 1

                    occupied = [cid for cid, count in obs_cell_counts.items() if count > 0]
                    occupied_cells_count = len(occupied)
                    spatial_occupancy_ratio = float(occupied_cells_count / active_cells_count)
                    spatial_gini = _calculate_gini(list(obs_cell_counts.values()))
        except Exception as e:
            logger.debug(f"Failed to calculate spatial grid dispersion: {e}")

    p_occ, p_eq, score_spatial = _score_spatial_dispersion(
        spatial_occupancy_ratio, spatial_gini, edges_per_km2
    )

    # ---------------------------------------------------------
    # Pillar 3: Congestion Dynamic Signal
    # ---------------------------------------------------------
    congested_count: Optional[int] = None
    congestion_share: Optional[float] = None
    speed_cv: Optional[float] = None

    if observed_edges is not None and "current_speed" in observed_edges.columns and len(observed_edges) > 0:
        cur_speeds = pd.to_numeric(observed_edges["current_speed"], errors="coerce").dropna()
        if len(cur_speeds) >= 2:
            mean_s = float(cur_speeds.mean())
            std_s = float(cur_speeds.std())
            if mean_s > 0 and not np.isnan(std_s):
                speed_cv = float(std_s / mean_s)

            # Determine reference freeflow speed
            ref_speeds = None
            if "sumo_freeflow_speed_kmh" in observed_edges.columns:
                sff = pd.to_numeric(observed_edges["sumo_freeflow_speed_kmh"], errors="coerce")
                if (sff > 0).any():
                    ref_speeds = sff
            if ref_speeds is None and "freeflow_speed" in observed_edges.columns:
                ff = pd.to_numeric(observed_edges["freeflow_speed"], errors="coerce")
                if (ff > 0).any():
                    ref_speeds = ff
            if ref_speeds is None and net_obj is not None and hasattr(net_obj, "edge_attributes"):
                net_speeds = [
                    float(net_obj.edge_attributes.get(str(eid), {}).get("speed", 13.89) * 3.6)
                    for eid in observed_edges.get("edge_id", [])
                ]
                ref_speeds = pd.Series(net_speeds, index=observed_edges.index)

            if ref_speeds is not None:
                valid_mask = (ref_speeds > 0) & (cur_speeds.notna())
                if valid_mask.sum() > 0:
                    ratios = cur_speeds[valid_mask] / ref_speeds[valid_mask]
                    congested_mask = ratios < 0.85
                    congested_count = int(congested_mask.sum())
                    congestion_share = float(congested_count / valid_mask.sum())

    score_congestion = _score_congestion_dynamics(congestion_share, speed_cv)

    # ---------------------------------------------------------
    # Pillar 4: Map-Matching Fidelity
    # ---------------------------------------------------------
    median_confidence: Optional[float] = None
    low_conf_share: Optional[float] = None
    if observed_edges is not None and "match_confidence" in observed_edges.columns:
        conf = pd.to_numeric(observed_edges["match_confidence"], errors="coerce").dropna()
        if not conf.empty:
            median_confidence = float(conf.median())
            low_conf_share = float((conf < 0.85).mean())

    p_rate, p_conf, score_fidelity = _score_match_fidelity(
        match_rate, median_confidence, low_conf_share
    )

    # ---------------------------------------------------------
    # Pillar 5: Sample Volume & Scale
    # ---------------------------------------------------------
    score_volume = _score_sample_volume(unique_matched_edges)

    # ---------------------------------------------------------
    # Aggregate Score & Recommendation
    # ---------------------------------------------------------
    score_components = {
        "critical_hierarchy": score_critical,
        "spatial_dispersion": score_spatial,
        "congestion_dynamics": score_congestion,
        "match_fidelity": score_fidelity,
        "sample_volume": score_volume,
    }
    score = int(sum(score_components.values()))

    warnings: List[str] = []
    if fetched_segments == 0:
        warnings.append("no_traffic_segments")
    if matched_edges == 0:
        warnings.append("no_matched_edges")
    elif unique_matched_edges < 5:
        warnings.append("very_few_matched_edges")

    if match_rate < 0.70:
        warnings.append("low_match_rate")
    if critical_coverage_ratio is not None and critical_coverage_ratio < 0.08:
        warnings.append("low_critical_coverage")
    if spatial_gini is not None and spatial_gini > 0.55:
        warnings.append("spatial_clustering")
    if spatial_occupancy_ratio is not None and spatial_occupancy_ratio < 0.50:
        warnings.append("low_spatial_coverage")
    if congestion_share is not None and congestion_share < 0.03 and (speed_cv is None or speed_cv < 0.15):
        warnings.append("lacks_congestion_dynamics")
    if median_confidence is not None and median_confidence < 0.94:
        warnings.append("low_match_confidence")
    if edges_per_km2 is not None and edges_per_km2 < 1.5:
        warnings.append("sparse_observed_edges")

    label = _label_from_score(score, warnings)
    recommendation = _recommendation(score, matched_edges, unique_matched_edges, warnings)

    crit_desc = (
        f"{critical_coverage_ratio * 100:.1f}% critical road coverage"
        if critical_coverage_ratio is not None
        else f"{observed_to_total_ratio * 100:.1f}% total coverage"
        if observed_to_total_ratio is not None
        else "unspecified coverage"
    )
    spatial_desc = (
        f"{spatial_occupancy_ratio * 100:.0f}% grid occupancy (Gini {spatial_gini:.2f})"
        if spatial_occupancy_ratio is not None and spatial_gini is not None
        else "neutral spatial dispersion"
    )

    summary = (
        f"{label.upper()} quality ({score}/100): "
        f"{matched_edges} traffic observations ({unique_matched_edges} unique edges) matched "
        f"from {fetched_segments} raw segments ({match_rate * 100:.1f}% match rate). "
        f"{crit_desc.capitalize()}, {spatial_desc}."
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
            "unique_matched_edges": unique_matched_edges,
            "total_network_edges": total_network_edges,
            "critical_network_edges": critical_total,
            "critical_matched_edges": critical_matched,
            "critical_coverage_ratio": _safe_float(critical_coverage_ratio, 4),
            "spatial_active_cells": active_cells_count,
            "spatial_occupied_cells": occupied_cells_count,
            "spatial_occupancy_ratio": _safe_float(spatial_occupancy_ratio, 4),
            "spatial_gini": _safe_float(spatial_gini, 3),
            "congested_edges_count": congested_count,
            "congestion_share": _safe_float(congestion_share, 3),
            "speed_cv": _safe_float(speed_cv, 3),
            "match_rate": _safe_float(match_rate, 4),
            "observed_to_total_ratio": _safe_float(observed_to_total_ratio, 4),
            "area_km2": _safe_float(area_km2, 2),
            "edges_per_km2": _safe_float(edges_per_km2, 2),
            "median_match_confidence": _safe_float(median_confidence, 4),
            "low_confidence_share": _safe_float(low_conf_share, 4),
        },
    }
