"""
Objective function for demand calibration.
Compares simulated vs observed edge speeds.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RELIABILITY_PENALTY_SCALE = 2000.0


def compute_fail_total(trip_stats: Optional[Dict[str, float]] = None) -> int:
    """Return routing failures + teleports from SUMO trip stats."""
    if not trip_stats:
        return 0
    routing_failures = int(trip_stats.get("routing_failures", 0) or 0)
    teleports = int(trip_stats.get("teleports", 0) or 0)
    return routing_failures + teleports


def calculate_reliability_penalty(
    fail_total: int,
    expected_vehicles: int,
    penalty_scale: float = RELIABILITY_PENALTY_SCALE,
) -> float:
    """Compute reliability penalty from failure rate."""
    if fail_total <= 0 or expected_vehicles <= 0:
        return 0.0
    failure_rate = fail_total / expected_vehicles
    return failure_rate * penalty_scale


class EdgeSpeedObjective:
    """Objective function based on edge speed matching."""

    def __init__(
        self,
        observed_edges: pd.DataFrame,
        weight_by_confidence: bool = True,
    ):
        """
        Initialize objective function.

        Args:
            observed_edges: DataFrame with columns:
                - edge_id
                - current_speed (observed)
                - freeflow_speed
                - match_confidence
            weight_by_confidence: Weight edges by match confidence
        """
        self.observed_edges = observed_edges.set_index("edge_id")
        self.weight_by_confidence = weight_by_confidence

        logger.info(f"Objective initialized with {len(observed_edges)} observed edges")

    def _calculate_edge_errors(self, simulated_speeds: Dict[str, float]) -> Tuple[List[float], int]:
        """Return per-edge speed errors and count of missing observed edges."""
        errors = []
        missing_count = 0

        for edge_id, obs_row in self.observed_edges.iterrows():
            obs_speed = obs_row["current_speed"]

            if edge_id in simulated_speeds:
                sim_speed = simulated_speeds[edge_id]
                error = sim_speed - obs_speed
            else:
                # Missing edge = no simulated traffic, assume free-flow speed.
                freeflow = obs_row.get("freeflow_speed", 50.0)
                sim_speed = freeflow
                error = sim_speed - obs_speed
                missing_count += 1

            errors.append(error)

        return errors, missing_count

    def calculate_loss_components(
        self,
        simulated_speeds: Dict[str, float],
        trip_stats: Optional[Dict[str, float]] = None,
        expected_vehicles: int = 0,
    ) -> Dict[str, float]:
        """
        Calculate objective components.

        Returns:
            Dict with keys: mae, coverage_penalty, e_loss, fail_total,
            reliability_penalty, loss, missing_edges.
        """
        errors, missing_count = self._calculate_edge_errors(simulated_speeds)

        if not errors:
            return {
                "mae": float("inf"),
                "coverage_penalty": 0.0,
                "e_loss": float("inf"),
                "fail_total": compute_fail_total(trip_stats),
                "reliability_penalty": 0.0,
                "loss": float("inf"),
                "missing_edges": missing_count,
            }

        mae = float(np.mean(np.abs(errors)))

        coverage_penalty = 0.0
        if missing_count > 0:
            coverage_penalty = (missing_count / len(self.observed_edges)) * 10.0

        e_loss = mae + coverage_penalty
        fail_total = compute_fail_total(trip_stats)
        reliability_penalty = calculate_reliability_penalty(fail_total, expected_vehicles)

        if reliability_penalty > 0.0:
            routing_failures = int((trip_stats or {}).get("routing_failures", 0) or 0)
            teleports = int((trip_stats or {}).get("teleports", 0) or 0)
            failure_rate = fail_total / expected_vehicles
            logger.debug(
                "Failure penalty: %s backlog + %s teleports = %s/%s (%.1f%%) = +%.2f km/h",
                routing_failures,
                teleports,
                fail_total,
                expected_vehicles,
                failure_rate * 100.0,
                reliability_penalty,
            )

        return {
            "mae": mae,
            "coverage_penalty": float(coverage_penalty),
            "e_loss": float(e_loss),
            "fail_total": int(fail_total),
            "reliability_penalty": float(reliability_penalty),
            "loss": float(e_loss + reliability_penalty),
            "missing_edges": int(missing_count),
        }

    def calculate_loss(
        self,
        simulated_speeds: Dict[str, float],
        trip_stats: Optional[Dict[str, float]] = None,
        expected_vehicles: int = 0,
    ) -> float:
        """
        Calculate loss (Weighted MAE + Penalty).

        Args:
            simulated_speeds: Dict mapping edge_id -> mean speed (km/h)
            trip_stats: Optional dict with routing failures (from valid trips.xml)
            expected_vehicles: Total vehicles that SHOULD have run

        Returns:
            Float loss value (lower is better)
        """
        components = self.calculate_loss_components(
            simulated_speeds,
            trip_stats=trip_stats,
            expected_vehicles=expected_vehicles,
        )
        return components["loss"]

    def calculate_metrics(
        self,
        simulated_speeds: Dict[str, float],
    ) -> Dict:
        """
        Calculate detailed metrics for analysis.

        Returns:
            Dict with metrics: mae, mse, matched_edges, missing_edges
        """
        errors = []
        matched = 0
        missing = 0

        for edge_id, obs_row in self.observed_edges.iterrows():
            obs_speed = obs_row["current_speed"]

            if edge_id in simulated_speeds:
                sim_speed = simulated_speeds[edge_id]
                error = sim_speed - obs_speed
                matched += 1
            else:
                missing += 1
                # Use freeflow for error calculation in metrics too
                freeflow = obs_row.get("freeflow_speed", 50.0)
                error = freeflow - obs_speed

            errors.append(error)

        if errors:
            mae = np.mean(np.abs(errors))
            mse = np.mean(np.square(errors))
        else:
            mae = float("inf")
            mse = float("inf")

        return {
            "mae": mae,
            "mse": mse,
            "matched_edges": matched,
            "missing_edges": missing,
            "zero_flow_edges": missing,  # Alias for clarity
            "total_edges": len(self.observed_edges),
            "avg_speed_diff": np.mean(errors) if errors else 0.0,
        }
