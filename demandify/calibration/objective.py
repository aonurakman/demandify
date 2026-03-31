"""
Objective function for demand calibration.
Compares simulated vs observed edge speeds.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def compute_fail_total(trip_stats: Optional[Dict[str, float]] = None) -> int:
    """Return routing failures + teleports from SUMO trip stats."""
    if not trip_stats:
        return 0
    routing_failures = int(trip_stats.get("routing_failures", 0) or 0)
    teleports = int(trip_stats.get("teleports", 0) or 0)
    return routing_failures + teleports

def calculate_failure_rate(fail_total: int, expected_vehicles: int) -> float:
    """Compute failure rate with explicit zero/invalid handling."""
    if fail_total <= 0 and expected_vehicles <= 0:
        return 0.0
    if expected_vehicles <= 0:
        return float("inf")
    return fail_total / expected_vehicles


class EdgeSpeedObjective:
    """Objective function based on edge speed matching."""

    def __init__(self, observed_edges: pd.DataFrame):
        """
        Initialize objective function.

        Args:
            observed_edges: DataFrame with columns:
                - edge_id
                - current_speed (observed)
                - sumo_freeflow_speed_kmh
                - match_confidence
        """
        self.observed_edges = observed_edges.set_index("edge_id")

    @staticmethod
    def _sumo_freeflow_kmh(obs_row: pd.Series) -> float:
        """Return a finite SUMO free-flow fallback speed in km/h."""
        value = obs_row.get("sumo_freeflow_speed_kmh", 50.0)
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            return 50.0
        return value_f if np.isfinite(value_f) else 50.0

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
                # Missing observed edge = no simulated speed signal, use the matched
                # SUMO edge free-flow speed as the uncongested fallback.
                sim_speed = self._sumo_freeflow_kmh(obs_row)
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
            Dict with keys: mae, fail_total, failure_rate, loss, missing_edges.
        """
        errors, missing_count = self._calculate_edge_errors(simulated_speeds)

        if not errors:
            return {
                "mae": float("inf"),
                "fail_total": compute_fail_total(trip_stats),
                "failure_rate": float("inf"),
                "loss": float("inf"),
                "missing_edges": missing_count,
            }

        mae = float(np.mean(np.abs(errors)))
        fail_total = compute_fail_total(trip_stats)
        failure_rate = calculate_failure_rate(fail_total, expected_vehicles)

        return {
            "mae": mae,
            "fail_total": int(fail_total),
            "failure_rate": float(failure_rate),
            "loss": float(mae),
            "missing_edges": int(missing_count),
        }

    def calculate_loss(
        self,
        simulated_speeds: Dict[str, float],
        trip_stats: Optional[Dict[str, float]] = None,
        expected_vehicles: int = 0,
    ) -> float:
        """
        Calculate loss (MAE).

        Args:
            simulated_speeds: Dict mapping edge_id -> mean speed (km/h)
            trip_stats: Optional dict with routing failures (from valid trips.xml)
            expected_vehicles: Total vehicles that SHOULD have run

        Returns:
            Float MAE value (lower is better)
        """
        components = self.calculate_loss_components(
            simulated_speeds,
            trip_stats=trip_stats,
            expected_vehicles=expected_vehicles,
        )
        return components["mae"]

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
                # Use SUMO free-flow fallback in metrics as well.
                sumo_freeflow = self._sumo_freeflow_kmh(obs_row)
                error = sumo_freeflow - obs_speed

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
