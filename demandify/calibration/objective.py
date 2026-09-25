"""Objective function for demand calibration.

Loss is expressed as a dimensionless speed-ratio MAE — the mean absolute
difference between simulated and observed speed, each normalised by the
SUMO edge free-flow speed:

    error(edge, interval) = (sim_speed - obs_speed) / sumo_freeflow_speed

This makes the objective scale-independent and decouples it from the
systematic bias introduced by calibrating a car-only network against
real-world mixed-traffic observations.  A ratio error of 0.1 means the
simulation is off by 10 % of that edge's free-flow speed, regardless of
whether the edge is a residential street at 30 km/h or a motorway at
120 km/h.
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
    """Objective function based on observed-edge speed matching.

    All errors are expressed as a fraction of the SUMO free-flow speed
    (dimensionless speed ratio). MAE values therefore live in [0, ∞)
    where 0 is a perfect match and 1 means the simulation is wrong by
    one full free-flow speed unit on average.
    """

    def __init__(self, observed_edges: pd.DataFrame):
        """
        Initialise objective function.

        Args:
            observed_edges: DataFrame with columns:
                - edge_id
                - current_speed  (observed, km/h)
                - sumo_freeflow_speed_kmh  (SUMO network speed limit, km/h)
                - match_confidence
        """
        self.observed_edges = observed_edges.set_index("edge_id")

    @staticmethod
    def _sumo_freeflow_kmh(obs_row: pd.Series) -> float:
        """Return a finite, positive SUMO free-flow speed in km/h.

        Guaranteed to be ≥ 1.0 so it is always safe to use as a divisor.
        """
        value = obs_row.get("sumo_freeflow_speed_kmh", 50.0)
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            return 50.0
        if not np.isfinite(value_f) or value_f < 1.0:
            return 50.0
        return value_f

    @staticmethod
    def _measurement_interval_count(simulated_speeds: Dict[str, float]) -> int:
        """Return the number of parsed measurement intervals, if available."""
        value = getattr(simulated_speeds, "measurement_intervals", 0)
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _interval_speed_lookup(simulated_speeds: Dict[str, float]) -> Dict[str, Dict[int, float]]:
        """Return sparse per-edge interval speeds, if available."""
        value = getattr(simulated_speeds, "interval_speeds", None)
        return value if isinstance(value, dict) else {}

    def _calculate_edge_errors(self, simulated_speeds: Dict[str, float]) -> Tuple[List[float], int]:
        """Compute per-(edge, interval) normalised speed errors and the missing-edge count.

        Each error is expressed as a speed ratio relative to the SUMO edge
        free-flow speed, making the loss dimensionless and road-class agnostic:

            error = (sim_speed - obs_speed) / sumo_freeflow_speed

        When a simulated interval speed is unavailable, the SUMO free-flow
        speed is used as the "no traffic → uncongested" fallback, consistent
        with the assumption that an unobserved road is flowing freely.

        Two evaluation paths:
          • Interval-aware (primary): per-edge × per-post-warmup-interval errors
          • Edge-mean fallback (legacy): used when interval traces are absent
        """
        errors: List[float] = []
        missing_count = 0
        measurement_intervals = self._measurement_interval_count(simulated_speeds)
        interval_speeds = self._interval_speed_lookup(simulated_speeds)

        if measurement_intervals > 0:
            # --- Interval-aware path (primary) ---
            for edge_id, obs_row in self.observed_edges.iterrows():
                obs_speed = obs_row["current_speed"]
                freeflow = self._sumo_freeflow_kmh(obs_row)
                edge_interval_speeds = interval_speeds.get(edge_id, {})

                if edge_interval_speeds:
                    for interval_idx in range(measurement_intervals):
                        sim_speed = edge_interval_speeds.get(interval_idx, freeflow)
                        errors.append((sim_speed - obs_speed) / freeflow)
                else:
                    # No simulated traffic on this edge; assume free-flow for all intervals.
                    errors.extend([(freeflow - obs_speed) / freeflow] * measurement_intervals)
                    missing_count += 1

            return errors, missing_count

        # --- Edge-mean fallback (legacy) ---
        for edge_id, obs_row in self.observed_edges.iterrows():
            obs_speed = obs_row["current_speed"]
            freeflow = self._sumo_freeflow_kmh(obs_row)

            if edge_id in simulated_speeds:
                sim_speed = simulated_speeds[edge_id]
            else:
                sim_speed = freeflow
                missing_count += 1

            errors.append((sim_speed - obs_speed) / freeflow)

        return errors, missing_count

    def calculate_loss_components(
        self,
        simulated_speeds: Dict[str, float],
        trip_stats: Optional[Dict[str, float]] = None,
        expected_vehicles: int = 0,
    ) -> Dict[str, float]:
        """Calculate objective components.

        Returns:
            Dict with keys: mae, fail_total, failure_rate, loss, missing_edges.
            ``mae`` is a dimensionless speed-ratio value (not km/h).
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
        """Calculate loss (dimensionless speed-ratio MAE, lower is better).

        Args:
            simulated_speeds: Dict-like edge speeds, optionally with attached
                per-interval traces from SUMO edgeData output.
            trip_stats: Optional dict with routing failures / teleports.
            expected_vehicles: Total vehicles that should have run.

        Returns:
            Float MAE value (dimensionless, lower is better).
        """
        return self.calculate_loss_components(
            simulated_speeds,
            trip_stats=trip_stats,
            expected_vehicles=expected_vehicles,
        )["mae"]

    def calculate_metrics(
        self,
        simulated_speeds: Dict[str, float],
    ) -> Dict:
        """Calculate detailed per-edge speed metrics for analysis and logging.

        Returns:
            Dict with keys: mae, mse, matched_edges, missing_edges,
            zero_flow_edges, total_edges, avg_speed_diff.
            ``mae`` and ``mse`` are dimensionless speed-ratio values.
        """
        errors: List[float] = []
        edge_discrepancies: Dict[str, float] = {}
        matched = 0
        missing = 0
        measurement_intervals = self._measurement_interval_count(simulated_speeds)
        interval_speeds = self._interval_speed_lookup(simulated_speeds)

        if measurement_intervals > 0:
            for edge_id, obs_row in self.observed_edges.iterrows():
                obs_speed = obs_row["current_speed"]
                freeflow = self._sumo_freeflow_kmh(obs_row)
                edge_interval_speeds = interval_speeds.get(edge_id, {})

                if edge_interval_speeds:
                    matched += 1
                    edge_errs = []
                    for interval_idx in range(measurement_intervals):
                        sim_speed = edge_interval_speeds.get(interval_idx, freeflow)
                        err = (sim_speed - obs_speed) / freeflow
                        errors.append(err)
                        edge_errs.append(err)
                    edge_discrepancies[str(edge_id)] = float(np.mean(edge_errs))
                else:
                    missing += 1
                    err = (freeflow - obs_speed) / freeflow
                    errors.extend([err] * measurement_intervals)
                    edge_discrepancies[str(edge_id)] = float(err)
        else:
            # Legacy edge-mean path
            for edge_id, obs_row in self.observed_edges.iterrows():
                obs_speed = obs_row["current_speed"]
                freeflow = self._sumo_freeflow_kmh(obs_row)

                if edge_id in simulated_speeds:
                    sim_speed = simulated_speeds[edge_id]
                    matched += 1
                else:
                    sim_speed = freeflow
                    missing += 1

                err = (sim_speed - obs_speed) / freeflow
                errors.append(err)
                edge_discrepancies[str(edge_id)] = float(err)

        if errors:
            mae = float(np.mean(np.abs(errors)))
            mse = float(np.mean(np.square(errors)))
            avg_diff = float(np.mean(errors))
        else:
            mae = float("inf")
            mse = float("inf")
            avg_diff = 0.0

        return {
            "mae": mae,
            "mse": mse,
            "matched_edges": matched,
            "missing_edges": missing,
            "zero_flow_edges": missing,
            "total_edges": len(self.observed_edges),
            "avg_speed_diff": avg_diff,
            "edge_discrepancies": edge_discrepancies,
        }
