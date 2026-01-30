"""
Objective function for demand calibration.
Compares simulated vs observed edge speeds.
"""
from typing import Dict
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class EdgeSpeedObjective:
    """Objective function based on edge speed matching."""
    
    def __init__(
        self,
        observed_edges: pd.DataFrame,
        weight_by_confidence: bool = True
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
        self.observed_edges = observed_edges.set_index('edge_id')
        self.weight_by_confidence = weight_by_confidence
        
        logger.info(f"Objective initialized with {len(observed_edges)} observed edges")
    
    def calculate_loss(
        self,
        simulated_speeds: Dict[str, float],
        trip_stats: Dict[str, float] = None,
        expected_vehicles: int = 0
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
        errors = []
        missing_count = 0
        
        for edge_id, obs_row in self.observed_edges.iterrows():
            obs_speed = obs_row['current_speed']
            
            if edge_id in simulated_speeds:
                sim_speed = simulated_speeds[edge_id]
                error = sim_speed - obs_speed
            else:
                # Missing edge = No traffic = Free Flow condition
                # If the observed data shows congestion (low speed) but we have no traffic,
                # this is a massive error.
                # We assume the empty road would allow free-flow speed.
                freeflow = obs_row.get('freeflow_speed', 50.0) # Default to 50 if missing
                sim_speed = freeflow
                
                # Error is (FreeFlow - Observed). 
                # If Obs=10 and Free=50, Error=40 (Too Fast). Correct direction.
                error = sim_speed - obs_speed
                missing_count += 1
            
            errors.append(error)
        
        if not errors:
            return float('inf')
            
        # 1. Base MAE (Mean Absolute Error)
        mae = np.mean(np.abs(errors))
        
        # 2. Weighted MAE (Penalize "Too Slow" more?)
        # For now, standard MAE is fine.
        weighted_mae = mae
        
        # 3. Routing Failure + Teleport Penalty (CRITICAL)
        # If trips failed to enter the network (gridlock at entry) or teleported (gridlock inside), penalize HEAVILY.
        if trip_stats:
            routing_failures = trip_stats.get('routing_failures', 0)
            teleports = trip_stats.get('teleports', 0) # Teleported vehicles (jammed)
            
            total_failures = routing_failures + teleports
            
            if total_failures > 0 and expected_vehicles and expected_vehicles > 0:
                failure_rate = total_failures / expected_vehicles
                # Exponential penalty: 10% failure -> +200 loss. 50% failure -> +1000 loss
                routing_penalty = failure_rate * 2000.0
                
                logger.debug(f"Failure penalty: {routing_failures} backlog + {teleports} teleports = {total_failures}/{expected_vehicles} ({failure_rate:.1%}) = +{routing_penalty:.2f} km/h")
                weighted_mae += routing_penalty
        
        # 4. Zero Flow Penalty (Soft)
        # We already penalized the error (FreeFlow - Obs), but we can add a small
        # extra penalty per missing edge to encourage coverage coverage.
        if missing_count > 0:
            coverage_penalty = (missing_count / len(self.observed_edges)) * 10.0
            weighted_mae += coverage_penalty

        return weighted_mae
    
    def calculate_metrics(
        self,
        simulated_speeds: Dict[str, float]
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
            obs_speed = obs_row['current_speed']
            
            if edge_id in simulated_speeds:
                sim_speed = simulated_speeds[edge_id]
                error = sim_speed - obs_speed
                matched += 1
            else:
                missing += 1
                # Use freeflow for error calculation in metrics too
                freeflow = obs_row.get('freeflow_speed', 50.0)
                error = freeflow - obs_speed
                
            errors.append(error)
        
        if errors:
            mae = np.mean(np.abs(errors))
            mse = np.mean(np.square(errors))
        else:
            mae = float('inf')
            mse = float('inf')
        
        return {
            'mae': mae,
            'mse': mse,
            'matched_edges': matched,
            'missing_edges': missing,
            'zero_flow_edges': missing, # Alias for clarity
            'total_edges': len(self.observed_edges),
            'avg_speed_diff': np.mean(errors) if errors else 0.0
        }
