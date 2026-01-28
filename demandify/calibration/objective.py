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
        simulated_speeds: Dict[str, float]
    ) -> float:
        """
        Calculate loss (Mean Absolute Error) between simulated and observed speeds.
        
        Args:
            simulated_speeds: Dict mapping edge_id -> simulated_speed_kmh
        
        Returns:
            Loss value (lower is better)
        """
        errors = []
        weights = []
        
        for edge_id, obs_row in self.observed_edges.iterrows():
            if edge_id not in simulated_speeds:
                # Edge not in simulation (no traffic)
                # Treat as error
                error = obs_row['current_speed']
                weight = obs_row.get('match_confidence', 1.0) if self.weight_by_confidence else 1.0
            else:
                sim_speed = simulated_speeds[edge_id]
                obs_speed = obs_row['current_speed']
                
                error = abs(sim_speed - obs_speed)
                weight = obs_row.get('match_confidence', 1.0) if self.weight_by_confidence else 1.0
            
            errors.append(error)
            weights.append(weight)
        
        # Weighted mean absolute error
        if sum(weights) > 0:
            weighted_mae = sum(e * w for e, w in zip(errors, weights)) / sum(weights)
        else:
            weighted_mae = float('inf')
        
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
                errors.append(error)
                matched += 1
            else:
                missing += 1
        
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
            'total_edges': len(self.observed_edges)
        }
