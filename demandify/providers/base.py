"""
Base provider interface for traffic data.
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
from datetime import datetime
import pandas as pd


class TrafficProvider(ABC):
    """Abstract base class for traffic data providers."""
    
    @abstractmethod
    async def fetch_traffic_snapshot(
        self,
        bbox: Tuple[float, float, float, float],  # (west, south, east, north)
        timestamp: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch traffic data for a given bounding box.
        
        Args:
            bbox: Bounding box (west, south, east, north) in WGS84
            timestamp: Optional timestamp (defaults to NOW)
        
        Returns:
            DataFrame with columns:
                - segment_id: str
                - geometry: LineString or list of (lon, lat) tuples
                - current_speed: float (km/h)
                - freeflow_speed: float (km/h)
                - timestamp: datetime
                - quality: Optional[float] (0-1 confidence)
        """
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the provider name."""
        pass
    
    @abstractmethod
    def get_provider_metadata(self) -> Dict:
        """Return metadata about the provider configuration."""
        pass
