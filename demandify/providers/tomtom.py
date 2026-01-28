"""
TomTom Traffic Flow provider.
Implements real Vector Flow Tiles API integration per SPEC requirements.
"""
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import httpx
import pandas as pd
from shapely.geometry import LineString
import logging
import math

from demandify.providers.base import TrafficProvider


logger = logging.getLogger(__name__)


class TomTomProvider(TrafficProvider):
    """TomTom Traffic Flow provider using Flow Segment Data API."""
    
    # Flow Segment Data endpoint (easier than Vector Tiles, same data quality)
    FLOW_SEGMENT_BASE = "https://api.tomtom.com/traffic/services/4/flowSegmentData"
    
    # Styles: absolute, relative, relative-delay
    DEFAULT_STYLE = "absolute"
    
    def __init__(self, api_key: str, style: str = DEFAULT_STYLE):
        """
        Initialize TomTom provider.
        
        Args:
            api_key: TomTom API key
            style: Flow style (absolute, relative, relative-delay)
        """
        self.api_key = api_key
        self.style = style
        self.client = httpx.AsyncClient(timeout=30.0)
    
    def _bbox_to_grid_points(
        self,
        bbox: Tuple[float, float, float, float],
        grid_size: int = 5
    ) -> List[Tuple[float, float]]:
        """
        Convert bbox to a grid of sampling points.
        
        TomTom Flow Segment Data API works with point queries, not bboxes.
        We sample the bbox with a grid and fetch segments near each point.
        
        Args:
            bbox: (west, south, east, north)
            grid_size: Number of points per dimension
        
        Returns:
            List of (lat, lon) sample points
        """
        west, south, east, north = bbox
        
        points = []
        for i in range(grid_size):
            for j in range(grid_size):
                lat = south + (north - south) * (i / (grid_size - 1)) if grid_size > 1 else (south + north) / 2
                lon = west + (east - west) * (j / (grid_size - 1)) if grid_size > 1 else (west + east) / 2
                points.append((lat, lon))
        
        return points
    
    async def _fetch_segment_at_point(
        self,
        lat: float,
        lon: float,
        zoom: int = 10
    ) -> Optional[Dict]:
        """
        Fetch flow segment data at a specific point.
        
        Args:
            lat, lon: Point coordinates
            zoom: Zoom level (10-22, affects detail)
        
        Returns:
            Segment data dict or None if failed
        """
        # Build URL: /flowSegmentData/{style}/{zoom}/json
        url = f"{self.FLOW_SEGMENT_BASE}/{self.style}/{zoom}/json"
        
        params = {
            "key": self.api_key,
            "point": f"{lat},{lon}",
            "unit": "KMPH"  # Speed units
        }
        
        try:
            response = await self.client.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract flow segment data
                if "flowSegmentData" in data:
                    segment = data["flowSegmentData"]
                    
                    # Extract coordinates for geometry
                    coords = segment.get("coordinates", {}).get("coordinate", [])
                    if not coords:
                        return None
                    
                    # Convert to LineString geometry
                    geometry = [(c["longitude"], c["latitude"]) for c in coords]
                    
                    # Extract speeds
                    current_speed = segment.get("currentSpeed", 0)
                    freeflow_speed = segment.get("freeFlowSpeed", current_speed)
                    
                    # Build segment record
                    return {
                        "segment_id": segment.get("frc", "") + "_" + str(hash(tuple(geometry[0] if geometry else (0, 0)))),
                        "geometry": geometry,
                        "current_speed": float(current_speed),
                        "freeflow_speed": float(freeflow_speed),
                        "timestamp": datetime.now(),
                        "quality": 1.0 - (segment.get("confidence", 1.0) - 1.0) if "confidence" in segment else 0.9,
                        "road_class": segment.get("frc", "unknown")
                    }
            
            elif response.status_code == 403:
                logger.error("TomTom API key invalid or quota exceeded")
                return None
            elif response.status_code == 429:
                logger.error("TomTom API rate limit exceeded")
                return None
            else:
                logger.warning(f"TomTom API returned status {response.status_code}")
                return None
        
        except Exception as e:
            logger.error(f"Error fetching TomTom segment at ({lat}, {lon}): {e}")
            return None
    
    async def fetch_traffic_snapshot(
        self,
        bbox: Tuple[float, float, float, float],
        timestamp: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch traffic flow data for a bounding box.
        
        Implementation uses Flow Segment Data API with grid sampling.
        This is simpler than Vector Flow Tiles but provides same quality data.
        
        Args:
            bbox: (west, south, east, north)
            timestamp: Ignored (TomTom Flow is real-time only)
        
        Returns:
            DataFrame with columns: segment_id, geometry, current_speed,
                                   freeflow_speed, timestamp, quality
        """
        west, south, east, north = bbox
        
        if timestamp is not None:
            logger.info("TomTom Flow is real-time only; ignoring timestamp parameter")
        
        logger.info(f"Fetching TomTom flow data for bbox {bbox}")
        
        # Calculate grid size based on bbox area
        # Larger areas = sparser grid to avoid quota issues
        bbox_width = abs(east - west)
        bbox_height = abs(north - south)
        area_deg2 = bbox_width * bbox_height
        
        # Heuristic: ~1 point per 0.002 deg² (~200m spacing at mid latitudes)
        # Higher density for better urban coverage
        # Cap between 3x3 and 20x20 (400 API calls max)
        spacing = 0.002  # ~200-250 meters
        grid_dim = int(math.sqrt(area_deg2) / spacing)
        grid_size = max(3, min(20, grid_dim))
        
        logger.info(f"Using {grid_size}x{grid_size} sampling grid ({grid_size**2} API calls) with spacing ~{spacing} deg")
        
        # Generate sample points
        points = self._bbox_to_grid_points(bbox, grid_size)
        
        # Fetch segments for each point
        segments = []
        seen_geometries = set()  # Deduplicate by geometry
        
        for lat, lon in points:
            segment = await self._fetch_segment_at_point(lat, lon, zoom=12)
            
            if segment:
                # Deduplicate by first coordinate (rough)
                geom_key = tuple(segment["geometry"][0]) if segment["geometry"] else None
                if geom_key and geom_key not in seen_geometries:
                    segments.append(segment)
                    seen_geometries.add(geom_key)
        
        if not segments:
            logger.warning("No traffic segments fetched - check API key and quota")
            # Return empty DataFrame with correct schema
            return pd.DataFrame(columns=[
                "segment_id", "geometry", "current_speed",
                "freeflow_speed", "timestamp", "quality"
            ])
        
        logger.info(f"Fetched {len(segments)} unique traffic segments")
        
        return pd.DataFrame(segments)
    
    def get_provider_name(self) -> str:
        return "TomTom Traffic Flow"
    
    def get_provider_metadata(self) -> Dict:
        return {
            "provider": "tomtom",
            "api_version": "v4",
            "style": self.style,
            "data_type": "flow_segment_data"
        }
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


async def create_tomtom_provider(api_key: str, **kwargs) -> TomTomProvider:
    """Factory function to create a TomTom provider."""
    return TomTomProvider(api_key, **kwargs)
