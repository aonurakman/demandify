"""
Geometry utilities for demandify.
"""
from typing import Tuple
import math


def calculate_bbox_area_km2(west: float, south: float, east: float, north: float) -> float:
    """
    Calculate approximate area of a bbox in km².
    Uses a simple approximation.
    
    Args:
        west, south, east, north: Bounding box coordinates
    
    Returns:
        Area in km²
    """
    # Approximate km per degree at different latitudes
    lat_km_per_deg = 111.0
    
    # Average latitude for longitude scaling
    avg_lat = (south + north) / 2
    lon_km_per_deg = 111.0 * math.cos(math.radians(avg_lat))
    
    width_km = (east - west) * lon_km_per_deg
    height_km = (north - south) * lat_km_per_deg
    
    return width_km * height_km
