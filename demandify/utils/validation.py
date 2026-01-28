"""
Input validation utilities.
"""
import shutil
from typing import Tuple


def check_sumo_availability() -> Tuple[bool, str]:
    """
    Check if SUMO tools are available on the system.
    
    Returns:
        (is_available, message)
    """
    required_tools = ["netconvert", "duarouter", "sumo"]
    missing = []
    
    for tool in required_tools:
        if not shutil.which(tool):
            missing.append(tool)
    
    if missing:
        return False, f"Missing SUMO tools: {', '.join(missing)}. Install SUMO and add to PATH."
    
    return True, "All SUMO tools found"


def validate_bbox(west: float, south: float, east: float, north: float) -> Tuple[bool, str]:
    """
    Validate a bounding box.
    
    Returns:
        (is_valid, message)
    """
    # Basic validation
    if not (-180 <= west <= 180 and -180 <= east <= 180):
        return False, "Longitude must be between -180 and 180"
    
    if not (-90 <= south <= 90 and -90 <= north <= 90):
        return False, "Latitude must be between -90 and 90"
    
    if west >= east:
        return False, "West must be less than East"
    
    if south >= north:
        return False, "South must be less than North"
    
    return True, "Valid bbox"


def calculate_bbox_area_km2(west: float, south: float, east: float, north: float) -> float:
    """
    Calculate approximate area of a bbox in km².
    Uses a simple approximation.
    """
    # Approximate km per degree at different latitudes
    lat_km_per_deg = 111.0
    
    # Average latitude for longitude scaling
    avg_lat = (south + north) / 2
    import math
    lon_km_per_deg = 111.0 * math.cos(math.radians(avg_lat))
    
    width_km = (east - west) * lon_km_per_deg
    height_km = (north - south) * lat_km_per_deg
    
    return width_km * height_km
