"""
Cache key generation for content-addressed caching.
"""
import hashlib
import json
from typing import Any, Dict


def generate_cache_key(data: Dict[str, Any]) -> str:
    """
    Generate a deterministic cache key from data.
    
    Args:
        data: Dictionary of parameters
    
    Returns:
        Hex string hash
    """
    # Sort keys for deterministic serialization
    serialized = json.dumps(data, sort_keys=True, separators=(',', ':'))
    
    # SHA256 hash
    hash_obj = hashlib.sha256(serialized.encode('utf-8'))
    
    return hash_obj.hexdigest()


def bbox_key(west: float, south: float, east: float, north: float) -> str:
    """Generate cache key for a bounding box."""
    return generate_cache_key({
        'type': 'bbox',
        'west': round(west, 6),
        'south': round(south, 6),
        'east': round(east, 6),
        'north': round(north, 6)
    })


def osm_key(bbox_key: str) -> str:
    """Generate cache key for OSM data."""
    return generate_cache_key({
        'type': 'osm',
        'bbox_key': bbox_key
    })


def network_key(osm_key: str, car_only: bool, seed: int) -> str:
    """Generate cache key for SUMO network."""
    return generate_cache_key({
        'type': 'network',
        'osm_key': osm_key,
        'car_only': car_only,
        'seed': seed
    })


def traffic_key(
    bbox_key: str,
    provider: str,
    timestamp_bucket: str,
    zoom: int = None,
    style: str = None
) -> str:
    """
    Generate cache key for traffic data.
    
    Args:
        bbox_key: Bbox cache key
        provider: Provider name
        timestamp_bucket: Timestamp rounded bucket (e.g., "2024-01-28T18:05")
        zoom: Optional tile zoom (for tile-based providers)
        style: Optional provider style (absolute/relative etc)
    """
    payload = {
        'type': 'traffic',
        'bbox_key': bbox_key,
        'provider': provider,
        'timestamp_bucket': timestamp_bucket
    }
    if zoom is not None:
        payload['zoom'] = zoom
    if style is not None:
        payload['style'] = style
    return generate_cache_key(payload)


def matching_key(
    bbox_key: str,
    network_cache_key: str,
    provider: str,
    timestamp_bucket: str
) -> str:
    """Generate cache key for map-matching results."""
    return generate_cache_key({
        'type': 'matching',
        'bbox_key': bbox_key,
        'network_key': network_cache_key,
        'provider': provider,
        'timestamp_bucket': timestamp_bucket
    })
