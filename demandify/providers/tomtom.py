"""
TomTom Traffic Flow provider.
Prefers Vector Flow Tiles; falls back to Flow Segment sampling.
"""
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import httpx
import pandas as pd
from shapely.geometry import LineString
import logging
import math

from demandify.providers.base import TrafficProvider

try:
    from mapbox_vector_tile import decode as mvt_decode
    HAS_MVT = True
except Exception:
    HAS_MVT = False


logger = logging.getLogger(__name__)


def _lonlat_to_tile(lon: float, lat: float, zoom: int) -> Tuple[int, int]:
    """Convert lon/lat to XYZ tile indices."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return xtile, ytile


def _tile_bounds(x: int, y: int, zoom: int) -> Tuple[float, float, float, float]:
    """Return west, south, east, north bounds of a tile."""
    n = 2.0 ** zoom
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return west, south, east, north


def _tile_to_lonlat(x: float, y: float, x_tile: int, y_tile: int, zoom: int, extent: int = 4096) -> Tuple[float, float]:
    """Convert vector-tile local coords to lon/lat."""
    west, south, east, north = _tile_bounds(x_tile, y_tile, zoom)
    lon = west + (x / extent) * (east - west)
    lat = north - (y / extent) * (north - south)
    return lon, lat


class TomTomProvider(TrafficProvider):
    """TomTom Traffic Flow provider preferring Vector Flow Tiles."""
    
    FLOW_SEGMENT_BASE = "https://api.tomtom.com/traffic/services/4/flowSegmentData"
    TILE_BASE = "https://api.tomtom.com/traffic/map/4/tile/flow"
    DEFAULT_STYLE = "absolute"
    
    def __init__(self, api_key: str, style: str = DEFAULT_STYLE, tile_zoom: int = 12):
        """
        Initialize TomTom provider.
        
        Args:
            api_key: TomTom API key
            style: Flow style (absolute, relative, relative-delay)
            tile_zoom: Tile zoom for vector tiles (higher = more detail, more tiles)
        """
        self.api_key = api_key
        self.style = style
        self.tile_zoom = tile_zoom
        self.client = httpx.AsyncClient(timeout=30.0)
        self.use_tiles = HAS_MVT
        if not HAS_MVT:
            logger.warning("mapbox-vector-tile not available; falling back to Flow Segment sampling.")
    
    def _bbox_to_tiles(self, bbox: Tuple[float, float, float, float]) -> List[Tuple[int, int]]:
        """Compute XYZ tiles covering bbox."""
        west, south, east, north = bbox
        x_min, y_max = _lonlat_to_tile(west, south, self.tile_zoom)
        x_max, y_min = _lonlat_to_tile(east, north, self.tile_zoom)
        
        tiles = []
        for x in range(min(x_min, x_max), max(x_min, x_max) + 1):
            for y in range(min(y_min, y_max), max(y_min, y_max) + 1):
                tiles.append((x, y))
        return tiles
    
    async def _fetch_tile(self, x: int, y: int) -> Optional[bytes]:
        """Fetch vector flow tile bytes."""
        url = f"{self.TILE_BASE}/{self.style}/{self.tile_zoom}/{x}/{y}.pbf"
        params = {"key": self.api_key}
        
        try:
            resp = await self.client.get(url, params=params)
            if resp.status_code == 200:
                return resp.content
            if resp.status_code == 403:
                logger.error("TomTom API key invalid or quota exceeded (tiles)")
            elif resp.status_code == 429:
                logger.error("TomTom tile rate limit exceeded")
            else:
                logger.warning(f"TomTom tile fetch returned status {resp.status_code}")
        except Exception as e:
            logger.error(f"Error fetching tile {x}/{y}/{self.tile_zoom}: {e}")
        return None
    
    def _decode_tile(self, tile_bytes: bytes, x: int, y: int) -> List[Dict]:
        """Decode vector tile into flow segments."""
        segments = []
        try:
            decoded = mvt_decode(tile_bytes)
        except Exception as e:
            logger.error(f"Failed to decode vector tile {x}/{y}/{self.tile_zoom}: {e}")
            return segments
        
        for layer_name, layer in decoded.items():
            features = layer.get("features", [])
            extent = layer.get("extent", 4096)
            for feat in features:
                props = feat.get("properties", {})
                geom = feat.get("geometry")
                if not geom or geom["type"] != "LineString":
                    continue
                coords = []
                for px, py in geom["coordinates"]:
                    lon, lat = _tile_to_lonlat(px, py, x, y, self.tile_zoom, extent=extent)
                    coords.append((lon, lat))
                if not coords:
                    continue
                
                current_speed = props.get("currentSpeed") or props.get("current_speed")
                freeflow = props.get("freeFlowSpeed") or props.get("freeflowSpeed") or current_speed
                confidence = props.get("confidence", 0.9)
                segment_id = props.get("id") or props.get("segmentId") or f"{layer_name}_{hash(tuple(coords[0]))}"
                segments.append({
                    "segment_id": str(segment_id),
                    "geometry": coords,
                    "current_speed": float(current_speed) if current_speed is not None else 0.0,
                    "freeflow_speed": float(freeflow) if freeflow is not None else 0.0,
                    "timestamp": datetime.utcnow(),
                    "quality": float(confidence) if confidence is not None else 0.9
                })
        return segments
    
    def _bbox_to_grid_points(
        self,
        bbox: Tuple[float, float, float, float],
        grid_size: int = 5
    ) -> List[Tuple[float, float]]:
        """Fallback: convert bbox to sampling grid for Flow Segment API."""
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
        zoom: int = 12
    ) -> Optional[Dict]:
        """Fallback: Flow Segment Data at a point."""
        url = f"{self.FLOW_SEGMENT_BASE}/{self.style}/{zoom}/json"
        params = {
            "key": self.api_key,
            "point": f"{lat},{lon}",
            "unit": "KMPH"
        }
        
        try:
            response = await self.client.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if "flowSegmentData" in data:
                    segment = data["flowSegmentData"]
                    coords = segment.get("coordinates", {}).get("coordinate", [])
                    if not coords:
                        return None
                    geometry = [(c["longitude"], c["latitude"]) for c in coords]
                    current_speed = segment.get("currentSpeed", 0)
                    freeflow_speed = segment.get("freeFlowSpeed", current_speed)
                    return {
                        "segment_id": segment.get("frc", "") + "_" + str(hash(tuple(geometry[0] if geometry else (0, 0)))),
                        "geometry": geometry,
                        "current_speed": float(current_speed),
                        "freeflow_speed": float(freeflow_speed),
                        "timestamp": datetime.utcnow(),
                        "quality": float(segment.get("confidence", 0.9)),
                        "road_class": segment.get("frc", "unknown")
                    }
            elif response.status_code == 403:
                logger.error("TomTom API key invalid or quota exceeded (flow segment)")
            elif response.status_code == 429:
                logger.error("TomTom API rate limit exceeded (flow segment)")
            else:
                logger.warning(f"TomTom API returned status {response.status_code}")
        except Exception as e:
            logger.error(f"Error fetching TomTom segment at ({lat}, {lon}): {e}")
        return None
    
    async def _fetch_via_tiles(self, bbox: Tuple[float, float, float, float]) -> pd.DataFrame:
        """Fetch via vector tiles."""
        if not self.use_tiles:
            return pd.DataFrame()
        
        tiles = self._bbox_to_tiles(bbox)
        segments = []
        seen = set()
        
        for x, y in tiles:
            tile_bytes = await self._fetch_tile(x, y)
            if not tile_bytes:
                continue
            decoded = self._decode_tile(tile_bytes, x, y)
            for seg in decoded:
                geom_key = tuple(seg["geometry"][0]) if seg.get("geometry") else None
                if geom_key and geom_key in seen:
                    continue
                seen.add(geom_key)
                segments.append(seg)
        
        if segments:
            logger.info(f"Fetched {len(segments)} segments from {len(tiles)} tiles (zoom={self.tile_zoom})")
            return pd.DataFrame(segments)
        return pd.DataFrame()
    
    async def _fetch_via_flow_segments(self, bbox: Tuple[float, float, float, float]) -> pd.DataFrame:
        """Fallback grid sampling."""
        west, south, east, north = bbox
        bbox_width = abs(east - west)
        bbox_height = abs(north - south)
        area_deg2 = bbox_width * bbox_height
        spacing = 0.003  # slightly coarser to reduce calls
        grid_dim = int(math.sqrt(area_deg2) / spacing)
        grid_size = max(3, min(12, grid_dim))
        
        logger.info(f"Using fallback Flow Segment sampling grid {grid_size}x{grid_size} ({grid_size**2} calls)")
        points = self._bbox_to_grid_points(bbox, grid_size)
        
        segments = []
        seen_geometries = set()
        for lat, lon in points:
            segment = await self._fetch_segment_at_point(lat, lon, zoom=12)
            if segment:
                geom_key = tuple(segment["geometry"][0]) if segment["geometry"] else None
                if geom_key and geom_key not in seen_geometries:
                    segments.append(segment)
                    seen_geometries.add(geom_key)
        return pd.DataFrame(segments)
    
    async def fetch_traffic_snapshot(
        self,
        bbox: Tuple[float, float, float, float],
        timestamp: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch traffic flow data for a bounding box.
        Prefers vector flow tiles; falls back to Flow Segment grid sampling.
        """
        if timestamp is not None:
            logger.info("TomTom Flow is real-time; ignoring timestamp override (bucketed upstream).")
        
        logger.info(f"Fetching TomTom flow data for bbox {bbox} (tiles preferred)")
        
        df = await self._fetch_via_tiles(bbox) if self.use_tiles else pd.DataFrame()
        if df.empty:
            df = await self._fetch_via_flow_segments(bbox)
        
        if df.empty:
            logger.warning("No traffic segments fetched - check API key/quota or area coverage.")
            return pd.DataFrame(columns=[
                "segment_id", "geometry", "current_speed",
                "freeflow_speed", "timestamp", "quality"
            ])
        
        return df
    
    def get_provider_name(self) -> str:
        return "TomTom Traffic Flow"
    
    def get_provider_metadata(self) -> Dict:
        return {
            "provider": "tomtom",
            "api_version": "v4",
            "style": self.style,
            "data_type": "vector_flow_tiles" if self.use_tiles else "flow_segment_data",
            "tile_zoom": self.tile_zoom
        }
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


async def create_tomtom_provider(api_key: str, **kwargs) -> TomTomProvider:
    """Factory function to create a TomTom provider."""
    return TomTomProvider(api_key, **kwargs)
