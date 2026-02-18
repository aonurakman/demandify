"""
TomTom Traffic Flow provider.
Prefers Vector Flow Tiles; falls back to Flow Segment sampling.
"""
from typing import Any, Dict, Tuple, Optional, List, Sequence
from datetime import datetime
import hashlib
import os
import httpx
import pandas as pd
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


def _tile_to_lonlat(
    x: float,
    y: float,
    x_tile: int,
    y_tile: int,
    zoom: int,
    extent: int = 4096
) -> Tuple[float, float]:
    """
    Convert vector-tile local coords (0..extent) to lon/lat (WGS84).
    
    Uses Web Mercator math (lat is non-linear in y).
    """
    scale = (2.0 ** zoom) * float(extent)
    global_x = (x_tile * extent) + x
    global_y = (y_tile * extent) + y
    
    lon = (global_x / scale) * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * global_y / scale))))
    return lon, lat


def _geometry_bbox(geometry: Sequence[Tuple[float, float]]) -> Optional[Tuple[float, float, float, float]]:
    """Return (west, south, east, north) bbox for a geometry (lon,lat points)."""
    if not geometry:
        return None
    xs = [p[0] for p in geometry]
    ys = [p[1] for p in geometry]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_intersects(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    """Return True if bbox a intersects bbox b. Bbox = (west,south,east,north)."""
    aw, a_s, ae, an = a
    bw, b_s, be, bn = b
    return not (ae < bw or be < aw or an < b_s or bn < a_s)


def _get_prop_float(props: Dict, keys: Sequence[str]) -> Optional[float]:
    """Get the first present property key as float."""
    for key in keys:
        if key not in props:
            continue
        val = props.get(key)
        if val is None:
            continue
        try:
            return float(val)
        except Exception:
            continue
    return None


class TomTomProvider(TrafficProvider):
    """TomTom Traffic Flow provider preferring Vector Flow Tiles."""
    
    FLOW_SEGMENT_BASE = "https://api.tomtom.com/traffic/services/4/flowSegmentData"
    TILE_BASE = "https://api.tomtom.com/traffic/map/4/tile/flow"
    DEFAULT_STYLE = "absolute"
    # Temporary safeguard: keep tile logic in code, but default to flow-segment sampling.
    # Set DEMANDIFY_ENABLE_TOMTOM_TILES=1 to re-enable tile mode.
    DEFAULT_ENABLE_TILES = False
    CURRENT_SPEED_KEYS = (
        "traffic_level",
        "trafficLevel",
        "trafficlevel",
        "currentSpeed",
        "current_speed",
        "cs",
        "cspd",
        "cSpeed",
        "speed",
        "sp",
        "s",
        "c",
    )
    FREEFLOW_SPEED_KEYS = (
        "freeFlowSpeed",
        "freeflowSpeed",
        "freeflow_speed",
        "ffs",
        "ffSpeed",
        "ff",
        "f",
        "free",
    )
    CONFIDENCE_KEYS = ("confidence", "cn", "conf")
    
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
        env_flag = os.getenv("DEMANDIFY_ENABLE_TOMTOM_TILES")
        if env_flag is None:
            tiles_enabled = self.DEFAULT_ENABLE_TILES
        else:
            tiles_enabled = env_flag.strip().lower() in {"1", "true", "yes", "on"}

        self.use_tiles = HAS_MVT and tiles_enabled
        if not HAS_MVT:
            logger.warning("mapbox-vector-tile not available; falling back to Flow Segment sampling.")
        elif not tiles_enabled:
            logger.info(
                "TomTom vector tile mode temporarily disabled; using Flow Segment sampling. "
                "Set DEMANDIFY_ENABLE_TOMTOM_TILES=1 to re-enable tiles."
            )
    
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

    @staticmethod
    def _empty_tile_diagnostics(tile_count: int) -> Dict[str, Any]:
        return {
            "tiles_total": int(tile_count),
            "tile_fetch_ok": 0,
            "tile_fetch_failed": 0,
            "tile_fetch_fail_reasons": {},
            "tile_decode_failures": 0,
            "layers_total": 0,
            "features_total": 0,
            "feature_drop_reasons": {
                "missing_geometry": 0,
                "unsupported_geometry_type": 0,
                "empty_geometry_lines": 0,
                "short_geometry_line": 0,
                "invalid_geometry_point": 0,
                "missing_speed_property": 0,
            },
            "segments_decoded": 0,
            "segments_dropped_outside_bbox": 0,
            "segments_dropped_duplicate_id": 0,
            "segments_kept": 0,
        }

    @staticmethod
    def _increment_count(counter: Dict[str, int], key: str, value: int = 1) -> None:
        if value <= 0:
            return
        counter[key] = int(counter.get(key, 0)) + int(value)

    @staticmethod
    def _merge_counts(target: Dict[str, int], source: Dict[str, int]) -> None:
        for key, value in source.items():
            TomTomProvider._increment_count(target, key, int(value))

    @staticmethod
    def _format_counts(counts: Dict[str, int]) -> str:
        non_zero = [(key, int(value)) for key, value in counts.items() if int(value) > 0]
        if not non_zero:
            return "none"
        non_zero.sort(key=lambda item: item[0])
        return ", ".join(f"{key}={value}" for key, value in non_zero)
    
    async def _fetch_tile(self, x: int, y: int) -> Tuple[Optional[bytes], str]:
        """Fetch vector flow tile bytes and return (content, status_reason)."""
        url = f"{self.TILE_BASE}/{self.style}/{self.tile_zoom}/{x}/{y}.pbf"
        params = {"key": self.api_key}
        
        try:
            resp = await self.client.get(url, params=params)
            if resp.status_code == 200:
                if resp.content:
                    return resp.content, "ok"
                logger.warning(f"TomTom tile {x}/{y}/{self.tile_zoom} returned HTTP 200 with empty payload")
                return None, "http_200_empty"
            if resp.status_code == 403:
                logger.error("TomTom API key invalid or quota exceeded (tiles)")
                return None, "http_403"
            elif resp.status_code == 429:
                logger.error("TomTom tile rate limit exceeded")
                return None, "http_429"
            else:
                logger.warning(f"TomTom tile fetch returned status {resp.status_code}")
                return None, f"http_{resp.status_code}"
        except Exception as e:
            logger.error(f"Error fetching tile {x}/{y}/{self.tile_zoom}: {e}")
            return None, "exception"
        return None, "unknown"
    
    def _decode_tile(self, tile_bytes: bytes, x: int, y: int) -> Tuple[List[Dict], Dict[str, Any]]:
        """Decode vector tile into flow segments plus decode diagnostics."""
        segments = []
        diagnostics = {
            "decode_error": 0,
            "layers_total": 0,
            "features_total": 0,
            "feature_drop_reasons": {
                "missing_geometry": 0,
                "unsupported_geometry_type": 0,
                "empty_geometry_lines": 0,
                "short_geometry_line": 0,
                "invalid_geometry_point": 0,
                "missing_speed_property": 0,
            },
            "segments_decoded": 0,
        }
        try:
            decoded = mvt_decode(tile_bytes)
        except Exception as e:
            logger.error(f"Failed to decode vector tile {x}/{y}/{self.tile_zoom}: {e}")
            diagnostics["decode_error"] = 1
            return segments, diagnostics
        
        diagnostics["layers_total"] = len(decoded)
        
        for layer_name, layer in decoded.items():
            features = layer.get("features", [])
            diagnostics["features_total"] += len(features)
            extent = layer.get("extent", 4096)
            for feat in features:
                props = feat.get("properties", {})
                geom = feat.get("geometry")
                if not geom:
                    diagnostics["feature_drop_reasons"]["missing_geometry"] += 1
                    continue
                
                geom_type = geom.get("type")
                if geom_type not in ("LineString", "MultiLineString"):
                    diagnostics["feature_drop_reasons"]["unsupported_geometry_type"] += 1
                    continue
                
                # Normalize to a single coordinate sequence (take the longest line if MultiLineString)
                lines = geom.get("coordinates", [])
                if geom_type == "LineString":
                    lines = [lines]
                if not lines:
                    diagnostics["feature_drop_reasons"]["empty_geometry_lines"] += 1
                    continue
                line = max(lines, key=len) if isinstance(lines, list) else lines
                if not line or len(line) < 2:
                    diagnostics["feature_drop_reasons"]["short_geometry_line"] += 1
                    continue
                
                coords: List[Tuple[float, float]] = []
                invalid_point = False
                for point in line:
                    if not isinstance(point, (list, tuple)) or len(point) != 2:
                        invalid_point = True
                        break
                    px, py = point
                    lon, lat = _tile_to_lonlat(px, py, x, y, self.tile_zoom, extent=extent)
                    coords.append((lon, lat))
                if invalid_point:
                    diagnostics["feature_drop_reasons"]["invalid_geometry_point"] += 1
                    continue
                if len(coords) < 2:
                    diagnostics["feature_drop_reasons"]["short_geometry_line"] += 1
                    continue
                
                # Property keys differ by TomTom tile version; try common long + short forms.
                current_speed = _get_prop_float(
                    props,
                    self.CURRENT_SPEED_KEYS,
                )
                if current_speed is None:
                    # If we can't read speeds, this segment is not useful for calibration.
                    diagnostics["feature_drop_reasons"]["missing_speed_property"] += 1
                    continue
                
                freeflow = _get_prop_float(
                    props,
                    self.FREEFLOW_SPEED_KEYS,
                )
                if freeflow is None:
                    freeflow = current_speed
                
                confidence = _get_prop_float(props, self.CONFIDENCE_KEYS)
                if confidence is None:
                    confidence = 0.9
                
                segment_id = (
                    props.get("id")
                    or props.get("segmentId")
                    or props.get("segment_id")
                    or props.get("uuid")
                )
                if segment_id is None:
                    h = hashlib.sha1(repr(coords).encode("utf-8")).hexdigest()
                    segment_id = f"{layer_name}_{h}"
                segments.append({
                    "segment_id": str(segment_id),
                    "geometry": coords,
                    "current_speed": float(current_speed),
                    "freeflow_speed": float(freeflow),
                    "timestamp": datetime.utcnow(),
                    "quality": float(confidence) if confidence is not None else 0.9
                })
                diagnostics["segments_decoded"] += 1
        return segments, diagnostics
    
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
        seen_ids = set()
        diagnostics = self._empty_tile_diagnostics(len(tiles))
        
        for x, y in tiles:
            tile_bytes, fetch_reason = await self._fetch_tile(x, y)
            if not tile_bytes:
                diagnostics["tile_fetch_failed"] += 1
                self._increment_count(diagnostics["tile_fetch_fail_reasons"], fetch_reason)
                continue
            diagnostics["tile_fetch_ok"] += 1
            decoded, tile_decode_diag = self._decode_tile(tile_bytes, x, y)
            diagnostics["tile_decode_failures"] += int(tile_decode_diag.get("decode_error", 0))
            diagnostics["layers_total"] += int(tile_decode_diag.get("layers_total", 0))
            diagnostics["features_total"] += int(tile_decode_diag.get("features_total", 0))
            self._merge_counts(
                diagnostics["feature_drop_reasons"],
                tile_decode_diag.get("feature_drop_reasons", {}),
            )
            diagnostics["segments_decoded"] += int(tile_decode_diag.get("segments_decoded", 0))
            for seg in decoded:
                geom = seg.get("geometry") or []
                seg_bbox = _geometry_bbox(geom)
                if seg_bbox and not _bbox_intersects(seg_bbox, bbox):
                    diagnostics["segments_dropped_outside_bbox"] += 1
                    continue
                
                seg_id = seg.get("segment_id")
                if seg_id and seg_id in seen_ids:
                    diagnostics["segments_dropped_duplicate_id"] += 1
                    continue
                if seg_id:
                    seen_ids.add(seg_id)
                segments.append(seg)
                diagnostics["segments_kept"] += 1
        
        if segments:
            df = pd.DataFrame(segments)
            logger.info(f"Fetched {len(df)} segments from {len(tiles)} tiles (zoom={self.tile_zoom})")
            return df

        logger.warning(
            "Tile mode yielded 0 usable segments (zoom=%s, tiles=%s, fetch_ok=%s, "
            "fetch_failed=%s [%s], decode_failures=%s, layers=%s, features=%s, "
            "feature_drops=[%s], outside_bbox=%s, duplicate_id=%s)",
            self.tile_zoom,
            diagnostics["tiles_total"],
            diagnostics["tile_fetch_ok"],
            diagnostics["tile_fetch_failed"],
            self._format_counts(diagnostics["tile_fetch_fail_reasons"]),
            diagnostics["tile_decode_failures"],
            diagnostics["layers_total"],
            diagnostics["features_total"],
            self._format_counts(diagnostics["feature_drop_reasons"]),
            diagnostics["segments_dropped_outside_bbox"],
            diagnostics["segments_dropped_duplicate_id"],
        )
        return pd.DataFrame()
    
    async def _fetch_via_flow_segments(self, bbox: Tuple[float, float, float, float]) -> pd.DataFrame:
        """Fallback grid sampling."""
        west, south, east, north = bbox
        bbox_width = abs(east - west)
        bbox_height = abs(north - south)
        area_deg2 = bbox_width * bbox_height
        spacing = 0.002  # denser sampling for better Flow Segment coverage
        grid_dim = int(math.sqrt(area_deg2) / spacing)
        grid_size = max(3, min(20, grid_dim))
        
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
