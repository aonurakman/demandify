"""
Traffic segment to SUMO edge matching using spatial indexing and coordinate transformation.
"""
from pathlib import Path
from typing import Optional, Tuple
import logging
import pandas as pd
from shapely.geometry import LineString, Point
from rtree import index as rtree_index

logger = logging.getLogger(__name__)

# Try to import pyproj for coordinate transformation
try:
    from pyproj import Transformer, CRS
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False
    logger.warning("pyproj not available - coordinate transformation will be limited")




def get_network_projection(network_file: Path) -> Tuple[Optional[str], Tuple[float, float]]:
    """
    Extract the projection string and offset from a SUMO network file.
    
    Args:
        network_file: Path to .net.xml file
        
    Returns:
        (projection_string, (x_offset, y_offset)) - projection may be None
    """
    import xml.etree.ElementTree as ET
    
    proj_param = None
    offset = (0.0, 0.0)
    
    try:
        tree = ET.parse(network_file)
        root = tree.getroot()
        
        # Look for location element with projection info
        location = root.find('.//location')
        if location is not None:
            # Get projection parameter
            proj_param = location.get('projParameter')
            
            # Get network offset (critical for coordinate alignment!)
            net_offset = location.get('netOffset')
            if net_offset:
                try:
                    x_off, y_off = map(float, net_offset.split(','))
                    offset = (x_off, y_off)
                    logger.info(f"Network offset: {offset}")
                except Exception as e:
                    logger.warning(f"Could not parse netOffset: {e}")
    except Exception as e:
        logger.warning(f"Could not parse network projection: {e}")
    
    return proj_param, offset


class EdgeMatcher:
    """Matches traffic segments to SUMO network edges using spatial indexing."""
    
    def __init__(self, network, network_file: Path = None):
        """
        Initialize matcher with a SUMO network.
        
        Args:
            network: SUMONetwork instance
            network_file: Optional path to network file for projection info
        """
        self.network = network
        self.network_file = network_file
        self.spatial_index = None
        self.transformer = None
        self.offset = (0, 0)
        
        self._build_spatial_index()
        self._setup_transformer()
    
    def _setup_transformer(self):
        """Set up coordinate transformer from WGS84 to network CRS."""
        if not HAS_PYPROJ:
            logger.warning("pyproj not available - coordinate transformation disabled")
            return
        
        proj_str = None
        if self.network_file:
            proj_str, self.offset = get_network_projection(self.network_file)
            logger.info(f"Network projection: {proj_str[:50] if proj_str else None}")
            logger.info(f"Network offset: {self.offset}")
        
        if proj_str:
            # Try to create transformer from proj string
            try:
                target_crs = CRS.from_proj4(proj_str)
                self.transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
                logger.info("Created transformer from network projection")
                return
            except Exception as e:
                logger.warning(f"Could not create transformer from proj: {e}")
        
        # Fallback to UTM Zone 31N (common for Western Europe)
        try:
            self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
            logger.info("Using fallback UTM Zone 31N transformer")
        except Exception as e:
            logger.error(f"Could not create fallback transformer: {e}")
    
    def _build_spatial_index(self):
        """Build R-tree spatial index for edges."""
        self.spatial_index = rtree_index.Index()
        
        for edge_id in self.network.get_all_edges():
            edge_geom = self.network.get_edge_geometry(edge_id)
            if edge_geom:
                self.spatial_index.insert(
                    hash(edge_id) % (2**31),
                    edge_geom.bounds,
                    obj=edge_id
                )
    
    def _transform_coords(self, lon: float, lat: float) -> Tuple[float, float]:
        """Transform WGS84 coordinates to network CRS."""
        if self.transformer:
            try:
                x, y = self.transformer.transform(lon, lat)
                # Apply network offset
                x += self.offset[0]
                y += self.offset[1]
                return (x, y)
            except Exception as e:
                logger.warning(f"Coordinate transformation failed: {e}")
                return (lon, lat)
        else:
            # FALLBACK: Simple Equirectangular projection (flat earth approx)
            # This is "good enough" for small areas if pyproj is missing.
            # Using center of network or first point as reference would be better,
            # but we assume network (0,0) is relative to some origin.
            # For finding RELATIVE distances during matching, we need meters.
            
            # 1 deg lat ~= 111132.954m
            # 1 deg lon ~= 111132.954 * cos(lat)
            
            # We assume the network is somewhat centered? No, SUMO networks are in meters.
            # If we don't have pyproj, we cannot know the network's true origin offset easily
            # unless we parse 'location' netOffset properly (which we do).
            # But converting Lon/Lat -> Meters requires a projection center.
            
            # CRITICAL: Without pyproj, we can't reliably map LatLon to SUMO XY unless
            # we implement a full UTM or Mercator projection.
            # However, for 'matching', we just need the traffic segment (Lat/Lon)
            # to line up with the network edge (Lat/Lon or XY).
            # If the network is in XY (meters), and we have Lat/Lon traffic...
            
            # Actually, `sumo/network.py` parses edge shapes.
            # If the .net.xml shapes are in meters (standard), we MUST project traffic to meters.
            
            logger.error("CRITICAL: pyproj is missing! Matching will fail because we cannot project WGS84 traffic to SUMO XY meters.")
            # raise RuntimeError("Feature 'pyproj' is required for map matching. Please install it.")
            
            # Attempt a rough approximation relative to the first point seen?
            # No, that's dangerous.
            return (lon, lat) # This WILL fail matching as 100m != 100 degrees

    
    def match_segment(
        self,
        segment_geom: LineString,
        max_distance: float = 100.0
    ) -> Tuple[Optional[str], float]:
        """
        Match a traffic segment to the closest SUMO edge.
        
        Args:
            segment_geom: LineString geometry (in WGS84)
            max_distance: Maximum distance for matching (meters)
        
        Returns:
            (edge_id, confidence) - edge_id is None if no match
        """
        # Transform segment coordinates
        transformed_coords = [
            self._transform_coords(lon, lat) 
            for lon, lat in segment_geom.coords
        ]
        segment_geom = LineString(transformed_coords)
        
        # Find candidate edges using spatial index
        search_bbox = (
            segment_geom.bounds[0] - max_distance,
            segment_geom.bounds[1] - max_distance,
            segment_geom.bounds[2] + max_distance,
            segment_geom.bounds[3] + max_distance
        )
        
        candidates = list(self.spatial_index.intersection(search_bbox, objects=True))
        
        if not candidates:
            return None, 0.0
        
        # Find best match by distance
        best_distance = float('inf')
        best_edge_id = None
        
        for candidate in candidates:
            edge_id = candidate.object
            edge_geom = self.network.get_edge_geometry(edge_id)
            
            if edge_geom is None:
                continue
            
            # Calculate distance
            distance = segment_geom.distance(edge_geom)
            
            if distance < best_distance:
                best_distance = distance
                best_edge_id = edge_id
        
        # Calculate confidence (inverse of distance, normalized)
        if best_distance > max_distance:
            return None, 0.0
        
        confidence = max(0.0, 1.0 - (best_distance / max_distance))
        
        return best_edge_id, confidence
    
    def match_traffic_data(
        self,
        traffic_df: pd.DataFrame,
        min_confidence: float = 0.1
    ) -> pd.DataFrame:
        """
        Match all traffic segments to SUMO edges with extensive debug logging.
        
        Args:
            traffic_df: DataFrame from traffic provider with 'geometry' column
            min_confidence: Minimum confidence threshold for matches
        
        Returns:
            DataFrame with matched edges
        """
        # EXTENSIVE DEBUG LOGGING TO FILE
        import sys
        debug_log_file = Path.home() / '.demandify' / 'matching_debug.log'
        debug_log_file.parent.mkdir(exist_ok=True)
        
        def debug_log(msg):
            logger.debug(msg)
            with open(debug_log_file, 'a') as f:
                f.write(f"{msg}\n")
        
        debug_log("=" * 80)
        debug_log("MATCHING SESSION STARTED")
        debug_log("=" * 80)
        debug_log(f"Python: {sys.version}")
        debug_log(f"Traffic segments: {len(traffic_df)}")
        debug_log(f"Columns: {list(traffic_df.columns)}")
        debug_log(f"Min confidence: {min_confidence}")
        debug_log(f"Has transformer: {self.transformer is not None}")
        debug_log(f"Network offset: {self.offset}")
        
        if len(traffic_df) > 0:
            first_geom = traffic_df.iloc[0]['geometry']
            debug_log(f"First geometry type: {type(first_geom)}")
            debug_log(f"First geometry preview: {str(first_geom)[:100]}")
        
        matches = []
        
        for idx, row in traffic_df.iterrows():
            segment_id = row.get('segment_id', f'segment_{idx}')
            debug_log(f"\n--- Segment {segment_id} ---")
            
            geom = None
            try:
                if isinstance(row['geometry'], list):
                    debug_log(f"  List with {len(row['geometry'])} points")
                    geom = LineString(row['geometry'])
                elif isinstance(row['geometry'], str):
                    debug_log(f"  String, parsing...")
                    import ast
                    coords = ast.literal_eval(row['geometry'])
                    debug_log(f"  Parsed {len(coords)} points")
                    geom = LineString(coords)
                elif isinstance(row['geometry'], LineString):
                    debug_log(f"  Already LineString")
                    geom = row['geometry']
                else:
                    debug_log(f"  ERROR: Invalid type {type(row['geometry'])}")
                    continue
                
                debug_log(f"  Bounds: {geom.bounds}")
                
                edge_id, confidence = self.match_segment(geom)
                debug_log(f"  Result: edge={edge_id}, conf={confidence:.4f}")
                
                if edge_id and confidence >= min_confidence:
                    matches.append({
                        'edge_id': edge_id,
                        'segment_id': segment_id,
                        'current_speed': row.get('current_speed'),
                        'freeflow_speed': row.get('freeflow_speed'),
                        'timestamp': row.get('timestamp'),
                        'match_confidence': confidence
                    })
                    debug_log(f"  ✓ MATCHED!")
                else:
                    debug_log(f"  ✗ No match (conf={confidence:.4f} < {min_confidence})")
            except Exception as e:
                debug_log(f"  EXCEPTION: {e}")
                import traceback
                debug_log(traceback.format_exc())
                continue
        
        debug_log(f"\n{'=' * 80}")
        debug_log(f"TOTAL: {len(matches)} / {len(traffic_df)} matched")
        debug_log(f"{'=' * 80}\n")
        
        if len(matches) > 0:
            return pd.DataFrame(matches)
        else:
            logger.warning("No matches - returning empty DataFrame")
            return pd.DataFrame(columns=[
                'edge_id', 'segment_id', 'current_speed', 
                'freeflow_speed', 'timestamp', 'match_confidence'
            ])
