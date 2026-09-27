"""
SUMO network conversion from OSM data.
"""
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
from shapely.geometry import LineString, Point
import json

logger = logging.getLogger(__name__)

_DEFAULT_VCLASS = "passenger"


def _parse_vclass_list(value: str) -> set:
    if not value:
        return set()
    return set(value.split())


def _is_vclass_allowed(*, allow: str, disallow: str, vclass: str) -> bool:
    if allow:
        return vclass in _parse_vclass_list(allow)
    if disallow:
        return vclass not in _parse_vclass_list(disallow)
    return True


class SUMONetwork:
    """Handle SUMO network conversion and edge geometry extraction."""
    
    def __init__(self, network_file: Path):
        """
        Initialize with a SUMO network file.
        
        Args:
            network_file: Path to .net.xml file
        """
        self.network_file = network_file
        self.edges = []
        self.edge_geometries = {}
        self.edge_attributes = {}
        self.adjacency = {}  # Directed graph: {from_edge: {to_edges}}
        self.edge_allowed_lanes = {}  # {edge_id: {lane_index, ...}} for passenger routing
        self.conv_boundary: Optional[Tuple[float, float, float, float]] = None
        self.orig_boundary: Optional[Tuple[float, float, float, float]] = None
        
        if network_file.exists():
            self._parse_network()

    @staticmethod
    def _parse_boundary_attr(value: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
        """Parse SUMO location boundary attributes like 'minX,minY,maxX,maxY'."""
        if not value:
            return None
        try:
            min_x, min_y, max_x, max_y = map(float, value.split(","))
            return (min_x, min_y, max_x, max_y)
        except Exception:
            return None
    
    def _parse_network(self):
        """Parse the SUMO network file to extract edge geometries."""
        logger.debug(f"Parsing SUMO network: {self.network_file}")
        
        tree = ET.parse(self.network_file)
        root = tree.getroot()

        location = root.find(".//location")
        if location is not None:
            self.conv_boundary = self._parse_boundary_attr(location.get("convBoundary"))
            self.orig_boundary = self._parse_boundary_attr(location.get("origBoundary"))
        
        # Extract edges
        for edge in root.findall('.//edge'):
            edge_id = edge.get('id')
            
            # Skip internal edges
            if not edge_id or edge_id.startswith(':'):
                continue
            
            # Get lanes (use first lane geometry for edge)
            lanes = edge.findall('lane')
            if not lanes:
                continue
            
            allowed_lanes = set()
            for lane in lanes:
                try:
                    lane_idx = int(lane.get("index", "0"))
                except Exception:
                    continue
                if _is_vclass_allowed(
                    allow=lane.get("allow", ""),
                    disallow=lane.get("disallow", ""),
                    vclass=_DEFAULT_VCLASS,
                ):
                    allowed_lanes.add(lane_idx)

            self.edge_allowed_lanes[edge_id] = allowed_lanes

            if not allowed_lanes:
                continue

            lane_for_geom = None
            for lane in lanes:
                try:
                    lane_idx = int(lane.get("index", "0"))
                except Exception:
                    continue
                if lane_idx in allowed_lanes:
                    lane_for_geom = lane
                    break
            if lane_for_geom is None:
                lane_for_geom = lanes[0]

            shape_str = lane_for_geom.get('shape')
            
            if shape_str:
                # Parse shape: "x1,y1 x2,y2 x3,y3 ..."
                coords = []
                for point in shape_str.split():
                    x, y = map(float, point.split(','))
                    coords.append((x, y))
                
                if len(coords) >= 2:
                    self.edges.append(edge_id)
                    self.edge_geometries[edge_id] = LineString(coords)
                    
                    # Store attributes for filtering
                    # Use first lane's speed/width as proxy for edge
                    self.edge_attributes[edge_id] = {
                        'speed': float(lane_for_geom.get('speed', 13.89)),  # default 50km/h
                        'priority': int(edge.get('priority', -1)),
                        'numLanes': len(lanes),
                        'type': edge.get('type', '')
                    }
        
        # Extract connections (topology)
        # Assuming simple connections: from edge -> to edge
        for conn in root.findall('.//connection'):
            from_edge = conn.get('from')
            to_edge = conn.get('to')
            
            # Skip internal edges in topology
            if not from_edge or not to_edge or from_edge.startswith(':') or to_edge.startswith(':'):
                continue

            try:
                from_lane = int(conn.get("fromLane", "0"))
                to_lane = int(conn.get("toLane", "0"))
            except Exception:
                from_lane = None
                to_lane = None

            allowed_from = self.edge_allowed_lanes.get(from_edge)
            if from_lane is not None and allowed_from is not None and from_lane not in allowed_from:
                continue

            allowed_to = self.edge_allowed_lanes.get(to_edge)
            if to_lane is not None and allowed_to is not None and to_lane not in allowed_to:
                continue
                
            if from_edge not in self.adjacency:
                self.adjacency[from_edge] = set()
            self.adjacency[from_edge].add(to_edge)
        
        logger.debug(f"Parsed {len(self.edges)} edges and topology from network")
    
    def get_edge_geometry(self, edge_id: str) -> LineString:
        """Get the geometry for a given edge ID."""
        return self.edge_geometries.get(edge_id)
        
    def get_edge_attributes(self, edge_id: str) -> Dict:
        """Get attributes (speed, priority, etc) for an edge."""
        return self.edge_attributes.get(edge_id, {})
    
    def get_all_edges(self) -> List[str]:
        """Get all edge IDs."""
        return self.edges.copy()

    def get_network_boundary(self) -> Optional[Tuple[float, float, float, float]]:
        """Return network-space bounds, preferring SUMO's converted bbox when available."""
        if self.conv_boundary is not None:
            return self.conv_boundary
        if not self.edge_geometries:
            return None

        min_x = min(geom.bounds[0] for geom in self.edge_geometries.values())
        min_y = min(geom.bounds[1] for geom in self.edge_geometries.values())
        max_x = max(geom.bounds[2] for geom in self.edge_geometries.values())
        max_y = max(geom.bounds[3] for geom in self.edge_geometries.values())
        return (min_x, min_y, max_x, max_y)

    def get_edge_centroid(self, edge_id: str) -> Tuple[float, float]:
        """
        Get the centroid coordinates (x, y) of an edge.
        Returns:
            (x, y) tuple, or (0,0) if geometry is missing
        """
        geom = self.edge_geometries.get(edge_id)
        if geom:
            p = geom.centroid
            return (p.x, p.y)
        return (0.0, 0.0)


def convert_osm_to_sumo(
    osm_file: Path,
    output_net_file: Path,
    car_only: bool = True,
    seed: int = 42
) -> Tuple[Path, Dict]:
    """
    Convert OSM data to SUMO network using netconvert.
    
    Args:
        osm_file: Path to OSM XML file
        output_net_file: Path for output .net.xml file
        car_only: If True, only include car-accessible roads
        seed: Random seed for reproducibility
    
    Returns:
        (output_net_file, metadata)
    """
    logger.debug(f"Converting OSM to SUMO network: {osm_file} -> {output_net_file}")
    
    # Build netconvert command
    cmd = [
        "netconvert",
        "--osm-files", str(osm_file),
        "--output-file", str(output_net_file),
        "--geometry.remove",  # Remove geometry discontinuities
        "--roundabouts.guess",  # Guess roundabouts
        "--ramps.guess",  # Guess highway ramps
        "--junctions.join",  # Join junctions
        "--tls.guess-signals",  # Guess traffic lights
        "--tls.discard-simple",  # Discard simple TLS
        "--tls.join",  # Cluster multi-node traffic light junctions
        "--tls.default-type", "actuated",  # Demand-responsive actuated traffic lights
        "--crossings.guess",  # Guess pedestrian crossings at junctions
        "--crossings.guess.roundabout-priority",  # Prioritize roundabout flow over crossings
        "--remove-edges.isolated",  # Remove isolated edges
        "--keep-edges.components", "1",  # Keep only largest connected component
        "--seed", str(seed)
    ]
    
    if car_only:
        # Only keep edges accessible to passenger cars
        # This is simpler and more reliable than complex remove chains
        cmd.extend([
            "--keep-edges.by-vclass", "passenger"
        ])
    
    # Create output directory
    output_net_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run netconvert
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.debug(f"Network conversion successful: {output_net_file}")
        
        # Create metadata
        metadata = {
            "osm_file": str(osm_file),
            "output_file": str(output_net_file),
            "car_only": car_only,
            "seed": seed,
            "netconvert_args": cmd
        }
        
        # Save metadata
        meta_file = output_net_file.with_suffix('.meta.json')
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return output_net_file, metadata
        
    except subprocess.CalledProcessError as e:
        logger.error(f"netconvert failed: {e.stderr}")
        raise RuntimeError(f"Failed to convert OSM to SUMO: {e.stderr}")


# ---------------------------------------------------------------------------
# Effective-capacity derating helpers
# ---------------------------------------------------------------------------

_VTYPE_REF_SPEED_MS = 13.89  # 50 km/h — reference speed for tau derivation
_VTYPE_EFFECTIVE_LENGTH = 7.5  # vehicle length + min gap (m), Krauss defaults


def tau_from_capacity_factor(factor: float) -> float:
    """Convert an effective-capacity factor to a SUMO Krauss car-following tau.

    In the Krauss model, road throughput capacity at speed *v* is approximately:

        C(tau) = v / (L + tau * v)

    where *L* is the effective vehicle length (length + minGap).  Setting a
    higher *tau* increases the required headway between vehicles, which reduces
    the number of cars a lane can carry — equivalent to the friction introduced
    by trucks, buses, and other mixed-traffic modes absent from the simulation.

    The factor is defined relative to the default SUMO tau of 1.0 s:

        factor = C(tau_new) / C(tau_default=1.0)

    Solved for tau_new at the reference speed (50 km/h):

        tau_new = (L + v_ref) / (factor * v_ref) - L / v_ref

    Args:
        factor: Target capacity fraction relative to the default car-only
            capacity.  Must be in (0, 1].  1.0 returns the SUMO default tau
            (1.0 s) and means no derating.  0.85 gives roughly 15 % capacity
            reduction, matching typical urban mixed-traffic conditions.

    Returns:
        tau in seconds (≥ 1.0 s).
    """
    if not (0.0 < factor <= 1.0):
        raise ValueError(f"effective_capacity_factor must be in (0, 1]; got {factor}")
    v = _VTYPE_REF_SPEED_MS
    L = _VTYPE_EFFECTIVE_LENGTH
    tau = (L + v) / (factor * v) - L / v
    return max(1.0, tau)


def write_vehicle_types_xml(tau: float, output_path: Path) -> None:
    """Write a SUMO additional-file that overrides the passenger vType tau.

    Only *tau* is overridden; all other default parameters (length, minGap,
    accelration, sigma, …) remain at their SUMO defaults.  This file should be
    loaded via ``--additional-files`` in every simulation that uses the derated
    capacity model.

    Args:
        tau: The car-following reaction-time in seconds (≥ 1.0).
        output_path: Destination path for the XML file.
    """
    root = ET.Element("additional")
    ET.SubElement(root, "vType", {"id": "passenger", "tau": f"{tau:.4f}"})
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    logger.debug(
        "Written vehicle_types.xml with tau=%.4f (capacity factor ≈ %.2f): %s",
        tau,
        _VTYPE_REF_SPEED_MS / (_VTYPE_EFFECTIVE_LENGTH + tau * _VTYPE_REF_SPEED_MS)
        / (_VTYPE_REF_SPEED_MS / (_VTYPE_EFFECTIVE_LENGTH + 1.0 * _VTYPE_REF_SPEED_MS)),
        output_path,
    )
