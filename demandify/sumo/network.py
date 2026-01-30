"""
SUMO network conversion from OSM data.
"""
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET
from shapely.geometry import LineString, Point
import json

logger = logging.getLogger(__name__)


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
        self.edge_geometries = {}
        self.edge_attributes = {}
        self.adjacency = {}  # Directed graph: {from_edge: {to_edges}}
        
        if network_file.exists():
            self._parse_network()
    
    def _parse_network(self):
        """Parse the SUMO network file to extract edge geometries."""
        logger.debug(f"Parsing SUMO network: {self.network_file}")
        
        tree = ET.parse(self.network_file)
        root = tree.getroot()
        
        # Extract edges
        for edge in root.findall('.//edge'):
            edge_id = edge.get('id')
            
            # Skip internal edges
            if edge_id.startswith(':'):
                continue
            
            # Get lanes (use first lane geometry for edge)
            lanes = edge.findall('lane')
            if not lanes:
                continue
            
            first_lane = lanes[0]
            shape_str = first_lane.get('shape')
            
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
                        'speed': float(first_lane.get('speed', 13.89)),  # default 50km/h
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
            if from_edge.startswith(':') or to_edge.startswith(':'):
                continue
                
            if from_edge not in self.adjacency:
                self.adjacency[from_edge] = set()
            self.adjacency[from_edge].add(to_edge)
        
        logger.debug(f"Parsed {len(self.edges)} edges and topology from network")
    
    def get_connected_edges(self) -> List[str]:
        """
        Get the list of edges belonging to the Largest Connected Component (LCC).
        Uses BFS to find the largest weakly accessible component.
        """
        if not self.edges:
            return []
            
        # If no connections parsed (shouldn't happen in valid net), return all
        if not self.adjacency:
            return self.edges.copy()
            
        # Build undirected graph for "Weak Connectivity"
        # Or just use directed? For driving, Strong Connectivity is ideal,
        # but finding LCC of directed graph (SCC) is harder (Tarjan's).
        # We can approximate by finding Largest Weakly Connected Component
        # and assuming local reachability.
        # Actually, let's keep it simple: BFS from the edge with most outgoing connections?
        
        # Better: BFS from every unvisited node to find all components.
        visited = set()
        components = []
        
        # Create undirected adjacency for Weak Connectivity check
        undirected_adj = {e: set() for e in self.edges}
        for u, neighbors in self.adjacency.items():
            for v in neighbors:
                if u in undirected_adj and v in undirected_adj:
                    undirected_adj[u].add(v)
                    undirected_adj[v].add(u)
                    
        sorted_edges = sorted(self.edges) # Deterministic order
        
        for start_node in sorted_edges:
            if start_node in visited:
                continue
                
            # BFS for this component
            component = set()
            queue = [start_node]
            visited.add(start_node)
            component.add(start_node)
            
            idx = 0
            while idx < len(queue):
                u = queue[idx]
                idx += 1
                
                # Neighbors (undirected)
                for v in undirected_adj.get(u, []):
                    if v not in visited:
                        visited.add(v)
                        component.add(v)
                        queue.append(v)
            
            components.append(component)
            
        if not components:
            return []
            
        # Return largest component
        largest = max(components, key=len)
        logger.info(f"Connectivity Check: Found {len(components)} components. Largest has {len(largest)} edges ({(len(largest)/len(self.edges))*100:.1f}% coverage)")
        
        return list(largest)
    
    def get_edge_geometry(self, edge_id: str) -> LineString:
        """Get the geometry for a given edge ID."""
        return self.edge_geometries.get(edge_id)
        
    def get_edge_attributes(self, edge_id: str) -> Dict:
        """Get attributes (speed, priority, etc) for an edge."""
        return self.edge_attributes.get(edge_id, {})
    
    def get_all_edges(self) -> List[str]:
        """Get all edge IDs."""
        return self.edges.copy()

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
