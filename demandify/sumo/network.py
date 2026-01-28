"""
SUMO network conversion from OSM data.
"""
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET
from shapely.geometry import LineString
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
        
        logger.debug(f"Parsed {len(self.edges)} edges from network")
    
    def get_edge_geometry(self, edge_id: str) -> LineString:
        """Get the geometry for a given edge ID."""
        return self.edge_geometries.get(edge_id)
    
    def get_all_edges(self) -> List[str]:
        """Get all edge IDs."""
        return self.edges.copy()


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
