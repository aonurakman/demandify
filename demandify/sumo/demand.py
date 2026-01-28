"""
Seeded demand generation for SUMO.
"""
from typing import List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import logging
import subprocess

from demandify.sumo.network import SUMONetwork

logger = logging.getLogger(__name__)


class DemandGenerator:
    """Generate seeded synthetic demand for SUMO."""
    
    def __init__(self, network: SUMONetwork, seed: int = 42):
        """
        Initialize demand generator.
        
        Args:
            network: SUMO network
            seed: Random seed for reproducibility
        """
        self.network = network
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def select_od_candidates(
        self,
        num_origins: int = 20,
        num_destinations: int = 20
    ) -> Tuple[List[str], List[str]]:
        """
        Select candidate origin and destination edges.
        Seeded selection for reproducibility.
        
        Args:
            num_origins: Number of origin edges
            num_destinations: Number of destination edges
        
        Returns:
            (origin_edges, destination_edges)
        """
        all_edges = self.network.get_all_edges()
        
        if len(all_edges) < num_origins + num_destinations:
            logger.warning(f"Not enough edges. Requested {num_origins + num_destinations}, have {len(all_edges)}")
            num_origins = min(num_origins, len(all_edges) // 2)
            num_destinations = min(num_destinations, len(all_edges) // 2)
        
        # Seeded random selection
        selected = self.rng.choice(all_edges, size=num_origins + num_destinations, replace=False)
        
        origins = selected[:num_origins].tolist()
        destinations = selected[num_origins:].tolist()
        
        logger.info(f"Selected {len(origins)} origins and {len(destinations)} destinations")
        
        return origins, destinations
    
    def genome_to_demand_csv(
        self,
        genome: np.ndarray,
        od_pairs: List[Tuple[str, str]],
        departure_bins: List[Tuple[int, int]],
        output_file: Path
    ) -> pd.DataFrame:
        """
        Convert a genome (vehicle counts per OD pair and time bin) to demand.csv.
        
        Args:
            genome: 1D array of vehicle counts (length = num_od_pairs * num_bins)
            od_pairs: List of (origin_edge, destination_edge) tuples
            departure_bins: List of (start_time, end_time) tuples in seconds
            output_file: Path to save demand.csv
        
        Returns:
            DataFrame with demand
        """
        num_od = len(od_pairs)
        num_bins = len(departure_bins)
        
        assert len(genome) == num_od * num_bins, "Genome size mismatch"
        
        # Reshape genome to (num_od, num_bins)
        counts = genome.reshape(num_od, num_bins)
        
        # Generate individual trips
        trips = []
        trip_id = 0
        
        for od_idx, (origin, dest) in enumerate(od_pairs):
            for bin_idx, (start_time, end_time) in enumerate(departure_bins):
                count = int(counts[od_idx, bin_idx])
                
                # Generate individual departure times within the bin
                if count > 0:
                    # Seeded random jitter
                    bin_rng = np.random.RandomState(self.seed + trip_id)
                    departure_times = bin_rng.uniform(start_time, end_time, size=count)
                    
                    for dep_time in departure_times:
                        trips.append({
                            'ID': f'trip_{trip_id}',
                            'origin link id': origin,
                            'destination link id': dest,
                            'departure timestep': int(dep_time)
                        })
                        trip_id += 1
        
        # Create DataFrame
        demand_df = pd.DataFrame(trips)
        
        # Save to CSV
        output_file.parent.mkdir(parents=True, exist_ok=True)
        demand_df.to_csv(output_file, index=False)
        
        logger.debug(f"Generated {len(demand_df)} trips in demand.csv: {output_file}")
        
        return demand_df
    
    def demand_csv_to_trips_xml(
        self,
        demand_csv: Path,
        output_trips_file: Path
    ):
        """
        Convert demand.csv to SUMO trips.xml format.
        
        Args:
            demand_csv: Path to demand.csv
            output_trips_file: Path for output trips.xml
        """
        # Read demand
        demand_df = pd.read_csv(demand_csv)
        
        # Create XML
        root = ET.Element('routes')
        
        for _, row in demand_df.iterrows():
            trip = ET.SubElement(root, 'trip')
            trip.set('id', row['ID'])
            trip.set('depart', str(row['departure timestep']))
            trip.set('from', row['origin link id'])
            trip.set('to', row['destination link id'])
        
        # Write to file
        tree = ET.ElementTree(root)
        ET.indent(tree, space='  ')
        tree.write(output_trips_file, encoding='utf-8', xml_declaration=True)
        
        logger.debug(f"Created trips.xml: {output_trips_file}")
    
    def route_trips(
        self,
        network_file: Path,
        trips_file: Path,
        output_routes_file: Path
    ):
        """
        Route trips using duarouter.
        
        Args:
            network_file: SUMO network .net.xml
            trips_file: trips.xml file
            output_routes_file: Output routes.rou.xml file
        """
        logger.debug("🚗 Routing trips with duarouter")
        
        # Count trips in input file
        try:
            tree = ET.parse(trips_file)
            root = tree.getroot()
            num_trips = len(root.findall('trip'))
            logger.debug(f"  Input: {num_trips} trips to route")
        except Exception as e:
            logger.error(f"  Could not parse trips file: {e}")
            num_trips = "unknown"
        
        cmd = [
            "duarouter",
            "--net-file", str(network_file),
            "--trip-files", str(trips_file),
            "--output-file", str(output_routes_file),
            "--ignore-errors",  # Continue on errors
            "--repair",  # Try to repair routes
            "--no-warnings",
            "--seed", str(self.seed)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Count successfully routed vehicles
            try:
                tree = ET.parse(output_routes_file)
                root = tree.getroot()
                num_routes = len(root.findall('vehicle'))
                num_routes += len(root.findall('trip'))  # Some might still be trips
                
                logger.debug(f"  ✅ Output: {num_routes} routes generated")
                
                if num_routes == 0:
                    logger.error(f"  ❌ CRITICAL: duarouter produced 0 routes from {num_trips} trips!")
                    logger.error(f"  duarouter stderr: {result.stderr}")
                elif isinstance(num_trips, int) and num_routes < num_trips * 0.5:
                    logger.warning(f"  ⚠️  Low routing success: {num_routes}/{num_trips} ({100*num_routes/num_trips:.1f}%)")
                    if result.stderr:
                        logger.warning(f"  duarouter stderr: {result.stderr[:500]}")
                else:
                    if isinstance(num_trips, int):
                        logger.debug(f"  Routing success: {100*num_routes/num_trips:.1f}%")
                    
            except Exception as e:
                logger.error(f"  Could not parse routes file: {e}")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ duarouter failed: {e.stderr}")
            raise RuntimeError(f"Failed to route trips: {e.stderr}")

