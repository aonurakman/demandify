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
import math

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
        num_destinations: int = 20,
        max_od_pairs: int = 150,
        max_consecutive_failures: int = 10000,
        min_trip_distance: float = 0.0
    ) -> Tuple[List[str], List[str]]:
        """
        Select origin and destination edges by building validated OD pairs.
        Validates EACH pair individually and resamples failures.
        
        Args:
            num_origins: Hint for origins (actual will vary based on validated pairs)
            num_destinations: Hint for destinations (actual will vary)
            max_od_pairs: Target number of OD pairs to create
            max_consecutive_failures: Max failures before giving up
            min_trip_distance: Minimum Euclidean distance between origin and destination O/D
        
        Returns:
            (origin_edges, destination_edges) - unique edges used in validated OD pairs
        """
        all_edges = self.network.get_all_edges()
        
        if len(all_edges) < 2:
            raise ValueError(f"Insufficient edges in network: {len(all_edges)}")
        
        # Calculate weights for biased selection (favor major roads)
        weights = self._calculate_edge_weights(all_edges)
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights] if total_weight > 0 else None
        
        # Build valid OD pairs one at a time
        valid_pairs = []
        consecutive_failures = 0
        total_attempts = 0
        
        logger.info(f"Building up to {max_od_pairs} validated OD pairs (min_dist={min_trip_distance}m)...")
        
        current_min_dist = min_trip_distance
        logger.info(f"Generating {max_od_pairs} OD pairs (min_dist={int(min_trip_distance)}m)...")
        
        while len(valid_pairs) < max_od_pairs:
            # Safety break
            if total_attempts > max_od_pairs * 100 and total_attempts > 10000:
                logger.warning(f"Reached maximum attempt limit ({total_attempts}). Stopping with {len(valid_pairs)} pairs.")
                break
                
            if consecutive_failures > max_consecutive_failures:
                logger.warning(f"Stopped after {consecutive_failures} consecutive failures. Created {len(valid_pairs)} pairs.")
                break
                
            # Progress Logging (every 100 pairs)
            if len(valid_pairs) > 0 and len(valid_pairs) % 100 == 0 and consecutive_failures == 0:
                 logger.info(f"  ... {len(valid_pairs)}/{max_od_pairs} OD pairs found")

            # Adaptive Relaxation: If stuck, reduce min distance requirement
            if consecutive_failures > 500 and consecutive_failures % 500 == 0 and current_min_dist > 0:
                 old_dist = current_min_dist
                 current_min_dist *= 0.8
                 logger.info(f"  Relaxing min_dist from {int(old_dist)}m to {int(current_min_dist)}m after failures")

            # Sample a random OD pair
            candidates = self.rng.choice(all_edges, size=2, replace=False, p=probs)
            origin, destination = candidates[0], candidates[1]
            
            total_attempts += 1
            
            # 1. Filter by minimum Euclidean distance
            valid_dist = True
            if current_min_dist > 0:
                ox, oy = self.network.get_edge_centroid(origin)
                dx, dy = self.network.get_edge_centroid(destination)
                dist = math.hypot(dx - ox, dy - oy)
                if dist < current_min_dist:
                    valid_dist = False
            
            if not valid_dist:
                consecutive_failures += 1
                continue

            # 2. Validate reachability for this specific pair
            if self._has_route(origin, destination):
                valid_pairs.append((origin, destination))
                consecutive_failures = 0  # Reset on success
            else:
                consecutive_failures += 1
        
        if len(valid_pairs) == 0:
            raise ValueError("Could not create any valid OD pairs - network may be disconnected or constraints too strict")
        
        if consecutive_failures >= max_consecutive_failures:
            logger.warning(f"Stopped after {consecutive_failures} consecutive failures. "
                          f"Created {len(valid_pairs)} pairs (target was {max_od_pairs})")
        
        # Extract unique origins and destinations from validated pairs
        origins = list(set(pair[0] for pair in valid_pairs))
        destinations = list(set(pair[1] for pair in valid_pairs))
        
        logger.info(f"Created {len(valid_pairs)} valid OD pairs from {total_attempts} attempts: "
                   f"{len(origins)} unique origins, {len(destinations)} unique destinations")
        
        return origins, destinations
    
    def _calculate_edge_weights(self, edges: List[str]) -> List[float]:
        """Calculate selection weights for edges based on road importance."""
        weights = []
        for edge in edges:
            attrs = self.network.get_edge_attributes(edge)
            # Default priority 1 (minor), speed 13.89 (50kmh), lanes 1
            p = max(1, attrs.get('priority', 1))
            s = max(5.0, attrs.get('speed', 13.89))
            l = max(1, attrs.get('numLanes', 1))
            
            # Boost highways significantly
            weight = p * s * l
            weights.append(weight)
        return weights
    
    def _has_route(self, from_edge: str, to_edge: str) -> bool:
        """
        Check if there exists a directed route from from_edge to to_edge.
        Uses BFS on the network adjacency graph.
        
        Args:
            from_edge: Origin edge ID
            to_edge: Destination edge ID
            
        Returns:
            True if a route exists, False otherwise
        """
        if from_edge == to_edge:
            return True
        
        # BFS for reachability
        visited = {from_edge}
        queue = [from_edge]
        idx = 0
        max_depth = 1000  # Prevent infinite loops on large networks
        
        while idx < len(queue) and idx < max_depth:
            current = queue[idx]
            idx += 1
            
            # Get neighbors (outgoing edges)
            neighbors = self.network.adjacency.get(current, set())
            
            for neighbor in neighbors:
                if neighbor == to_edge:
                    return True  # Found a path!
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return False  # No path found
    
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
        
        # Sort by departure time (CRITICAL for SUMO)
        if not demand_df.empty:
            demand_df = demand_df.sort_values('departure timestep')
        
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

