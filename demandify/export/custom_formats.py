import csv
import shutil
import logging
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

from demandify.sumo.demand import DemandGenerator
from demandify.sumo.network import SUMONetwork

logger = logging.getLogger(__name__)


class URBDataExporter:
    """
    Exports simulation data using the URB project format.
    
    Structure:
    <run_id>/
        <run_id>/           # Subdirectory matching the scenario ID
            agents.csv          # Agent definitions (id, origin_idx, dest_idx, time, kind)
            od_<run_id>.txt    # Dictionary mapping indices to Origin/Destination Edge IDs
            <run_id>.net.xml   # SUMO Network
            <run_id>.rou.xml   # RouteRL/JanuX-compatible network graph copy
            <run_id>.edg.xml   # Plain Edges
            <run_id>.nod.xml   # Plain Nodes
            <run_id>.con.xml   # Plain Connections
            <run_id>.tll.xml   # Traffic Lights
            <run_id>.typ.xml   # Types
    """
    
    def __init__(self, run_id: str, output_dir: Path):
        self.run_id = run_id
        # The sub-folder named after the ID (e.g. run_paris/paris/)
        self.target_dir = output_dir / run_id
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
    def export(self, network_file: Path, routes_file: Path, min_connection_paths: int = 1):
        """Main export workflow."""
        logger.debug(f"Starting URB data export to {self.target_dir}")
        
        try:
            min_connection_paths = int(min_connection_paths)
            if min_connection_paths < 1:
                raise ValueError("min_connection_paths must be at least 1")

            # 1. Parse trips/routes and validate compatibility before writing files
            agents, origins, destinations, od_pairs = self._parse_routes(routes_file)
            self._validate_od_pairs(network_file, od_pairs, min_connection_paths)

            # 2. Copy Network and RouteRL graph companion with new names
            new_net = self.target_dir / f"{self.run_id}.net.xml"
            new_rou = self.target_dir / f"{self.run_id}.rou.xml"
            
            shutil.copy2(network_file, new_net)
            shutil.copy2(network_file, new_rou)
            
            # 2b. Copy Network Graph plot
            plot_file = self.target_dir.parent / "plots" / "network.png"
            if plot_file.exists():
                shutil.copy2(plot_file, self.target_dir / f"{self.run_id}.png")
            
            # 3. Generate Plain XML output (edg, nod, con, etc.)
            self._generate_plain_xml(new_net)

            # 4. Write agents.csv
            self._write_agents_csv(agents)
            
            # 5. Write od_<id>.txt
            self._write_od_txt(origins, destinations)
            
            logger.debug("URB data export completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to export URB data: {e}", exc_info=True)
            # Don't raise, just log error so pipeline doesn't crash on optional export

    def _generate_plain_xml(self, net_file: Path):
        """Run netconvert to generate plain output."""
        prefix = self.target_dir / self.run_id
        cmd = [
            "netconvert",
            "--sumo-net-file", str(net_file),
            "--plain-output-prefix", str(prefix),
            "--ignore-errors", "true",
            "--no-warnings", "true"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.debug(f"Generated plain XML files with prefix {prefix}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"netconvert failed to generate plain XMLs: {e.stderr}")
            # Non-critical, proceed

    def _parse_routes(
        self, routes_file: Path
    ) -> Tuple[List[Dict], List[str], List[str], List[Tuple[str, str]]]:
        """
        Parse .rou.xml to extract trips.
        Returns:
            agents: List of dicts
            origins: List of unique origin edge IDs
            destinations: List of unique destination edge IDs
            od_pairs: Sorted unique demanded OD edge pairs
        """
        agents = []
        unique_origins = set()
        unique_destinations = set()
        
        # We need to map edges to indices later, so we collect them first
        # But wait, we iterate once.
        # Let's collect raw data first
        raw_trips = []
        
        try:
            tree = ET.parse(routes_file)
            root = tree.getroot()
            
            # Parse <trip> (source) or <vehicle> (routed)
            # Since we use dynamic routing, we might only have trips.xml with <trip> elements
            # Or we might have a route file. We try both.
            
            # 1. Look for <trip> elements
            for i, trip in enumerate(root.findall('trip')):
                depart = float(trip.get('depart', 0))
                origin = trip.get('from')
                dest = trip.get('to')
                
                if origin and dest:
                    unique_origins.add(origin)
                    unique_destinations.add(dest)
                    
                    raw_trips.append({
                        'id': trip.get('id', str(i)),
                        'origin_edge': origin,
                        'dest_edge': dest,
                        'start_time': int(depart),
                        'kind': 'Human'
                    })

            # 2. Look for <vehicle> elements (if provided a routed file)
            for i, vehicle in enumerate(root.findall('vehicle')):
                depart = float(vehicle.get('depart', 0))
                route = vehicle.find('route')
                if route is not None:
                    edges = route.get('edges', '').strip().split()
                    if edges:
                        origin = edges[0]
                        dest = edges[-1]
                        
                        unique_origins.add(origin)
                        unique_destinations.add(dest)
                        
                        raw_trips.append({
                            'id': vehicle.get('id', str(i)),
                            'origin_edge': origin,
                            'dest_edge': dest,
                            'start_time': int(depart),
                            'kind': 'Human'
                        })
                        
        except Exception as e:
            logger.warning(f"Error parsing routes/trips file: {e}")
            
        # Convert sets to sorted lists for deterministic indexing
        sorted_origins = sorted(list(unique_origins))
        sorted_destinations = sorted(list(unique_destinations))
        
        # Build index maps
        origin_map = {edge: idx for idx, edge in enumerate(sorted_origins)}
        dest_map = {edge: idx for idx, edge in enumerate(sorted_destinations)}
        
        # Build final agents list with sequential integer ids to match
        # URB/RouteRL expectations that vehicle ids round-trip as ints.
        final_agents = []
        for agent_id, trip in enumerate(raw_trips):
            final_agents.append({
                'id': agent_id,
                'origin': origin_map[trip['origin_edge']], # Map to index
                'destination': dest_map[trip['dest_edge']], # Map to index
                'start_time': trip['start_time'],
                'kind': trip['kind']
            })

        od_pairs = sorted(
            {
                (trip["origin_edge"], trip["dest_edge"])
                for trip in raw_trips
            }
        )

        return final_agents, sorted_origins, sorted_destinations, od_pairs

    def _validate_od_pairs(
        self,
        network_file: Path,
        od_pairs: List[Tuple[str, str]],
        min_connection_paths: int,
    ) -> None:
        """Assert that exported OD pairs satisfy the configured path-count requirement."""
        if not od_pairs:
            logger.warning("URB export parsed no trips from %s", network_file)
            return

        network = SUMONetwork(network_file)
        demand_gen = DemandGenerator(network, seed=0)
        invalid_pairs = [
            (origin, destination)
            for origin, destination in od_pairs
            if not demand_gen.has_at_least_k_paths(origin, destination, min_connection_paths)
        ]
        if not invalid_pairs:
            return

        preview = ", ".join(f"{origin}->{destination}" for origin, destination in invalid_pairs[:5])
        raise ValueError(
            "URB export validation failed: "
            f"{len(invalid_pairs)} OD pairs do not satisfy min_connection_paths="
            f"{min_connection_paths}. Examples: {preview}"
        )

    def _write_agents_csv(self, agents: List[Dict]):
        path = self.target_dir / "agents.csv"
        fieldnames = ['id', 'origin', 'destination', 'start_time', 'kind']
        
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(agents)

    def _write_od_txt(self, origins: List[str], destinations: List[str]):
        path = self.target_dir / f"od_{self.run_id}.txt"
        
        # Format exactly as requested: { "origins" : ['id1', ...], ... }
        # Note: Reference used single quotes inside the list representation
        
        def to_single_quoted_list(items):
            # manual formatting to match ['a', 'b'] with single quotes
            inner = ", ".join(f"'{item}'" for item in items)
            return f"[{inner}]"
            
        content = "{\n"
        content += f'"origins" : {to_single_quoted_list(origins)},\n'
        content += f'"destinations" : {to_single_quoted_list(destinations)}\n'
        content += "}"
        
        with open(path, 'w') as f:
            f.write(content)
