import csv
from heapq import heappop, heappush
import shutil
import logging
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
            demand_routes.csv   # Per-agent route choice set (agent_id, source_id, route_id, probability, free_flow_time, edges)
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
        
    def export(
        self,
        network_file: Path,
        routes_file: Path,
        min_connection_paths: int = 1,
        vehroutes_file: Optional[Path] = None,
    ):
        """Main export workflow."""
        logger.debug(f"Starting URB data export to {self.target_dir}")
        
        try:
            min_connection_paths = int(min_connection_paths)
            if min_connection_paths < 1:
                raise ValueError("min_connection_paths must be at least 1")

            if vehroutes_file is None:
                candidate_vr = Path(routes_file).parent / "vehroutes.xml"
                if candidate_vr.exists():
                    vehroutes_file = candidate_vr

            # 1. Parse trips/routes and validate compatibility before writing files
            agents, origins, destinations, od_pairs, raw_trips = self._parse_routes(routes_file)
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

            # 6. Write demand_routes.csv
            self._write_demand_routes_csv(network_file, raw_trips, vehroutes_file)
            
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
    ) -> Tuple[List[Dict], List[str], List[str], List[Tuple[str, str]], List[Dict]]:
        """
        Parse .rou.xml to extract trips.
        Returns:
            agents: List of dicts
            origins: List of unique origin edge IDs
            destinations: List of unique destination edge IDs
            od_pairs: Sorted unique demanded OD edge pairs
            raw_trips: List of raw trip dicts preserving source_id and optional route edges
        """
        unique_origins = set()
        unique_destinations = set()
        raw_trips = []
        
        try:
            tree = ET.parse(routes_file)
            root = tree.getroot()
            
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
                        'kind': 'Human',
                        'edges': None,
                    })

            # 2. Look for <vehicle> elements (if provided a routed file)
            for i, vehicle in enumerate(root.findall('vehicle')):
                depart = float(vehicle.get('depart', 0))
                route = vehicle.find('route')
                if route is None:
                    route_dist = vehicle.find('routeDistribution')
                    if route_dist is not None:
                        dist_routes = route_dist.findall('route')
                        if dist_routes:
                            route = dist_routes[-1]
                if route is not None:
                    edges = [
                        e for e in route.get('edges', '').strip().split()
                        if not e.startswith(':')
                    ]
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
                            'kind': 'Human',
                            'edges': edges,
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

        return final_agents, sorted_origins, sorted_destinations, od_pairs, raw_trips

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

    @staticmethod
    def _extract_edge_free_flow_times(network_file: Path) -> Dict[str, float]:
        """Compute per-edge free-flow traversal time (length / speed in seconds) from .net.xml."""
        edge_fft: Dict[str, float] = {}
        try:
            tree = ET.parse(network_file)
            root = tree.getroot()
            for edge in root.findall("edge"):
                edge_id = edge.get("id")
                if not edge_id or edge_id.startswith(":"):
                    continue
                lane = edge.find("lane")
                if lane is None:
                    continue
                length = float(lane.get("length", 0.0))
                speed = max(float(lane.get("speed", 13.89)), 0.1)
                edge_fft[edge_id] = length / speed
        except Exception as e:
            logger.warning(f"Failed to extract edge free-flow times from {network_file}: {e}")
        return edge_fft

    @staticmethod
    def _parse_vehroutes(vehroutes_file: Optional[Path]) -> Dict[str, List[str]]:
        """Parse SUMO vehroute-output XML into a map of vehicle_id -> edge sequence."""
        if vehroutes_file is None or not Path(vehroutes_file).exists():
            return {}
        sim_routes: Dict[str, List[str]] = {}
        try:
            tree = ET.parse(vehroutes_file)
            root = tree.getroot()
            for vehicle in root.findall("vehicle"):
                vid = vehicle.get("id")
                if not vid:
                    continue
                route_elem = vehicle.find("route")
                if route_elem is None:
                    route_dist = vehicle.find("routeDistribution")
                    if route_dist is not None:
                        dist_routes = route_dist.findall("route")
                        if dist_routes:
                            route_elem = dist_routes[-1]
                if route_elem is None:
                    continue
                edges = [
                    e for e in route_elem.get("edges", "").strip().split()
                    if not e.startswith(":")
                ]
                if edges:
                    sim_routes[vid] = edges
        except Exception as e:
            logger.warning(f"Failed to parse vehroutes file {vehroutes_file}: {e}")
        return sim_routes

    @staticmethod
    def _find_shortest_fft_path(
        adjacency: Dict[str, Tuple[str, ...]],
        edge_fft: Dict[str, float],
        origin: str,
        destination: str,
    ) -> List[str]:
        """Find minimum free-flow travel-time path from origin to destination using Dijkstra."""
        if origin == destination:
            return [origin]

        start_cost = float(edge_fft.get(origin, 1.0))
        best_cost: Dict[str, float] = {origin: start_cost}
        predecessor: Dict[str, Optional[str]] = {origin: None}
        heap: List[Tuple[float, str]] = [(start_cost, origin)]

        while heap:
            cost, current = heappop(heap)
            if current == destination:
                break
            if cost > best_cost.get(current, float("inf")):
                continue
            for neighbor in adjacency.get(current, ()):
                step_cost = float(edge_fft.get(neighbor, 1.0))
                new_cost = cost + step_cost
                if new_cost < best_cost.get(neighbor, float("inf")):
                    best_cost[neighbor] = new_cost
                    predecessor[neighbor] = current
                    heappush(heap, (new_cost, neighbor))

        if destination not in predecessor:
            return []

        path: List[str] = []
        curr: Optional[str] = destination
        while curr is not None:
            path.append(curr)
            curr = predecessor[curr]
        path.reverse()
        return path

    def _write_demand_routes_csv(
        self,
        network_file: Path,
        raw_trips: List[Dict],
        vehroutes_file: Optional[Path] = None,
    ) -> None:
        """Write demand_routes.csv associating each agent with its winning route."""
        path = self.target_dir / "demand_routes.csv"
        fieldnames = [
            "agent_id",
            "source_id",
            "route_id",
            "probability",
            "free_flow_time",
            "edges",
        ]

        edge_fft = self._extract_edge_free_flow_times(network_file)
        sim_routes = self._parse_vehroutes(vehroutes_file)
        network = SUMONetwork(network_file)
        adjacency = {
            edge_id: tuple(sorted(neighbors))
            for edge_id, neighbors in network.adjacency.items()
        }
        fallback_cache: Dict[Tuple[str, str], List[str]] = {}

        rows = []
        for agent_id, trip in enumerate(raw_trips):
            source_id = str(trip["id"])
            origin = str(trip["origin_edge"])
            dest = str(trip["dest_edge"])

            route_edges = sim_routes.get(source_id) or trip.get("edges")
            if not route_edges or route_edges[0] != origin or route_edges[-1] != dest:
                cache_key = (origin, dest)
                if cache_key not in fallback_cache:
                    fallback_path = self._find_shortest_fft_path(
                        adjacency, edge_fft, origin, dest
                    )
                    if not fallback_path:
                        fallback_path = [origin, dest] if origin != dest else [origin]
                    fallback_cache[cache_key] = fallback_path
                route_edges = fallback_cache[cache_key]

            fft = sum(float(edge_fft.get(edge_id, 0.0)) for edge_id in route_edges)
            rows.append({
                "agent_id": agent_id,
                "source_id": source_id,
                "route_id": 0,
                "probability": 1.0,
                "free_flow_time": fft,
                "edges": " ".join(route_edges),
            })

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

