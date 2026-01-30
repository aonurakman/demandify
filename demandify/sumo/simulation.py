"""
SUMO simulation execution and edge statistics extraction.
"""
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Tuple
import xml.etree.ElementTree as ET
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class SUMOSimulation:
    """Run SUMO simulations and extract edge statistics."""
    
    def __init__(
        self,
        network_file: Path,
        vehicle_file: Path,  # Can be trips.xml or routes.rou.xml
        step_length: float = 1.0,
        warmup_time: int = 300,  # 5 minutes
        simulation_time: int = 900,  # 15 minutes
        seed: int = None,  # For deterministic routing
        use_dynamic_routing: bool = True,  # If True, vehicle_file is trips.xml
        debug: bool = False
    ):
        """
        Initialize SUMO simulation.
        
        Args:
            network_file: Path to .net.xml
            vehicle_file: Path to trips.xml (dynamic) or routes.rou.xml (precomputed)
            step_length: Simulation step length in seconds
            warmup_time: Warmup period in seconds
            simulation_time: Total simulation time in seconds
            seed: Random seed for deterministic routing (required for dynamic routing)
            use_dynamic_routing: If True, SUMO will route trips dynamically
            debug: If True, preserve intermediate files (tripinfo, etc.)
        """
        self.network_file = network_file
        self.vehicle_file = vehicle_file
        self.step_length = step_length
        self.warmup_time = warmup_time
        self.simulation_time = simulation_time
        self.seed = seed
        self.use_dynamic_routing = use_dynamic_routing
        self.debug = debug
        
        if use_dynamic_routing and seed is None:
            logger.warning("Dynamic routing enabled but no seed provided - results may not be reproducible")
    
    def run(
        self,
        output_dir: Path = None,
        edge_data_file: Path = None,
        expected_vehicles: int = None
    ) -> Tuple[Dict[str, float], int]:
        """
        Run SUMO simulation and extract edge statistics.
        
        Args:
            output_dir: Directory for simulation outputs (temp if None)
            edge_data_file: Output file for edge statistics XML
            expected_vehicles: Number of vehicles in trips.xml (for failure tracking)
        
        Returns:
            Tuple of (edge_stats, trip_stats)
            - edge_stats: Dict mapping edge_id -> mean_speed
            - trip_stats: Dict with keys 'routing_failures', 'total_trips', 'avg_duration', 'avg_waiting_time'
        """
        # Create temp directory if needed
        if output_dir is None:
            temp_dir = tempfile.mkdtemp(prefix='sumo_sim_')
            output_dir = Path(temp_dir)
            cleanup = True
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            cleanup = False
        
        try:
            # Create config file
            config_file = output_dir / "simulation.sumocfg"
            edge_output = output_dir / "edge_data.xml"
            tripinfo_output = output_dir / "tripinfo.xml"
            statistic_output = output_dir / "statistics.xml"
            
            self._create_config(config_file, edge_output, tripinfo_output, statistic_output)
            
            # Run SUMO
            logger.debug("Running SUMO simulation")
            
            cmd = [
                "sumo",  # Use sumo (no GUI)
                "-c", str(config_file),
                "--no-warnings",
                "--no-step-log",
                "--duration-log.disable",
                "--ignore-route-errors"  # Skip vehicles with no valid route
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.debug("SUMO simulation completed")
            
            # Parse edge statistics
            edge_stats = self._parse_edge_data(edge_output)
            
            # Parse trip statistics from tripinfo
            trip_stats = {
                'routing_failures': 0,
                'teleports': 0,
                'total_trips': 0,
                'avg_duration': 0.0,
                'avg_waiting_time': 0.0
            }
            
            if tripinfo_output.exists():
                trip_stats = self._parse_trip_stats(tripinfo_output, expected_vehicles)
            
            # Parse global statistics if available
            if statistic_output.exists():
                global_stats = self._parse_statistic_output(statistic_output)
                # Update failures:
                # True Failures = Loaded (Intent) - Inserted (Success)
                # Note: "Waiting" vehicles are also failures for this specific simulation window
                if global_stats['loaded'] > 0:
                    loaded = global_stats['loaded']
                    inserted = global_stats['inserted']
                    trip_stats['routing_failures'] = loaded - inserted
                    trip_stats['teleports'] = global_stats.get('teleports', 0)
                    trip_stats['total_trips'] = loaded
                    
                    # Log if there's a discrepancy
                    if trip_stats['routing_failures'] > 0:
                         logger.debug(f"Insertion Backlog: {trip_stats['routing_failures']} vehicles failed to enter")
                    if trip_stats['teleports'] > 0:
                         logger.debug(f"Teleportations: {trip_stats['teleports']} vehicles teleported")

            # Copy edge data if requested
            if edge_data_file:
                shutil.copy(edge_output, edge_data_file)
            
            # Clean up tripinfo/stats to prevent bloat (unless debug)
            if not self.debug:
                if tripinfo_output.exists(): tripinfo_output.unlink()
                if statistic_output.exists(): statistic_output.unlink()
            
            return edge_stats, trip_stats
            
        except subprocess.CalledProcessError as e:
            logger.error(f"SUMO simulation failed: {e.stderr}")
            raise RuntimeError(f"SUMO simulation failed: {e.stderr}")
        
        finally:
            if cleanup:
                shutil.rmtree(output_dir, ignore_errors=True)
    
    def _create_config(self, config_file: Path, edge_output: Path, tripinfo_output: Path, statistic_output: Path):
        """Create SUMO configuration file with tripinfo for failure tracking."""
        output_dir = config_file.parent.resolve()
        additional_file = output_dir / "additional.xml"

        # Resolve paths first to handle symlinks/relative inputs
        edge_output = edge_output.resolve()
        tripinfo_output = tripinfo_output.resolve()
        statistic_output = statistic_output.resolve()
        network_path = self.network_file.resolve() if self.network_file else None
        vehicle_path = self.vehicle_file.resolve() if self.vehicle_file else None
        
        # Helper to make paths relative to output_dir (for portability)
        def make_relative(path: Path) -> str:
            try:
                return str(path.relative_to(output_dir))
            except ValueError:
                # Fallback: try os.path.relpath which handles ".." (up-level)
                import os
                try:
                    return os.path.relpath(str(path), str(output_dir))
                except ValueError:
                    # Fallback to absolute if on different drives etc.
                    return str(path)

        # 1. Create additional.xml
        add_root = ET.Element('additional')
        ET.SubElement(add_root, 'edgeData', {
            'id': 'edge_data_0',
            'file': make_relative(edge_output),
            'freq': '60',            # Output every 60 seconds
            'excludeEmpty': 'false'  # Include edges with no traffic
        })
        
        add_tree = ET.ElementTree(add_root)
        ET.indent(add_tree, space='  ')
        add_tree.write(additional_file, encoding='utf-8', xml_declaration=True)

        # 2. Create main sumocfg
        root = ET.Element('configuration')
        
        # Input
        input_elem = ET.SubElement(root, 'input')
        ET.SubElement(input_elem, 'net-file').set('value', make_relative(network_path))
        ET.SubElement(input_elem, 'route-files').set('value', make_relative(vehicle_path))
        ET.SubElement(input_elem, 'additional-files').set('value', make_relative(additional_file))
        
        # Time
        time_elem = ET.SubElement(root, 'time')
        ET.SubElement(time_elem, 'begin').set('value', '0')
        ET.SubElement(time_elem, 'end').set('value', str(self.simulation_time))
        ET.SubElement(time_elem, 'step-length').set('value', str(self.step_length))
        
        # Routing configuration (for dynamic routing)
        if self.use_dynamic_routing:
            routing_elem = ET.SubElement(root, 'routing')
            # Disable dynamic rerouting during simulation (use initial routes only)
            ET.SubElement(routing_elem, 'device.rerouting.probability').set('value', '0')
            # Use Dijkstra algorithm for initial routing
            ET.SubElement(routing_elem, 'routing-algorithm').set('value', 'dijkstra')
        
        # Random (seed for deterministic behavior)
        if self.seed is not None:
            random_elem = ET.SubElement(root, 'random')
            ET.SubElement(random_elem, 'seed').set('value', str(self.seed))
        
        # Output (tripinfo for routing failure tracking)
        output_elem = ET.SubElement(root, 'output')
        ET.SubElement(output_elem, 'tripinfo-output').set('value', make_relative(tripinfo_output))
        ET.SubElement(output_elem, 'statistic-output').set('value', make_relative(statistic_output))
        
        # Write config
        tree = ET.ElementTree(root)
        ET.indent(tree, space='  ')
        tree.write(config_file, encoding='utf-8', xml_declaration=True)
    
    def _parse_edge_data(self, edge_data_file: Path) -> Dict[str, float]:
        """
        Parse SUMO edge data output to extract mean speeds.
        
        Returns:
            Dict mapping edge_id -> mean_speed (km/h)
        """
        logger.debug(f"Parsing edge data: {edge_data_file}")
        
        edge_speeds = {}
        total_intervals = 0
        warmup_intervals = 0
        measurement_intervals = 0
        edges_in_warmup = set()
        edges_in_measurement = set()
        
        try:
            tree = ET.parse(edge_data_file)
            root = tree.getroot()
        except ET.ParseError as e:
            logger.error(f"Failed to parse edge data file {edge_data_file}: {e}")
            logger.warning("Returning empty edge stats due to XML parsing error. This may happen if SUMO crashed or was killed.")
            return {}
        
        # Edge data is in intervals
        for interval in root.findall('interval'):
            total_intervals += 1
            begin = float(interval.get('begin', 0))
            end = float(interval.get('end', 0))
            
            # Track edges in warmup
            if begin < self.warmup_time:
                warmup_intervals += 1
                for edge in interval.findall('edge'):
                    if edge.get('speed') and edge.get('speed') != '-1.00':
                        edges_in_warmup.add(edge.get('id'))
                continue
            
            # Measurement period
            measurement_intervals += 1
            for edge in interval.findall('edge'):
                edge_id = edge.get('id')
                speed = edge.get('speed')
                
                if speed and speed != '-1.00':  # -1 means no data
                    speed_kmh = float(speed) * 3.6  # Convert m/s to km/h
                    edges_in_measurement.add(edge_id)
                    
                    if edge_id not in edge_speeds:
                        edge_speeds[edge_id] = []
                    
                    edge_speeds[edge_id].append(speed_kmh)
        
        # Calculate mean speeds
        mean_speeds = {}
        for edge_id, speeds in edge_speeds.items():
            if speeds:
                mean_speeds[edge_id] = sum(speeds) / len(speeds)
        
        # Detailed logging
        # Detailed logging - change to DEBUG to avoid spam
        logger.debug(f"📊 Edge data summary:")
        logger.debug(f"  Total intervals: {total_intervals}")
        logger.debug(f"  Warmup intervals (t < {self.warmup_time}s): {warmup_intervals}")
        logger.debug(f"  Measurement intervals (t >= {self.warmup_time}s): {measurement_intervals}")
        logger.debug(f"  Edges with traffic during warmup: {len(edges_in_warmup)}")
        logger.debug(f"  Edges with traffic during measurement: {len(edges_in_measurement)}")
        logger.debug(f"  Extracted speeds for {len(mean_speeds)} edges")
        
        if len(mean_speeds) == 0 and len(edges_in_warmup) > 0:
            logger.warning(f"⚠️  DIAGNOSIS: {len(edges_in_warmup)} edges had traffic during warmup, ")
            logger.warning(f"   but 0 edges during measurement (t >= {self.warmup_time}s)")
            logger.warning(f"   → Vehicles likely complete trips before measurement starts!")
        
        return mean_speeds

    def _parse_trip_stats(self, tripinfo_file: Path, expected_vehicles: int = None) -> Dict:
        """
        Parse tripinfo.xml to extract detailed trip statistics.
        
        Args:
            tripinfo_file: Path to tripinfo.xml output
            expected_vehicles: Number of vehicles in trips.xml (for failure tracking)
            
        Returns:
            Dict with trip statistics
        """
        stats = {
            'routing_failures': 0,
            'total_trips': 0,
            'completed_trips': 0,
            'avg_duration': 0.0,
            'avg_waiting_time': 0.0
        }
        
        try:
            tree = ET.parse(tripinfo_file)
            root = tree.getroot()
            
            tripinfos = root.findall('tripinfo')
            completed_count = len(tripinfos)
            stats['completed_trips'] = completed_count
            
            # Calculate averages
            if completed_count > 0:
                durations = [float(t.get('duration', 0)) for t in tripinfos]
                waitings = [float(t.get('waitingTime', 0)) for t in tripinfos]
                stats['avg_duration'] = sum(durations) / completed_count
                stats['avg_waiting_time'] = sum(waitings) / completed_count
            
            # Routing failures
            if expected_vehicles is not None:
                # Failures = expected - successful
                stats['routing_failures'] = max(0, expected_vehicles - completed_count)
                stats['total_trips'] = expected_vehicles
            else:
                stats['total_trips'] = completed_count
            
            if stats['routing_failures'] > 0:
                logger.debug(f"Routing failures: {stats['routing_failures']}/{expected_vehicles} vehicles failed to route")
                
            return stats
            
        except Exception as e:
            logger.warning(f"Failed to parse tripinfo: {e}")
            return stats

    def _parse_statistic_output(self, stat_file: Path) -> Dict:
        """Parse SUMO statistics output for global counts."""
        stats = {'inserted': 0, 'loaded': 0, 'running': 0, 'waiting': 0, 'teleports': 0}
        try:
            tree = ET.parse(stat_file)
            root = tree.getroot()
            # <vehicles loaded="1500" inserted="1450" running="1300" waiting="50" teleports="0"/>
            veh = root.find('vehicles')
            if veh is not None:
                stats['loaded'] = int(veh.get('loaded', 0))
                stats['inserted'] = int(veh.get('inserted', 0))
                stats['running'] = int(veh.get('running', 0))
                stats['waiting'] = int(veh.get('waiting', 0))
                stats['teleports'] = int(veh.get('teleports', 0))
            return stats
        except Exception as e:
            logger.debug(f"Failed to parse statistics: {e}")
            return stats
