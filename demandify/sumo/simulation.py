"""
SUMO simulation execution and edge statistics extraction.
"""
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict
import xml.etree.ElementTree as ET
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class SUMOSimulation:
    """Run SUMO simulations and extract edge statistics."""
    
    def __init__(
        self,
        network_file: Path,
        routes_file: Path,
        step_length: float = 1.0,
        warmup_time: int = 300,  # 5 minutes
        simulation_time: int = 900  # 15 minutes
    ):
        """
        Initialize SUMO simulation.
        
        Args:
            network_file: Path to .net.xml
            routes_file: Path to .rou.xml
            step_length: Simulation step length in seconds
            warmup_time: Warmup period in seconds
            simulation_time: Total simulation time in seconds
        """
        self.network_file = network_file
        self.routes_file = routes_file
        self.step_length = step_length
        self.warmup_time = warmup_time
        self.simulation_time = simulation_time
    
    def run(
        self,
        output_dir: Path = None,
        edge_data_file: Path = None
    ) -> Dict[str, float]:
        """
        Run SUMO simulation and extract edge statistics.
        
        Args:
            output_dir: Directory for simulation outputs (temp if None)
            edge_data_file: Output file for edge statistics XML
        
        Returns:
            Dict mapping edge_id -> mean_speed
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
            
            self._create_config(config_file, edge_output)
            
            # Run SUMO
            logger.debug("Running SUMO simulation")
            
            cmd = [
                "sumo",  # Use sumo (no GUI)
                "-c", str(config_file),
                "--no-warnings",
                "--no-step-log",
                "--duration-log.disable"
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
            
            # Copy edge data if requested
            if edge_data_file:
                shutil.copy(edge_output, edge_data_file)
            
            return edge_stats
            
        except subprocess.CalledProcessError as e:
            logger.error(f"SUMO simulation failed: {e.stderr}")
            raise RuntimeError(f"SUMO simulation failed: {e.stderr}")
        
        finally:
            if cleanup:
                shutil.rmtree(output_dir, ignore_errors=True)
    
    def _create_config(self, config_file: Path, edge_output: Path):
        """Create SUMO configuration file and additional edge-data config."""
        output_dir = config_file.parent
        additional_file = output_dir / "additional.xml"

        # 1. Create additional.xml with edgeData configuration
        # This is where 'freq' attribute is actually respected
        add_root = ET.Element('additional')
        ET.SubElement(add_root, 'edgeData', {
            'id': 'edge_data_0',
            'file': str(edge_output),
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
        ET.SubElement(input_elem, 'net-file').set('value', str(self.network_file))
        ET.SubElement(input_elem, 'route-files').set('value', str(self.routes_file))
        ET.SubElement(input_elem, 'additional-files').set('value', str(additional_file))
        
        # Time
        time_elem = ET.SubElement(root, 'time')
        ET.SubElement(time_elem, 'begin').set('value', '0')
        ET.SubElement(time_elem, 'end').set('value', str(self.simulation_time))
        ET.SubElement(time_elem, 'step-length').set('value', str(self.step_length))
        
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
        
        tree = ET.parse(edge_data_file)
        root = tree.getroot()
        
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
