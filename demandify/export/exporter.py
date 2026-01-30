"""
Scenario export and project folder generation.
"""
from pathlib import Path
from typing import Dict
import shutil
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ScenarioExporter:
    """Export calibrated scenario to a project folder."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize exporter.
        
        Args:
            output_dir: Directory to export scenario to
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export(
        self,
        network_file: Path,
        demand_csv: Path,
        trips_file: Path,
        # routes_file removed - using dynamic routing
        observed_edges_csv: Path,
        run_metadata: Dict
    ) -> Path:
        """
        Export complete scenario for dynamic SUMO routing.
        
        Args:
            network_file: SUMO network .net.xml
            demand_csv: demand.csv file
            trips_file: trips.xml file (SUMO will route dynamically)
            observed_edges_csv: observed_edges.csv
            run_metadata: Metadata dictionary
        
        Returns:
            Path to output directory
        """
        logger.info(f"Exporting scenario to {self.output_dir}")
        
        # Helper to handle file placement
        def ensure_in_project(src: Path, name: str) -> Path:
            src = src.resolve()
            dest_root = self.output_dir.resolve()
            
            # Check if source is already inside destination directory
            # We use string comparison of absolute paths for robustness
            if str(src).startswith(str(dest_root)):
                return src
            
            # Else copy to root (fallback behavior)
            dst = self.output_dir / name
            if src != dst:
                shutil.copy2(src, dst)
            return dst
        
        # Ensure files are tracked (don't copy if already in data/ or sumo/)
        final_net = ensure_in_project(network_file, "network.net.xml")
        final_demand = ensure_in_project(demand_csv, "demand.csv")
        final_trips = ensure_in_project(trips_file, "trips.xml")
        final_edges = ensure_in_project(observed_edges_csv, "observed_edges.csv")
        
        # Determine location for sumocfg
        # We prefer to put it alongside the network file (usually in sumo/)
        sumocfg_path = final_net.parent / "scenario.sumocfg"
        
        # Get seed from metadata for deterministic routing
        seed = run_metadata.get('run_info', {}).get('seed', 42)
        
        # Generate sumocfg with dynamic routing configuration
        self._create_sumocfg(
            network_file=final_net,
            trips_file=final_trips,  # Pass trips.xml instead of routes
            output_file=sumocfg_path,
            simulation_time=run_metadata.get('simulation_config', {}).get('window_minutes', 15) * 60 + 
                          run_metadata.get('simulation_config', {}).get('warmup_minutes', 5) * 60,
            seed=seed
        )
        
        # Save metadata
        self._save_metadata(run_metadata)
        
        logger.info(f"Scenario exported successfully to {self.output_dir}")
        
        return self.output_dir
    
    def _create_sumocfg(
        self,
        network_file: Path,
        trips_file: Path,  # Changed from routes_file
        output_file: Path,
        simulation_time: int,
        seed: int = 42
    ):
        """Create SUMO configuration file for dynamic routing."""
        import xml.etree.ElementTree as ET
        import os
        
        # Calculate relative paths from sumocfg location
        try:
            rel_net = os.path.relpath(network_file, output_file.parent)
            rel_trips = os.path.relpath(trips_file, output_file.parent)
        except ValueError:
            # Fallback to absolute if on different drives (rare)
            rel_net = str(network_file)
            rel_trips = str(trips_file)
        
        root = ET.Element('configuration')
        
        # Input
        input_elem = ET.SubElement(root, 'input')
        ET.SubElement(input_elem, 'net-file').set('value', rel_net)
        ET.SubElement(input_elem, 'route-files').set('value', rel_trips)  # trips.xml instead of routes.rou.xml
        
        # Time
        time_elem = ET.SubElement(root, 'time')
        ET.SubElement(time_elem, 'begin').set('value', '0')
        ET.SubElement(time_elem, 'end').set('value', str(simulation_time))
        
        # Routing configuration for dynamic routing
        routing_elem = ET.SubElement(root, 'routing')
        ET.SubElement(routing_elem, 'device.rerouting.probability').set('value', '0')  # No dynamic rerouting
        ET.SubElement(routing_elem, 'routing-algorithm').set('value', 'dijkstra')
        
        # Seed for reproducibility
        if seed is not None:
            random_elem = ET.SubElement(root, 'random')
            ET.SubElement(random_elem, 'seed').set('value', str(seed))
        
        # Write
        tree = ET.ElementTree(root)
        ET.indent(tree, space='  ')
        tree.write(output_file, encoding='utf-8', xml_declaration=True)
        
        logger.info(f"Created scenario.sumocfg with dynamic routing (seed={seed}): {output_file}")
    
    def _save_metadata(self, metadata: Dict):
        """Save run metadata as JSON."""
        meta_file = self.output_dir / "run_meta.json"
        
        # Ensure datetime objects are serializable
        serializable_meta = self._make_serializable(metadata)
        
        with open(meta_file, 'w') as f:
            json.dump(serializable_meta, f, indent=2)
        
        logger.info(f"Saved metadata: {meta_file}")
    
    def _make_serializable(self, obj):
        """Make object JSON serializable."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
