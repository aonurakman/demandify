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
        routes_file: Path,
        observed_edges_csv: Path,
        run_metadata: Dict
    ) -> Path:
        """
        Export complete scenario.
        
        Args:
            network_file: SUMO network .net.xml
            demand_csv: demand.csv file
            trips_file: trips.xml file
            routes_file: routes.rou.xml file
            observed_edges_csv: observed_edges.csv
            run_metadata: Metadata dictionary
        
        Returns:
            Path to output directory
        """
        logger.info(f"Exporting scenario to {self.output_dir}")
        
        # Copy files to output directory (only if not already there)
        def safe_copy(src: Path, dst: Path):
            """Copy file only if source and destination are different."""
            src = Path(src).resolve()
            dst = Path(dst).resolve()
            if src != dst:
                shutil.copy(src, dst)
        
        safe_copy(network_file, self.output_dir / "network.net.xml")
        safe_copy(demand_csv, self.output_dir / "demand.csv")
        safe_copy(trips_file, self.output_dir / "trips.xml")
        safe_copy(routes_file, self.output_dir / "routes.rou.xml")
        safe_copy(observed_edges_csv, self.output_dir / "observed_edges.csv")
        
        # Generate sumocfg
        self._create_sumocfg(
            network_file=self.output_dir / "network.net.xml",
            routes_file=self.output_dir / "routes.rou.xml",
            output_file=self.output_dir / "scenario.sumocfg",
            simulation_time=run_metadata.get('simulation_config', {}).get('window_minutes', 15) * 60 + 
                          run_metadata.get('simulation_config', {}).get('warmup_minutes', 5) * 60
        )
        
        # Save metadata
        self._save_metadata(run_metadata)
        
        logger.info(f"Scenario exported successfully to {self.output_dir}")
        
        return self.output_dir
    
    def _create_sumocfg(
        self,
        network_file: Path,
        routes_file: Path,
        output_file: Path,
        simulation_time: int
    ):
        """Create SUMO configuration file."""
        import xml.etree.ElementTree as ET
        
        root = ET.Element('configuration')
        
        # Input
        input_elem = ET.SubElement(root, 'input')
        ET.SubElement(input_elem, 'net-file').set('value', network_file.name)
        ET.SubElement(input_elem, 'route-files').set('value', routes_file.name)
        
        # Time
        time_elem = ET.SubElement(root, 'time')
        ET.SubElement(time_elem, 'begin').set('value', '0')
        ET.SubElement(time_elem, 'end').set('value', str(simulation_time))
        
        # Write
        tree = ET.ElementTree(root)
        ET.indent(tree, space='  ')
        tree.write(output_file, encoding='utf-8', xml_declaration=True)
        
        logger.info(f"Created scenario.sumocfg: {output_file}")
    
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
