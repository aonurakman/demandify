"""
Worker module for parallel simulation execution.
Functionality is isolated here to avoid pickling complications with multiprocessing.
"""
import logging
import time
import shutil
import math
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

from demandify.sumo.simulation import SUMOSimulation
from demandify.calibration.objective import EdgeSpeedObjective

# Setup logger for the worker process
# Note: multiprocessing workers inherit logger configuration on Fork (Linux/Mac)
# But on Spawn (Windows/Mac sometimes) they need setup.
# We rely on the parent process setting up logging before pool creation or basicConfig.
logger = logging.getLogger(__name__)

@dataclass
class SimulationConfig:
    """Configuration for a single simulation run."""
    run_id: str
    network_file: Path
    od_pairs: List[Tuple[str, str]]
    departure_bins: List[Tuple[int, int]]
    observed_edges: pd.DataFrame
    warmup_time: int
    simulation_time: int
    step_length: float = 1.0
    debug: bool = False
    
    # Paths
    output_base_dir: Path = Path("temp_sims")


def generate_demand_files(
    genome: np.ndarray,
    od_pairs: List[Tuple[str, str]],
    departure_bins: List[Tuple[int, int]],
    seed: int,
    output_dir: Path
) -> Path:
    """
    Generate demand files (trips.xml) from genome.
    Re-implements DemandGenerator logic locally to avoid picking the heavy class.
    
    Returns:
        Path to trips.xml
    """
    demand_csv = output_dir / "demand.csv"
    trips_file = output_dir / "trips.xml"
    
    num_od = len(od_pairs)
    num_bins = len(departure_bins)
    
    # Reshape genome
    counts = genome.reshape(num_od, num_bins)
    
    trips = []
    trip_id = 0
    
    # Generate trips
    # Vectorized approach or fast loop
    for od_idx, (origin, dest) in enumerate(od_pairs):
        for bin_idx, (start_time, end_time) in enumerate(departure_bins):
            # Ensure non-negative integer count
            count = int(max(0, round(counts[od_idx, bin_idx])))
            
            if count > 0:
                # Seeded random jitter for this specific bin/OD combo
                # Unique seed based on input seed + identifiers
                # We use a simple hash combination to be deterministic but varied
                local_seed = (seed + od_idx * 1000 + bin_idx * 100000) % (2**32)
                bin_rng = np.random.RandomState(local_seed)
                
                departure_times = bin_rng.uniform(start_time, end_time, size=count)
                
                for dep_time in departure_times:
                    trips.append({
                        'ID': f't_{od_idx}_{bin_idx}_{trip_id}',
                        'depart': f"{dep_time:.2f}",
                        'from': origin,
                        'to': dest
                    })
                    trip_id += 1
    
    # Create XML directly (faster than CSV -> XML)
    root = ET.Element('routes')
    for trip in trips:
        t = ET.SubElement(root, 'trip')
        t.set('id', trip['ID'])
        t.set('depart', trip['depart'])
        t.set('from', trip['from'])
        t.set('to', trip['to'])
        
    tree = ET.ElementTree(root)
    ET.indent(tree, space='  ')
    tree.write(trips_file, encoding='utf-8', xml_declaration=True)
    
    return trips_file


def evaluate_for_ga(
    genome: np.ndarray,
    config: SimulationConfig
) -> Tuple[float, Dict[str, Any]]:
    """
    Wrapper for GA evaluation.
    Returns (loss, metrics).
    """
    return run_simulation_worker(genome, config)


def run_simulation_worker(
    genome: np.ndarray,
    config: SimulationConfig,
    worker_idx: Optional[int] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Worker function to run a single simulation evaluation.
    Designed to be used with functools.partial(run_simulation_worker, config=...).
    
    Args:
        genome: The genome (vehicle counts)
        config: Simulation configuration
        worker_idx: Optional worker ID. If None, derived from process ID.
              
    Returns:
        (loss, metrics_dict)
    """
    import os
    if worker_idx is None:
        worker_idx = os.getpid() % 100
    
    # Unique temp directory for this worker/eval
    timestamp = int(time.time() * 1000)
    temp_dir = config.output_base_dir / f"w{worker_idx}_{timestamp}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Generate Demand
        seed = int(np.sum(genome)) + worker_idx # Simple seed variation based on genome content
        trips_file = generate_demand_files(
            genome, 
            config.od_pairs, 
            config.departure_bins, 
            seed, 
            temp_dir
        )
        
        # 2. Run Simulation
        sim = SUMOSimulation(
            network_file=config.network_file,
            vehicle_file=trips_file,
            step_length=config.step_length,
            warmup_time=config.warmup_time,
            simulation_time=config.simulation_time,
            seed=42, # Deterministic routing inside SUMO
            use_dynamic_routing=True
        )
        
        expected_vehicles = int(np.sum(genome))
        
        # We pass output_dir=temp_dir so SUMOSimulation generates artifacts there
        # and doesn't delete them immediately if we want to debug
        # We pass output_dir=temp_dir so SUMOSimulation generates artifacts there
        # and doesn't delete them immediately if we want to debug
        simulated_speeds, trip_stats = sim.run(
            output_dir=temp_dir,
            expected_vehicles=expected_vehicles
        )
        
        routing_failures = trip_stats.get('routing_failures', 0)

        
        # 3. Calculate Objective
        objective = EdgeSpeedObjective(config.observed_edges)
        loss = objective.calculate_loss(
            simulated_speeds,
            trip_stats=trip_stats,
            expected_vehicles=expected_vehicles
        )
        
        # 4. Metrics
        metrics = objective.calculate_metrics(simulated_speeds)
        metrics['routing_failures'] = routing_failures
        metrics['total_vehicles'] = expected_vehicles
        metrics['zero_flow_edges'] = metrics['missing_edges'] # Same thing essentially
        metrics['avg_trip_duration'] = trip_stats.get('avg_duration', 0.0)
        metrics['avg_waiting_time'] = trip_stats.get('avg_waiting_time', 0.0)
        metrics['worker_id'] = worker_idx
        metrics['loss'] = loss # Explicitly include loss in metrics for aggregation if needed
        
        # 5. Debug Artifacts
        if config.debug:
            # Preserve artifacts: move temp_dir to debug storage?
            # For now, just DON'T delete it.
            # But we should probably rename it to include generation info if possible.
            # Since worker doesn't know generation, we rely on timestamp.
            # We can log the location.
            pass
        else:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        return loss, metrics
        
    except Exception as e:
        logger.error(f"Worker {worker_idx} failed: {e}")
        # Clean up even on fail
        shutil.rmtree(temp_dir, ignore_errors=True)
        # Return infinite loss
        return float('inf'), {"error": str(e)}
