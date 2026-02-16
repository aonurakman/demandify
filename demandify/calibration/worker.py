"""
Worker module for parallel simulation execution.
Functionality is isolated here to avoid pickling complications with multiprocessing.
"""
import logging
import time
import shutil
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

from demandify.sumo.simulation import SUMOSimulation
from demandify.sumo.departure_schedule import (
    sequential_departure_times,
    format_departure_time,
)
from demandify.calibration.objective import EdgeSpeedObjective

# Setup logger for the worker process
# Note: multiprocessing workers inherit logger configuration on Fork (Linux/Mac)
# But on Spawn (Windows/Mac sometimes) they need setup.
# We rely on the parent process setting up logging before pool creation or basicConfig.
logger = logging.getLogger(__name__)
WORKER_FAILURE_FAIL_TOTAL_SENTINEL = 1


def build_worker_error_metrics(error_message: str, worker_idx: Optional[int] = None) -> Dict[str, Any]:
    """Build structured metrics for failed worker evaluations."""
    metrics = {
        "worker_error": True,
        "error": error_message,
        "routing_failures": WORKER_FAILURE_FAIL_TOTAL_SENTINEL,
        "teleports": 0,
        "fail_total": WORKER_FAILURE_FAIL_TOTAL_SENTINEL,
        "reliability_penalty": float("inf"),
        "e_loss": float("inf"),
        "loss": float("inf"),
        "zero_flow_edges": None,
        "total_vehicles": 0,
        "avg_trip_duration": 0.0,
        "avg_waiting_time": 0.0,
    }
    if worker_idx is not None:
        metrics["worker_id"] = worker_idx
    return metrics

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
    seed: int = 42
    
    # Paths
    output_base_dir: Path = Path("temp_sims")


def _stable_seed(genome: np.ndarray, base_seed: int) -> int:
    """Deterministic seed from genome + base seed."""
    digest = hashlib.sha256(genome.tobytes()).digest()
    seed_val = int.from_bytes(digest[:8], byteorder="little")
    return (seed_val ^ base_seed) % (2**32 - 1)


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
    trips_file = output_dir / "trips.xml"
    
    num_od = len(od_pairs)
    num_bins = len(departure_bins)
    
    # Reshape genome
    counts = genome.reshape(num_od, num_bins)
    
    trips = []
    trip_id = 0

    # Kept for API compatibility: demand timing is now deterministic and seed-independent.
    _ = seed
    
    # Generate trips
    for od_idx, (origin, dest) in enumerate(od_pairs):
        for bin_idx, (start_time, end_time) in enumerate(departure_bins):
            # Ensure non-negative integer count
            count = int(max(0, round(counts[od_idx, bin_idx])))
            
            if count > 0:
                departure_times = sequential_departure_times(start_time, end_time, count)

                for dep_time in departure_times:
                    trips.append({
                        'ID': f't_{od_idx}_{bin_idx}_{trip_id}',
                        'depart_value': float(dep_time),
                        'from': origin,
                        'to': dest
                    })
                    trip_id += 1

    # SUMO is more robust when trip departures are non-decreasing.
    trips.sort(key=lambda trip: trip['depart_value'])
    
    # Create XML directly (faster than CSV -> XML)
    root = ET.Element('routes')
    for trip in trips:
        t = ET.SubElement(root, 'trip')
        t.set('id', trip['ID'])
        t.set('depart', format_departure_time(trip['depart_value']))
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
        seed = _stable_seed(genome, config.seed)
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
            seed=seed, # Deterministic routing inside SUMO
            use_dynamic_routing=True
        )
        
        expected_vehicles = int(np.sum(genome))
        
        # We pass output_dir=temp_dir so SUMOSimulation generates artifacts there
        # and doesn't delete them immediately if we want to debug
        simulated_speeds, trip_stats = sim.run(
            output_dir=temp_dir,
            expected_vehicles=expected_vehicles
        )
        
        # 3. Calculate Objective
        objective = EdgeSpeedObjective(config.observed_edges)
        loss_components = objective.calculate_loss_components(
            simulated_speeds,
            trip_stats=trip_stats,
            expected_vehicles=expected_vehicles
        )
        loss = loss_components["loss"]

        routing_failures = int(trip_stats.get("routing_failures", 0) or 0)
        teleports = int(trip_stats.get("teleports", 0) or 0)
        fail_total = int(loss_components.get("fail_total", routing_failures + teleports))
        
        # 4. Metrics
        metrics = objective.calculate_metrics(simulated_speeds)
        metrics['routing_failures'] = routing_failures
        metrics['teleports'] = teleports
        metrics['fail_total'] = fail_total
        metrics['reliability_penalty'] = float(loss_components.get("reliability_penalty", 0.0))
        metrics['e_loss'] = float(loss_components.get("e_loss", metrics.get("mae", loss)))
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
        # Return infinite loss with explicit infeasible/error marker metrics
        return float("inf"), build_worker_error_metrics(str(e), worker_idx=worker_idx)
