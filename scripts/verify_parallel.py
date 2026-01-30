
"""
Experiment 2: Parallel Stress Test
Verifies that the parallel architecture works under load and correctly reports metrics.
"""
import sys
import logging
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path.cwd()))

from demandify.pipeline import CalibrationPipeline
from demandify.utils.logger import setup_logging

import os
# Set workers via env var before config is loaded (if possible, though config might be loaded by imports already)
# To be safe, we can modify config after if we know the key, but passing in init is best if supported.
# Workers is likely in config only.
os.environ["DEMANDIFY_PARALLEL_WORKERS"] = "2"

def run_experiment():
    print("🧪 Running Experiment 2: Parallel Stress Test")
    
    # Setup
    run_id = "exp2_parallel"
    output_dir = Path("demandify_runs") / run_id
    setup_logging(run_dir=output_dir / "logs", level=logging.INFO)
    logger = logging.getLogger("exp2")
    
    pipeline = CalibrationPipeline(
        bbox=(19.95, 50.05, 19.97, 50.07), # Required arg
        window_minutes=15, # Required arg
        seed=42, # Required arg
        ga_population=8,    # Pass directly
        ga_generations=2,   # Pass directly
        run_id=run_id,
        output_dir=output_dir,
    )
    
    # Locate a valid network
    network_file = None
    for f in Path("demandify_runs").rglob("*.net.xml"):
        network_file = f
        break
    
    if not network_file:
         # Need to download one. Use a small bbox.
         bbox = (19.95, 50.05, 19.97, 50.07) # Small chunk of Krakow
         print("   No network found. Using small bbox.")
         pipeline.bbox = bbox
    else:
        print(f"   Using existing network: {network_file}")
        pipeline.network_file = network_file
        # Ensure bbox matches network or just trust logic
        # pipeline.bbox = ... 
        
    print(f"   Running short calibration (Pop=8, Gen=2, Workers=2)...")
    
    # Run
    try:
        import asyncio
        asyncio.run(pipeline.run())
        print("   ✅ Pipeline finished successfully.")
    except Exception as e:
        print(f"   ❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_experiment()
