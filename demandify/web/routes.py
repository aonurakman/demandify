"""
Web routes for demandify with pipeline execution.
"""
from fastapi import APIRouter, Request, HTTPException, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from typing import Optional
import asyncio
import uuid
import logging
from datetime import datetime

from demandify.config import get_config, save_api_key

logger = logging.getLogger(__name__)

# Setup router and templates
router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))

# In-memory run storage (in production, use Redis or DB)
active_runs = {}


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main UI page."""
    config = get_config()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "config": config,
        "has_api_key": config.tomtom_api_key is not None
    })


@router.post("/api/config/api-key")
async def save_api_key_endpoint(service: str = Form(...), key: str = Form(...)):
    """Save an API key."""
    if service not in ["tomtom"]:
        raise HTTPException(status_code=400, detail="Invalid service")
    
    save_api_key(service, key)
    return {"status": "success", "message": "API key saved"}


async def run_calibration_pipeline(run_id: str, params: dict):
    """Background task to run calibration pipeline with progress updates."""
    from demandify.pipeline import CalibrationPipeline
    from demandify.calibration.objective import EdgeSpeedObjective
    
    try:
        # Update progress helper
        def update_progress(stage: int, stage_name: str, message: str, level: str = "info"):
            if run_id in active_runs:
                active_runs[run_id]["progress"]["stage"] = stage
                active_runs[run_id]["progress"]["stage_name"] = stage_name
                active_runs[run_id]["progress"]["logs"].append({
                    "message": message,
                    "level": level
                })
        
        update_progress(0, "Initializing", "Starting calibration pipeline...")
        
        # Create output directory
        output_dir = Path.cwd() / "demandify_runs" / f"run_{run_id[:8]}"
        
        # Store output_dir in active_runs for log file access
        active_runs[run_id]["output_dir"] = str(output_dir)
        
        # Initialize pipeline
        bbox = tuple(params["bbox"])
        pipeline = CalibrationPipeline(
            bbox=bbox,
            window_minutes=params["window_minutes"],
            seed=params["seed"],
            ga_population=params["ga_population"],
            ga_generations=params["ga_generations"],
            ga_mutation_rate=params["ga_mutation_rate"],
            ga_crossover_rate=params["ga_crossover_rate"],
            ga_elitism=params["ga_elitism"],
            ga_mutation_sigma=params["ga_mutation_sigma"],
            ga_mutation_indpb=params["ga_mutation_indpb"],
            output_dir=output_dir
        )
        
        # Stage 1: Fetch traffic data
        update_progress(1, "Fetching Traffic Data", "Connecting to TomTom API...")
        traffic_df = await pipeline._fetch_traffic_data()
        traffic_data_file = output_dir / "traffic_data_raw.csv"
        traffic_df.to_csv(traffic_data_file, index=False)
        update_progress(1, "Fetching Traffic Data", f"✓ Fetched {len(traffic_df)} traffic segments")
        
        # Stage 2: Fetch OSM
        update_progress(2, "Download OSM", "Downloading OpenStreetMap data...")
        osm_file = await pipeline._fetch_osm_data()
        update_progress(2, "Download OSM", "✓ OSM data downloaded")
        
        # Stage 3: Build SUMO network
        update_progress(3, "Building Network", "Converting map to SUMO network...")
        network_file = pipeline._build_sumo_network(osm_file)
        update_progress(3, "Building Network", "✓ SUMO network created")
        
        # Stage 4: Match traffic to edges
        update_progress(4, "Matching Traffic", "Matching traffic data to road network...")
        observed_edges = pipeline._match_traffic_to_edges(traffic_df, network_file)
        
        # ⚠️ CHECK FOR ZERO OBSERVED EDGES
        if len(observed_edges) == 0:
            update_progress(4, "No Observed Edges", 
                          "⚠️ WARNING: No traffic sensors in this area. Cannot calibrate demand without ground truth data.",
                          level="warning")
            active_runs[run_id]["status"] = "warning_no_edges"
            active_runs[run_id]["metadata"] = {
                "warning": "no_observed_edges",
                "message": "TomTom has no traffic data for this bounding box"
            }
            logger.warning(f"Run {run_id}: No observed edges - user intervention required")
            return
        
        observed_edges_file = output_dir / "observed_edges.csv"
        observed_edges.to_csv(observed_edges_file, index=False)
        update_progress(4, "Matching Traffic", f"✓ Matched {len(observed_edges)} road segments")
        
        # Stage 5: Initialize demand
        update_progress(5, "Init Demand", "Initializing demand generation...")
        demand_gen, od_pairs, departure_bins = pipeline._initialize_demand(network_file)
        update_progress(5, "Init Demand", f"✓ Created {len(od_pairs)} OD pairs")
        
        # Stage 6: Calibrate demand (longest stage)
        update_progress(6, "Calibrating Demand", 
                       f"Running genetic algorithm ({params['ga_generations']} generations)...")
        best_genome, best_loss, loss_history = pipeline._calibrate_demand(
            demand_gen, od_pairs, departure_bins, observed_edges, network_file
        )
        update_progress(6, "Calibrating Demand", f"✓ Complete: loss={best_loss:.2f} km/h")
        
        # Stage 7: Generate final demand files
        update_progress(7, "Generating Demand", "Creating final trip and route files...")
        demand_csv, trips_file, routes_file = pipeline._generate_final_demand(
            demand_gen, best_genome, od_pairs, departure_bins, network_file
        )
        update_progress(7, "Generating Demand", "✓ Files created")
        
        # Stage 8: Export scenario
        update_progress(8, "Exporting Results", "Running final simulation and generating reports...")
        simulated_speeds = pipeline._run_final_simulation(network_file, routes_file)
        
        # Calculate final metrics
        objective = EdgeSpeedObjective(observed_edges)
        quality_metrics = objective.calculate_metrics(simulated_speeds)
        
        # Export metadata
        metadata = pipeline._export_results(
            network_file, demand_csv, trips_file, routes_file,
            observed_edges_file, traffic_data_file, observed_edges,
            simulated_speeds, best_loss, loss_history, quality_metrics
        )
        
        # Update final status
        active_runs[run_id]["status"] = "completed"
        active_runs[run_id]["metadata"] = metadata
        active_runs[run_id]["output_dir"] = str(output_dir)
        update_progress(8, "Complete", f"Scenario exported to {output_dir}")
        
    except Exception as e:
        logger.error(f"Pipeline failed for run {run_id}: {e}", exc_info=True)
        if run_id in active_runs:
            active_runs[run_id]["status"] = "failed"
            active_runs[run_id]["progress"]["logs"].append({
                "message": f"Error: {str(e)}",
                "level": "error"
            })


@router.post("/api/run")
async def start_run(
    background_tasks: BackgroundTasks,
    bbox_west: float = Form(...),
    bbox_south: float = Form(...),
    bbox_east: float = Form(...),
    bbox_north: float = Form(...),
    window_minutes: int = Form(15),
    seed: int = Form(42),
    ga_population: int = Form(50),
    ga_generations: int = Form(20),
    ga_mutation_rate: float = Form(0.5),
    ga_crossover_rate: float = Form(0.7),
    ga_elitism: int = Form(2),
    ga_mutation_sigma: int = Form(20),
    ga_mutation_indpb: float = Form(0.3),
    parallel_workers: Optional[int] = Form(None)
):
    """Start a new calibration run."""
    from demandify.utils.validation import validate_bbox, calculate_bbox_area_km2
    
    # Validate bbox
    is_valid, msg = validate_bbox(bbox_west, bbox_south, bbox_east, bbox_north)
    if not is_valid:
        raise HTTPException(status_code=400, detail=msg)
    
    # Check area
    area_km2 = calculate_bbox_area_km2(bbox_west, bbox_south, bbox_east, bbox_north)
    config = get_config()
    if area_km2 > config.max_bbox_area_km2:
        logger.warning(f"Large bbox area: {area_km2:.2f} km²")
    
    # Check API key
    if not config.tomtom_api_key:
        raise HTTPException(status_code=400, detail="TomTom API key not configured")
    
    # Create run ID
    run_id = str(uuid.uuid4())
    
    # Store run metadata
    active_runs[run_id] = {
        "status": "running",
        "progress": {
            "stage": 0,
            "stage_name": "Initializing",
            "logs": []
        },
        "created_at": datetime.now().isoformat()
    }
    
    # Parameters
    params = {
        "bbox": [bbox_west, bbox_south, bbox_east, bbox_north],
        "window_minutes": window_minutes,
        "seed": seed,
        "ga_population": ga_population,
        "ga_generations": ga_generations,
        "ga_mutation_rate": ga_mutation_rate,
        "ga_crossover_rate": ga_crossover_rate,
        "ga_elitism": ga_elitism,
        "ga_mutation_sigma": ga_mutation_sigma,
        "ga_mutation_indpb": ga_mutation_indpb
    }
    
    # Start background task
    background_tasks.add_task(run_calibration_pipeline, run_id, params)
    
    return {"run_id": run_id, "status": "started"}


@router.get("/api/run/{run_id}/progress")
async def get_progress(run_id: str):
    """Get progress for a run."""
    if run_id not in active_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    
    # QUICK FIX: Also read from log file if available
    output_dir = active_runs[run_id].get("output_dir")
    if output_dir:
        log_file = Path(output_dir) / "pipeline.log"
        if log_file.exists():
            try:
                # Read last 30 lines
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    recent_lines = lines[-30:] if len(lines) > 30 else lines
                    
                # Parse and add to logs
                for line in recent_lines:
                    if ' - INFO - ' in line or ' - WARNING - ' in line:
                        msg = line.split(' - ', 3)[-1].strip()
                        if msg and msg not in [l["message"] for l in active_runs[run_id]["progress"]["logs"][-10:]]:
                            level = "warning" if "WARNING" in line else "info"
                            active_runs[run_id]["progress"]["logs"].append({
                                "message": msg,
                                "level": level
                            })
            except Exception as e:
                logger.error(f"Error reading log file: {e}")
    
    return active_runs[run_id]["progress"]


@router.get("/api/run/{run_id}/status")
async def get_status(run_id: str):
    """Get full status for a run."""
    if run_id not in active_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    
    return active_runs[run_id]


@router.get("/results", response_class=HTMLResponse)
async def results_page(request: Request, run_id: str):
    """Results page for a completed run."""
    if run_id not in active_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    
    return templates.TemplateResponse("results.html", {
        "request": request,
        "run_id": run_id,
        "run_data": active_runs[run_id]
    })
