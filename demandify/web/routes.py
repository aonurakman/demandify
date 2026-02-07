"""
Web routes for demandify with pipeline execution.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from demandify.config import get_config, save_api_key

logger = logging.getLogger(__name__)

# Setup router and templates
router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))

# In-memory run storage (in production, use Redis or DB)
active_runs = {}

# Maximum number of log entries to keep in memory per run
MAX_LOG_ENTRIES = 100


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main UI page."""
    config = get_config()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "config": config, "has_api_key": config.tomtom_api_key is not None},
    )


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

    try:
        # Capture event loop for thread-safe progress updates
        loop = asyncio.get_running_loop()

        # Thread-safe update progress helper
        def _do_update(stage: int, stage_name: str, message: str, level: str):
            if run_id in active_runs:
                active_runs[run_id]["progress"]["stage"] = stage
                active_runs[run_id]["progress"]["stage_name"] = stage_name
                active_runs[run_id]["progress"]["logs"].append({"message": message, "level": level})
                # Trim logs to max length
                if len(active_runs[run_id]["progress"]["logs"]) > MAX_LOG_ENTRIES:
                    active_runs[run_id]["progress"]["logs"] = active_runs[run_id]["progress"][
                        "logs"
                    ][-MAX_LOG_ENTRIES:]

        def update_progress(stage: int, stage_name: str, message: str, level: str = "info"):
            loop.call_soon_threadsafe(_do_update, stage, stage_name, message, level)

        update_progress(0, "Initializing", "Starting calibration pipeline...")

        # Create output directory
        output_dir = Path.cwd() / "demandify_runs" / f"run_{run_id}"

        # Store output_dir in active_runs for log file access
        active_runs[run_id]["output_dir"] = str(output_dir)

        # Initialize pipeline
        bbox = tuple(params["bbox"])
        pipeline = CalibrationPipeline(
            bbox=bbox,
            window_minutes=params["window_minutes"],
            seed=params["seed"],
            warmup_minutes=params.get("warmup_minutes", 5),
            step_length=params.get("step_length", 1.0),
            parallel_workers=params.get("parallel_workers"),
            traffic_tile_zoom=params.get("traffic_tile_zoom", 12),
            ga_population=params["ga_population"],
            ga_generations=params["ga_generations"],
            ga_mutation_rate=params["ga_mutation_rate"],
            ga_crossover_rate=params["ga_crossover_rate"],
            ga_elitism=params["ga_elitism"],
            ga_mutation_sigma=params["ga_mutation_sigma"],
            ga_mutation_indpb=params["ga_mutation_indpb"],
            ga_immigrant_rate=params.get("ga_immigrant_rate", 0.03),
            ga_elite_top_pct=params.get("ga_elite_top_pct", 0.1),
            ga_magnitude_penalty_weight=params.get("ga_magnitude_penalty_weight", 0.001),
            ga_stagnation_patience=params.get("ga_stagnation_patience", 20),
            ga_stagnation_boost=params.get("ga_stagnation_boost", 1.5),
            ga_assortative_mating=params.get("ga_assortative_mating", True),
            ga_deterministic_crowding=params.get("ga_deterministic_crowding", True),
            num_origins=params.get("num_origins", 10),
            num_destinations=params.get("num_destinations", 10),
            max_od_pairs=params.get("max_od_pairs", 50),
            bin_minutes=params.get("bin_minutes", 5),
            initial_population=params.get("initial_population", 1000),
            run_id=run_id,
            output_dir=output_dir,
            progress_callback=update_progress,
        )

        # Run pipeline
        # The pipeline will handle all stages and call update_progress
        metadata = await pipeline.run()

        if metadata:
            # Update final status
            active_runs[run_id]["status"] = "completed"
            active_runs[run_id]["metadata"] = metadata
            update_progress(8, "Complete", f"Scenario exported to {output_dir}")
        else:
            # Aborted?
            active_runs[run_id]["status"] = "aborted"
            update_progress(0, "Aborted", "Run aborted.")

    except Exception as e:
        logger.error(f"Pipeline failed for run {run_id}: {e}", exc_info=True)
        if run_id in active_runs:
            active_runs[run_id]["status"] = "failed"
            active_runs[run_id]["progress"]["logs"].append(
                {"message": f"Error: {str(e)}", "level": "error"}
            )
            # Trim logs to max length
            if len(active_runs[run_id]["progress"]["logs"]) > MAX_LOG_ENTRIES:
                active_runs[run_id]["progress"]["logs"] = active_runs[run_id]["progress"]["logs"][
                    -MAX_LOG_ENTRIES:
                ]


@router.post("/api/check_feasibility")
async def check_feasibility(
    bbox_west: float = Form(...),
    bbox_south: float = Form(...),
    bbox_east: float = Form(...),
    bbox_north: float = Form(...),
    run_id: Optional[str] = Form(None),
    traffic_tile_zoom: int = Form(12),
):
    """Run preparation phase to check data quality."""
    from demandify.config import get_config
    from demandify.pipeline import CalibrationPipeline

    config = get_config()
    if not config.tomtom_api_key:
        raise HTTPException(status_code=400, detail="TomTom API key not configured")

    # Use provided ID or generate temp one
    actual_run_id = run_id if run_id else f"check_{uuid.uuid4().hex}"

    bbox = (bbox_west, bbox_south, bbox_east, bbox_north)

    try:
        pipeline = CalibrationPipeline(
            bbox=bbox,
            window_minutes=15,  # Default for check
            seed=42,  # Default
            traffic_tile_zoom=traffic_tile_zoom,
            run_id=actual_run_id,
        )

        # Run preparation
        context = await pipeline.prepare()

        return {
            "status": "success",
            "run_id": actual_run_id,
            "stats": {
                "fetched_segments": len(context["traffic_df"]),
                "matched_edges": len(context["observed_edges"]),
                "total_network_edges": context.get("total_edges", 0),
            },
        }
    except Exception as e:
        logger.error(f"Check failed: {e}")
        return {"status": "error", "message": str(e)}


@router.post("/api/run")
async def start_run(
    background_tasks: BackgroundTasks,
    bbox_west: float = Form(...),
    bbox_south: float = Form(...),
    bbox_east: float = Form(...),
    bbox_north: float = Form(...),
    run_id: Optional[str] = Form(None),
    window_minutes: int = Form(15),
    seed: int = Form(42),
    warmup_minutes: int = Form(5),
    step_length: float = Form(1.0),
    traffic_tile_zoom: int = Form(12),
    ga_population: int = Form(50),
    ga_generations: int = Form(20),
    ga_mutation_rate: float = Form(0.5),
    ga_crossover_rate: float = Form(0.7),
    ga_elitism: int = Form(2),
    ga_mutation_sigma: int = Form(20),
    ga_mutation_indpb: float = Form(0.3),
    ga_immigrant_rate: float = Form(0.03),
    ga_elite_top_pct: float = Form(0.1),
    ga_magnitude_penalty_weight: float = Form(0.001),
    ga_stagnation_patience: int = Form(20),
    ga_stagnation_boost: float = Form(1.5),
    ga_assortative_mating: bool = Form(True),
    ga_deterministic_crowding: bool = Form(True),
    num_origins: int = Form(10),
    num_destinations: int = Form(10),
    max_od_pairs: int = Form(50),
    bin_minutes: int = Form(5),
    initial_population: int = Form(1000),
    parallel_workers: Optional[int] = Form(None),
):
    """Start a new calibration run."""
    from demandify.utils.validation import calculate_bbox_area_km2, validate_bbox

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

    # Create or use run ID
    actual_run_id = run_id if run_id else str(uuid.uuid4())

    # Store run metadata
    active_runs[actual_run_id] = {
        "status": "running",
        "progress": {"stage": 0, "stage_name": "Initializing", "logs": []},
        "created_at": datetime.now().isoformat(),
    }

    # Parameters
    params = {
        "bbox": [bbox_west, bbox_south, bbox_east, bbox_north],
        "window_minutes": window_minutes,
        "seed": seed,
        "warmup_minutes": warmup_minutes,
        "step_length": step_length,
        "traffic_tile_zoom": traffic_tile_zoom,
        "ga_population": ga_population,
        "ga_generations": ga_generations,
        "ga_mutation_rate": ga_mutation_rate,
        "ga_crossover_rate": ga_crossover_rate,
        "ga_elitism": ga_elitism,
        "ga_mutation_sigma": ga_mutation_sigma,
        "ga_mutation_indpb": ga_mutation_indpb,
        "ga_immigrant_rate": ga_immigrant_rate,
        "ga_elite_top_pct": ga_elite_top_pct,
        "ga_magnitude_penalty_weight": ga_magnitude_penalty_weight,
        "ga_stagnation_patience": ga_stagnation_patience,
        "ga_stagnation_boost": ga_stagnation_boost,
        "ga_assortative_mating": ga_assortative_mating,
        "ga_deterministic_crowding": ga_deterministic_crowding,
        "num_origins": num_origins,
        "num_destinations": num_destinations,
        "max_od_pairs": max_od_pairs,
        "bin_minutes": bin_minutes,
        "initial_population": initial_population,
        "parallel_workers": parallel_workers,
    }

    # Start background task
    background_tasks.add_task(run_calibration_pipeline, actual_run_id, params)

    return {"run_id": actual_run_id, "status": "started"}


@router.get("/api/run/{run_id}/progress")
async def get_progress(run_id: str):
    """Get progress for a run."""
    if run_id not in active_runs:
        raise HTTPException(status_code=404, detail="Run not found")

    run = active_runs[run_id]

    # QUICK FIX: Also read from log file if available
    output_dir = run.get("output_dir")
    if output_dir:
        log_file = Path(output_dir) / "logs" / "pipeline.log"
        if log_file.exists():
            try:
                # Read last 30 lines
                with open(log_file, "r") as f:
                    lines = f.readlines()
                    recent_lines = lines[-30:] if len(lines) > 30 else lines

                # Parse and add to logs
                # Create set of recent messages for efficient deduplication
                recent_messages = {
                    log_entry["message"] for log_entry in run["progress"]["logs"][-10:]
                }
                for line in recent_lines:
                    if " - INFO - " in line or " - WARNING - " in line:
                        msg = line.split(" - ", 3)[-1].strip()
                        if msg and msg not in recent_messages:
                            level = "warning" if "WARNING" in line else "info"
                            run["progress"]["logs"].append({"message": msg, "level": level})

                # Trim logs to max length after appending
                if len(run["progress"]["logs"]) > MAX_LOG_ENTRIES:
                    run["progress"]["logs"] = run["progress"]["logs"][-MAX_LOG_ENTRIES:]
            except Exception as e:
                logger.error(f"Error reading log file: {e}")

    return {
        **run["progress"],
        "status": run.get("status", "running"),
    }


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

    return templates.TemplateResponse(
        "results.html", {"request": request, "run_id": run_id, "run_data": active_runs[run_id]}
    )


@router.get("/api/runs")
async def list_runs():
    """List all existing run IDs found in the run directory."""
    runs_dir = Path.cwd() / "demandify_runs"
    runs = []

    if runs_dir.exists():
        for path in runs_dir.iterdir():
            if path.is_dir() and path.name.startswith("run_"):
                # Extract ID from "run_ID"
                run_id = path.name[4:]
                runs.append(run_id)

    return {"runs": runs}
