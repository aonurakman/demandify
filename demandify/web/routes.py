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

from demandify.config import get_config, get_run_defaults, save_api_key
from demandify.offline_data import (
    get_offline_dataset_catalog,
    get_writable_offline_datasets_root,
    normalize_offline_dataset_name,
    offline_dataset_name_exists,
    resolve_offline_dataset,
)
from demandify.utils.data_quality import assess_data_quality
from demandify.utils.validation import calculate_bbox_area_km2, validate_bbox

logger = logging.getLogger(__name__)

# Setup router and templates
router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))
RUN_DEFAULTS = get_run_defaults()

# In-memory run storage (in production, use Redis or DB)
active_runs = {}

# Maximum number of log entries to keep in memory per run
MAX_LOG_ENTRIES = 100


def _trim_run_logs(run_id: str) -> None:
    """Cap in-memory logs for a run."""
    logs = active_runs[run_id]["progress"]["logs"]
    if len(logs) > MAX_LOG_ENTRIES:
        active_runs[run_id]["progress"]["logs"] = logs[-MAX_LOG_ENTRIES:]


def _parse_pipeline_log_line(line: str):
    """Parse a pipeline log file line into a UI log entry."""
    level = None
    if " - INFO - " in line:
        level = "info"
    elif " - WARNING - " in line:
        level = "warning"
    elif " - ERROR - " in line:
        level = "error"

    if level is None:
        return None

    message = line.split(" - ", 3)[-1].strip()
    if not message:
        return None

    return {"message": message, "level": level}


def _ingest_new_log_file_lines(run_id: str) -> None:
    """
    Incrementally ingest newly appended pipeline log lines.

    This avoids re-adding old lines on every progress poll.
    """
    run = active_runs[run_id]
    output_dir = run.get("output_dir")
    if not output_dir:
        return

    log_file = Path(output_dir) / "logs" / "pipeline.log"
    if not log_file.exists():
        return

    cursor_key = "_pipeline_log_cursor"
    current_cursor = run.get(cursor_key, 0)

    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"Error reading log file: {e}")
        return

    if current_cursor < 0 or current_cursor > len(lines):
        # File may have been truncated or recreated.
        current_cursor = 0

    new_lines = lines[current_cursor:]
    for line in new_lines:
        parsed = _parse_pipeline_log_line(line)
        if parsed is not None:
            run["progress"]["logs"].append(parsed)

    run[cursor_key] = len(lines)
    _trim_run_logs(run_id)


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main UI page."""
    config = get_config()
    offline_datasets = get_offline_dataset_catalog(include_generated=True, include_packaged=True)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "config": config,
            "run_defaults": RUN_DEFAULTS,
            "has_api_key": config.tomtom_api_key is not None,
            "offline_datasets": offline_datasets,
        },
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
                _trim_run_logs(run_id)

        def update_progress(stage: int, stage_name: str, message: str, level: str = "info"):
            loop.call_soon_threadsafe(_do_update, stage, stage_name, message, level)

        update_progress(0, "Initializing", "Starting calibration pipeline...")

        # Create output directory
        output_dir = Path.cwd() / "demandify_runs" / f"run_{run_id}"

        # Store output_dir in active_runs for log file access
        active_runs[run_id]["output_dir"] = str(output_dir)

        # Initialize pipeline
        bbox = tuple(params["bbox"]) if params.get("bbox") is not None else None
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
            offline_dataset=params.get("offline_dataset"),
            save_offline_dataset=params.get("save_offline_dataset", False),
            save_offline_dataset_name=params.get("save_offline_dataset_name"),
            save_offline_dataset_root=params.get("save_offline_dataset_root"),
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
            _trim_run_logs(run_id)


@router.post("/api/check_feasibility")
async def check_feasibility(
    data_mode: str = Form("create"),
    offline_dataset: Optional[str] = Form(None),
    bbox_west: Optional[float] = Form(None),
    bbox_south: Optional[float] = Form(None),
    bbox_east: Optional[float] = Form(None),
    bbox_north: Optional[float] = Form(None),
    run_id: Optional[str] = Form(None),
    traffic_tile_zoom: int = Form(RUN_DEFAULTS["traffic_tile_zoom"]),
):
    """Run preparation phase to check data quality."""
    from demandify.config import get_config
    from demandify.pipeline import CalibrationPipeline

    # Use provided ID or generate temp one
    actual_run_id = run_id if run_id else f"check_{uuid.uuid4().hex}"

    data_mode = (data_mode or "create").strip().lower()
    bbox: Optional[tuple] = None
    selected_dataset_ref: Optional[str] = None

    if data_mode not in {"create", "import"}:
        raise HTTPException(status_code=400, detail="Invalid data_mode. Use 'create' or 'import'.")

    if data_mode == "import":
        if not offline_dataset:
            raise HTTPException(status_code=400, detail="offline_dataset is required in import mode")
        try:
            resolved = resolve_offline_dataset(offline_dataset)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        bbox = (
            resolved.bbox["west"],
            resolved.bbox["south"],
            resolved.bbox["east"],
            resolved.bbox["north"],
        )
        selected_dataset_ref = resolved.dataset_id
    else:
        if None in {bbox_west, bbox_south, bbox_east, bbox_north}:
            raise HTTPException(status_code=400, detail="bbox is required in create mode")
        is_valid, msg = validate_bbox(bbox_west, bbox_south, bbox_east, bbox_north)
        if not is_valid:
            raise HTTPException(status_code=400, detail=msg)
        bbox = (bbox_west, bbox_south, bbox_east, bbox_north)
        config = get_config()
        if not config.tomtom_api_key:
            raise HTTPException(status_code=400, detail="TomTom API key not configured")

    try:
        pipeline = CalibrationPipeline(
            bbox=bbox,
            window_minutes=RUN_DEFAULTS["window_minutes"],
            seed=RUN_DEFAULTS["seed"],
            traffic_tile_zoom=traffic_tile_zoom,
            offline_dataset=selected_dataset_ref,
            run_id=actual_run_id,
        )

        # Run preparation
        context = await pipeline.prepare()
        quality = assess_data_quality(
            context["traffic_df"],
            context["observed_edges"],
            context.get("total_edges", 0),
            bbox=bbox,
        )

        return {
            "status": "success",
            "run_id": actual_run_id,
            "stats": {
                "fetched_segments": len(context["traffic_df"]),
                "matched_edges": len(context["observed_edges"]),
                "total_network_edges": context.get("total_edges", 0),
            },
            "data_mode": data_mode,
            "offline_dataset": selected_dataset_ref,
            "quality": quality,
        }
    except Exception as e:
        logger.error(f"Check failed: {e}")
        return {"status": "error", "message": str(e)}


@router.post("/api/run")
async def start_run(
    background_tasks: BackgroundTasks,
    data_mode: str = Form("create"),
    offline_dataset: Optional[str] = Form(None),
    bbox_west: Optional[float] = Form(None),
    bbox_south: Optional[float] = Form(None),
    bbox_east: Optional[float] = Form(None),
    bbox_north: Optional[float] = Form(None),
    run_id: Optional[str] = Form(None),
    window_minutes: int = Form(RUN_DEFAULTS["window_minutes"]),
    seed: int = Form(RUN_DEFAULTS["seed"]),
    warmup_minutes: int = Form(RUN_DEFAULTS["warmup_minutes"]),
    step_length: float = Form(RUN_DEFAULTS["step_length"]),
    traffic_tile_zoom: int = Form(RUN_DEFAULTS["traffic_tile_zoom"]),
    ga_population: int = Form(RUN_DEFAULTS["ga_population"]),
    ga_generations: int = Form(RUN_DEFAULTS["ga_generations"]),
    ga_mutation_rate: float = Form(RUN_DEFAULTS["ga_mutation_rate"]),
    ga_crossover_rate: float = Form(RUN_DEFAULTS["ga_crossover_rate"]),
    ga_elitism: int = Form(RUN_DEFAULTS["ga_elitism"]),
    ga_mutation_sigma: int = Form(RUN_DEFAULTS["ga_mutation_sigma"]),
    ga_mutation_indpb: float = Form(RUN_DEFAULTS["ga_mutation_indpb"]),
    ga_immigrant_rate: float = Form(RUN_DEFAULTS["ga_immigrant_rate"]),
    ga_elite_top_pct: float = Form(RUN_DEFAULTS["ga_elite_top_pct"]),
    ga_magnitude_penalty_weight: float = Form(RUN_DEFAULTS["ga_magnitude_penalty_weight"]),
    ga_stagnation_patience: int = Form(RUN_DEFAULTS["ga_stagnation_patience"]),
    ga_stagnation_boost: float = Form(RUN_DEFAULTS["ga_stagnation_boost"]),
    ga_assortative_mating: bool = Form(RUN_DEFAULTS["ga_assortative_mating"]),
    ga_deterministic_crowding: bool = Form(RUN_DEFAULTS["ga_deterministic_crowding"]),
    num_origins: int = Form(RUN_DEFAULTS["num_origins"]),
    num_destinations: int = Form(RUN_DEFAULTS["num_destinations"]),
    max_od_pairs: int = Form(RUN_DEFAULTS["max_od_pairs"]),
    bin_minutes: int = Form(RUN_DEFAULTS["bin_minutes"]),
    initial_population: int = Form(RUN_DEFAULTS["initial_population"]),
    parallel_workers: Optional[int] = Form(RUN_DEFAULTS["parallel_workers"]),
    save_offline_dataset: bool = Form(False),
    save_offline_dataset_name: Optional[str] = Form(None),
):
    """Start a new calibration run."""
    data_mode = (data_mode or "create").strip().lower()
    if data_mode not in {"create", "import"}:
        raise HTTPException(status_code=400, detail="Invalid data_mode. Use 'create' or 'import'.")

    config = get_config()
    resolved_dataset_ref: Optional[str] = None
    bbox_values: Optional[list] = None
    resolved_save_dataset_name: Optional[str] = None
    resolved_save_dataset_root: Optional[Path] = None

    if data_mode == "import":
        if not offline_dataset:
            raise HTTPException(status_code=400, detail="offline_dataset is required in import mode")
        try:
            resolved = resolve_offline_dataset(offline_dataset)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        resolved_dataset_ref = resolved.dataset_id
        bbox_values = [
            resolved.bbox["west"],
            resolved.bbox["south"],
            resolved.bbox["east"],
            resolved.bbox["north"],
        ]
    else:
        if None in {bbox_west, bbox_south, bbox_east, bbox_north}:
            raise HTTPException(status_code=400, detail="bbox is required in create mode")
        is_valid, msg = validate_bbox(bbox_west, bbox_south, bbox_east, bbox_north)
        if not is_valid:
            raise HTTPException(status_code=400, detail=msg)

        area_km2 = calculate_bbox_area_km2(bbox_west, bbox_south, bbox_east, bbox_north)
        if area_km2 > config.max_bbox_area_km2:
            logger.warning(f"Large bbox area: {area_km2:.2f} km²")

        if not config.tomtom_api_key:
            raise HTTPException(status_code=400, detail="TomTom API key not configured")
        bbox_values = [bbox_west, bbox_south, bbox_east, bbox_north]

    if save_offline_dataset:
        if data_mode != "create":
            raise HTTPException(
                status_code=400,
                detail="save_offline_dataset is only supported in create mode",
            )
        try:
            resolved_save_dataset_name = normalize_offline_dataset_name(
                save_offline_dataset_name or ""
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        if offline_dataset_name_exists(resolved_save_dataset_name):
            raise HTTPException(
                status_code=409,
                detail=f"Offline dataset '{resolved_save_dataset_name}' already exists. "
                "Use a different name.",
            )

        try:
            resolved_save_dataset_root = get_writable_offline_datasets_root()
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        if (resolved_save_dataset_root / resolved_save_dataset_name).exists():
            raise HTTPException(
                status_code=409,
                detail=f"Offline dataset '{resolved_save_dataset_name}' already exists at "
                f"{resolved_save_dataset_root}.",
            )
    elif save_offline_dataset_name and save_offline_dataset_name.strip():
        raise HTTPException(
            status_code=400,
            detail="save_offline_dataset_name requires save_offline_dataset=true",
        )

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
        "data_mode": data_mode,
        "offline_dataset": resolved_dataset_ref,
        "bbox": bbox_values,
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
        "save_offline_dataset": save_offline_dataset,
        "save_offline_dataset_name": resolved_save_dataset_name,
        "save_offline_dataset_root": (
            str(resolved_save_dataset_root) if resolved_save_dataset_root else None
        ),
    }

    # Start background task
    background_tasks.add_task(run_calibration_pipeline, actual_run_id, params)

    return {"run_id": actual_run_id, "status": "started"}


@router.get("/api/offline-datasets")
async def list_offline_datasets():
    """List all discovered offline datasets (generated + packaged)."""
    datasets = get_offline_dataset_catalog(include_generated=True, include_packaged=True)
    return {"datasets": datasets}


@router.get("/api/run/{run_id}/progress")
async def get_progress(run_id: str):
    """Get progress for a run."""
    if run_id not in active_runs:
        raise HTTPException(status_code=404, detail="Run not found")

    _ingest_new_log_file_lines(run_id)
    run = active_runs[run_id]

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
