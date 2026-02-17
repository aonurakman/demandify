"""
Dedicated dataset-sampler routes for building reusable offline calibration datasets.
This module is intentionally separate from the core calibration run routes.
"""

import asyncio
import json
import logging
import re
import shutil
import uuid
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from demandify.config import get_config, get_run_defaults
from demandify.utils.data_quality import assess_data_quality
from demandify.utils.validation import calculate_bbox_area_km2, validate_bbox

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))
RUN_DEFAULTS = get_run_defaults()

MAX_LOG_ENTRIES = 100
DATASETS_ROOT = Path.cwd() / "demandify_datasets"
PACKAGED_DATASETS_ROOT = Path(__file__).resolve().parent.parent / "offline_datasets"
DATASET_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")

# In-memory dataset build jobs (in production, use Redis or DB)
active_dataset_jobs = {}


def _parse_bbox_from_meta(meta: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Extract and validate bbox from dataset metadata."""
    raw = meta.get("bbox")
    if not isinstance(raw, dict):
        return None

    try:
        west = float(raw["west"])
        south = float(raw["south"])
        east = float(raw["east"])
        north = float(raw["north"])
    except (KeyError, TypeError, ValueError):
        return None

    if not all(isfinite(v) for v in (west, south, east, north)):
        return None

    is_valid, _ = validate_bbox(west, south, east, north)
    if not is_valid:
        return None

    return {"west": west, "south": south, "east": east, "north": north}


def _collect_dataset_catalog(root: Path, source: str) -> List[Dict[str, Any]]:
    """Collect dataset entries with bbox metadata from a source directory."""
    entries: List[Dict[str, Any]] = []
    if not root.exists():
        return entries

    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        meta_path = path / "dataset_meta.json"
        if not meta_path.exists():
            continue

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            bbox = _parse_bbox_from_meta(meta)
            if bbox is None:
                continue

            quality = meta.get("quality", {}) if isinstance(meta.get("quality"), dict) else {}
            entries.append(
                {
                    "id": f"{source}:{path.name}",
                    "name": path.name,
                    "source": source,
                    "bbox": bbox,
                    "created_at": meta.get("created_at"),
                    "quality_label": quality.get("label"),
                    "quality_score": quality.get("score"),
                }
            )
        except Exception as exc:
            logger.warning(f"Skipping invalid dataset metadata at {meta_path}: {exc}")

    return entries


def _known_datasets_catalog() -> List[Dict[str, Any]]:
    """Return merged catalog from generated and packaged offline dataset stores."""
    generated = _collect_dataset_catalog(DATASETS_ROOT, "generated")
    packaged = _collect_dataset_catalog(PACKAGED_DATASETS_ROOT, "packaged")
    catalog = generated + packaged
    catalog.sort(key=lambda item: (item["name"], item["source"]))
    return catalog


def _trim_job_logs(job_id: str) -> None:
    """Cap in-memory logs to prevent unbounded growth."""
    logs = active_dataset_jobs[job_id]["progress"]["logs"]
    if len(logs) > MAX_LOG_ENTRIES:
        active_dataset_jobs[job_id]["progress"]["logs"] = logs[-MAX_LOG_ENTRIES:]


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


def _ingest_new_log_file_lines(job_id: str) -> None:
    """
    Incrementally ingest newly appended pipeline log lines.

    This avoids re-adding old lines on every progress poll.
    """
    job = active_dataset_jobs[job_id]
    output_dir = job.get("output_dir")
    if not output_dir:
        return

    log_file = Path(output_dir) / "logs" / "pipeline.log"
    if not log_file.exists():
        return

    cursor_key = "_pipeline_log_cursor"
    current_cursor = job.get(cursor_key, 0)

    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as exc:
        logger.error(f"Error reading dataset job log file: {exc}")
        return

    if current_cursor < 0 or current_cursor > len(lines):
        # File may have been truncated or recreated.
        current_cursor = 0

    new_lines = lines[current_cursor:]
    for line in new_lines:
        parsed = _parse_pipeline_log_line(line)
        if parsed is not None:
            job["progress"]["logs"].append(parsed)

    job[cursor_key] = len(lines)
    _trim_job_logs(job_id)


def _validate_dataset_name(dataset_name: str) -> str:
    """Validate and normalize dataset name for filesystem-safe storage."""
    normalized = dataset_name.strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="Dataset name is required")

    if not DATASET_NAME_PATTERN.fullmatch(normalized):
        raise HTTPException(
            status_code=400,
            detail="Dataset name must use only letters, numbers, underscores, and hyphens",
        )

    return normalized


def _build_dataset_metadata(
    params: dict,
    context: dict,
    provider_meta: dict,
    quality: dict,
) -> dict:
    """Build metadata manifest for offline dataset reuse."""
    return {
        "dataset_name": params["dataset_name"],
        "created_at": datetime.now().isoformat(),
        "bbox": {
            "west": params["bbox"][0],
            "south": params["bbox"][1],
            "east": params["bbox"][2],
            "north": params["bbox"][3],
        },
        "traffic_tile_zoom": params["traffic_tile_zoom"],
        "provider": provider_meta or {},
        "stats": {
            "fetched_segments": int(len(context["traffic_df"])),
            "matched_edges": int(len(context["observed_edges"])),
            "total_network_edges": int(context.get("total_edges", 0)),
        },
        "quality": quality,
        "files": {
            "traffic_data_raw_csv": "data/traffic_data_raw.csv",
            "observed_edges_csv": "data/observed_edges.csv",
            "osm_file": "data/map.osm",
            "sumo_network": "sumo/network.net.xml",
            "network_plot": "plots/network.png",
            "pipeline_log": "logs/pipeline.log",
        },
    }


@router.get("/dataset-builder", response_class=HTMLResponse)
async def dataset_builder_page(request: Request):
    """Dedicated UI for building offline datasets."""
    config = get_config()
    known_datasets = _known_datasets_catalog()
    return templates.TemplateResponse(
        "dataset_builder.html",
        {
            "request": request,
            "run_defaults": RUN_DEFAULTS,
            "has_api_key": config.tomtom_api_key is not None,
            "known_datasets": known_datasets,
        },
    )


@router.get("/api/datasets")
async def list_datasets():
    """List created offline datasets."""
    datasets = []
    if DATASETS_ROOT.exists():
        for path in DATASETS_ROOT.iterdir():
            if path.is_dir():
                datasets.append(path.name)
    datasets.sort()
    return {"datasets": datasets}


async def run_dataset_builder(job_id: str, params: dict):
    """Background task that runs prepare-only pipeline and stores offline dataset files."""
    from demandify.pipeline import CalibrationPipeline

    try:
        loop = asyncio.get_running_loop()

        def _do_update(stage: int, stage_name: str, message: str, level: str):
            if job_id in active_dataset_jobs:
                active_dataset_jobs[job_id]["progress"]["stage"] = stage
                active_dataset_jobs[job_id]["progress"]["stage_name"] = stage_name
                active_dataset_jobs[job_id]["progress"]["logs"].append(
                    {"message": message, "level": level}
                )
                _trim_job_logs(job_id)

        def update_progress(stage: int, stage_name: str, message: str, level: str = "info"):
            loop.call_soon_threadsafe(_do_update, stage, stage_name, message, level)

        update_progress(0, "Initializing", "Starting offline dataset build...")

        output_dir = Path(params["output_dir"])
        active_dataset_jobs[job_id]["output_dir"] = str(output_dir)

        # Prepare-only run: fetch traffic, build network, and map-match observed edges.
        pipeline = CalibrationPipeline(
            bbox=tuple(params["bbox"]),
            window_minutes=RUN_DEFAULTS["window_minutes"],
            seed=RUN_DEFAULTS["seed"],
            traffic_tile_zoom=params["traffic_tile_zoom"],
            run_id=f"dataset_{job_id}",
            output_dir=output_dir,
            progress_callback=update_progress,
        )
        context = await pipeline.prepare()

        # Keep OSM source in dataset bundle for future offline workflows.
        osm_file = output_dir / "data" / "map.osm"
        if not osm_file.exists():
            osm_cache_path = await pipeline._fetch_osm_data()
            shutil.copy2(osm_cache_path, osm_file)
            update_progress(4, "Matching Traffic", "✓ Added map.osm to dataset bundle")

        quality = assess_data_quality(
            context["traffic_df"],
            context["observed_edges"],
            context.get("total_edges", 0),
            bbox=tuple(params["bbox"]),
        )
        metadata = _build_dataset_metadata(
            params, context, pipeline.provider_meta, quality
        )
        metadata_path = output_dir / "dataset_meta.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        if job_id in active_dataset_jobs:
            active_dataset_jobs[job_id]["status"] = "completed"
            active_dataset_jobs[job_id]["metadata"] = metadata
            update_progress(5, "Complete", f"Dataset saved to {output_dir}")

    except Exception as exc:
        logger.error(f"Dataset build failed for job {job_id}: {exc}", exc_info=True)
        if job_id in active_dataset_jobs:
            active_dataset_jobs[job_id]["status"] = "failed"
            active_dataset_jobs[job_id]["progress"]["logs"].append(
                {"message": f"Error: {str(exc)}", "level": "error"}
            )
            _trim_job_logs(job_id)


@router.post("/api/datasets/build")
async def start_dataset_build(
    background_tasks: BackgroundTasks,
    dataset_name: str = Form(...),
    bbox_west: float = Form(...),
    bbox_south: float = Form(...),
    bbox_east: float = Form(...),
    bbox_north: float = Form(...),
    traffic_tile_zoom: int = Form(RUN_DEFAULTS["traffic_tile_zoom"]),
):
    """Start a new offline dataset build job."""
    normalized_name = _validate_dataset_name(dataset_name)

    is_valid, msg = validate_bbox(bbox_west, bbox_south, bbox_east, bbox_north)
    if not is_valid:
        raise HTTPException(status_code=400, detail=msg)

    config = get_config()
    if not config.tomtom_api_key:
        raise HTTPException(status_code=400, detail="TomTom API key not configured")

    area_km2 = calculate_bbox_area_km2(bbox_west, bbox_south, bbox_east, bbox_north)
    if area_km2 > config.max_bbox_area_km2:
        logger.warning(
            f"Large bbox area for dataset '{normalized_name}': {area_km2:.2f} km² "
            f"(limit={config.max_bbox_area_km2:.2f} km²)"
        )

    DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
    output_dir = DATASETS_ROOT / normalized_name
    if output_dir.exists():
        raise HTTPException(
            status_code=409,
            detail=f"Dataset '{normalized_name}' already exists. Use a different name.",
        )

    job_id = f"{normalized_name}_{uuid.uuid4().hex[:8]}"
    active_dataset_jobs[job_id] = {
        "status": "running",
        "dataset_name": normalized_name,
        "created_at": datetime.now().isoformat(),
        "progress": {"stage": 0, "stage_name": "Initializing", "logs": []},
    }

    params = {
        "dataset_name": normalized_name,
        "bbox": [bbox_west, bbox_south, bbox_east, bbox_north],
        "traffic_tile_zoom": traffic_tile_zoom,
        "output_dir": str(output_dir),
    }
    background_tasks.add_task(run_dataset_builder, job_id, params)

    return {"job_id": job_id, "status": "started", "dataset_name": normalized_name}


@router.get("/api/datasets/{job_id}/progress")
async def get_dataset_progress(job_id: str):
    """Get progress for an offline dataset build job."""
    if job_id not in active_dataset_jobs:
        raise HTTPException(status_code=404, detail="Dataset job not found")

    _ingest_new_log_file_lines(job_id)
    job = active_dataset_jobs[job_id]

    return {
        **job["progress"],
        "status": job.get("status", "running"),
        "dataset_name": job.get("dataset_name"),
        "output_dir": job.get("output_dir"),
    }


@router.get("/api/datasets/{job_id}/status")
async def get_dataset_status(job_id: str):
    """Get full status payload for a dataset build job."""
    if job_id not in active_dataset_jobs:
        raise HTTPException(status_code=404, detail="Dataset job not found")
    return active_dataset_jobs[job_id]
