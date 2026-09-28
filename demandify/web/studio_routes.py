"""
Demand Studio API and UI routes.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from demandify import __version__
from demandify.sumo.studio_service import (
    list_studio_sources,
    resolve_source_paths,
    StudioNetworkManager,
    parse_demand_dataframe,
    compute_edge_flows,
    apply_multiplier,
    run_studio_test_simulation,
    save_studio_scenario,
    fetch_osm_and_build_studio_network,
)
import pandas as pd

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Demand Studio"])


# In-memory cache for loaded StudioNetworkManagers to avoid re-parsing SUMO XML on every click
_NETWORK_CACHE: Dict[str, StudioNetworkManager] = {}


def _get_network_manager(net_file: Path) -> StudioNetworkManager:
    key = str(net_file.resolve())
    if key not in _NETWORK_CACHE:
        _NETWORK_CACHE[key] = StudioNetworkManager(net_file)
    return _NETWORK_CACHE[key]


# Pydantic Request Models
class RouteRequest(BaseModel):
    source_type: str
    source_id: str
    origin: str
    destination: str


class DestinationLookupRequest(BaseModel):
    source_type: str
    source_id: str
    origin: str
    od_pairs: List[Dict[str, Any]] = Field(default_factory=list)


class MultiplierRequest(BaseModel):
    od_pairs: List[Dict[str, Any]]
    factor: float = Field(ge=0.05, le=3.0)
    preserve_min_one: bool = True


class TestSimulationRequest(BaseModel):
    source_type: str
    source_id: str
    od_pairs: List[Dict[str, Any]]
    window_minutes: int = 15
    mesosim: bool = True


class SaveScenarioRequest(BaseModel):
    source_type: str
    source_id: str
    scenario_id: str
    name: Optional[str] = None
    od_pairs: List[Dict[str, Any]]
    window_minutes: int = 15
    last_test_stats: Optional[Dict[str, Any]] = None


class FetchOSMRequest(BaseModel):
    bbox: List[float] = Field(..., min_length=4, max_length=4)
    name: str


@router.get("/demand-studio", response_class=HTMLResponse)
async def demand_studio_page(request: Request):
    """Render the main Demand Studio page."""
    from demandify.app import templates
    sources = list_studio_sources()
    return templates.TemplateResponse(
        request=request,
        name="demand_studio.html",
        context={
            "version": __version__,
            "sources": sources,
            "active_tab": "studio",
        },
    )


@router.get("/api/studio/sources")
async def get_studio_sources():
    """Return all cataloged offline datasets, runs, and saved scenarios."""
    return list_studio_sources()


@router.get("/api/studio/load")
async def load_studio_network(
    source_type: str = Query(..., description="dataset | run | scenario"),
    source_id: str = Query(..., description="ID or slug of the source"),
):
    """
    Load a road network and its associated demand for interactive editing.
    """
    try:
        net_file, demand_file = resolve_source_paths(source_type, source_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    mgr = _get_network_manager(net_file)
    od_pairs: List[Dict[str, Any]] = []

    if demand_file and demand_file.exists():
        try:
            df = pd.read_csv(demand_file)
            od_pairs = parse_demand_dataframe(df, window_minutes=15)
        except Exception as e:
            logger.warning(f"Error parsing demand file {demand_file}: {e}")

    # Compute aggregate edge flows
    edge_flows = compute_edge_flows(mgr, od_pairs)

    # Build network GeoJSON
    network_geojson = mgr.build_network_geojson(edge_flows=edge_flows)

    total_vehs_h = sum(int(x.get("vehs_per_hour", 0)) for x in od_pairs)

    return {
        "source_type": source_type,
        "source_id": source_id,
        "network_file": str(net_file),
        "demand_file": str(demand_file) if demand_file else None,
        "network": network_geojson,
        "edge_flows": edge_flows,
        "od_pairs": od_pairs,
        "routable_edges": mgr.get_routable_edges(),
        "summary": {
            "total_ods": len(od_pairs),
            "active_ods": sum(1 for x in od_pairs if x.get("vehs_per_hour", 0) > 0),
            "total_vehs_h": total_vehs_h,
            "total_vehs_min": round(total_vehs_h / 60.0, 2),
            "total_network_edges": network_geojson["edge_count"],
            "duration_minutes": 60,
        },
    }


@router.post("/api/studio/route")
async def compute_studio_route(req: RouteRequest):
    """
    Compute nominal route between origin and destination edge.
    Returns path edge sequence and GeoJSON line coordinates.
    """
    try:
        net_file, _ = resolve_source_paths(req.source_type, req.source_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    mgr = _get_network_manager(net_file)
    path = mgr.find_shortest_path(req.origin, req.destination)
    if not path:
        return {
            "routable": False,
            "path_edges": [],
            "coordinates": [],
            "distance_m": 0.0,
            "estimated_time_s": 0.0,
        }

    coords = mgr.get_path_geojson_coordinates(path)

    # Calculate approximate length and duration
    total_len = 0.0
    total_time = 0.0
    for edge in path:
        geom = mgr.network.get_edge_geometry(edge)
        attrs = mgr.network.get_edge_attributes(edge)
        speed = float(attrs.get("speed", 13.89))  # m/s
        if geom:
            total_len += geom.length
            total_time += (geom.length / max(speed, 1.0))

    return {
        "routable": True,
        "path_edges": path,
        "coordinates": coords,
        "distance_m": round(total_len, 1),
        "estimated_time_s": round(total_time, 1),
    }


@router.post("/api/studio/find-destinations")
async def find_studio_destinations(req: DestinationLookupRequest):
    """
    Given an origin edge, return all existing OD destinations from this origin
    along with their flows and route coordinates.
    """
    try:
        net_file, _ = resolve_source_paths(req.source_type, req.source_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    mgr = _get_network_manager(net_file)
    matching_ods = []

    for od in req.od_pairs:
        if str(od["origin"]) == str(req.origin):
            path = mgr.find_shortest_path(str(od["origin"]), str(od["destination"]))
            coords = mgr.get_path_geojson_coordinates(path) if path else []
            matching_ods.append({
                "id": od.get("id"),
                "destination": str(od["destination"]),
                "vehs_per_hour": od.get("vehs_per_hour", 0),
                "vehs_per_min": od.get("vehs_per_min", 0.0),
                "path_edges": path,
                "coordinates": coords,
            })

    # Also return origin edge info
    attrs = mgr.network.get_edge_attributes(req.origin)
    coords_orig = mgr.get_edge_geojson_coordinates(req.origin)

    return {
        "origin": req.origin,
        "origin_attributes": attrs,
        "origin_coordinates": coords_orig,
        "existing_destinations": matching_ods,
    }


@router.post("/api/studio/apply-multiplier")
async def apply_studio_multiplier(req: MultiplierRequest):
    """Apply scaling factor to OD pairs."""
    updated_ods, summary = apply_multiplier(
        req.od_pairs,
        factor=req.factor,
        preserve_min_one=req.preserve_min_one,
    )
    return {
        "od_pairs": updated_ods,
        "summary": summary,
    }


@router.post("/api/studio/test-simulation")
async def test_studio_simulation(req: TestSimulationRequest):
    """
    Execute a background mesoscopic simulation pass to test the demand.
    Returns edge congestion metrics and simulation statistics.
    """
    try:
        net_file, _ = resolve_source_paths(req.source_type, req.source_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = run_studio_test_simulation(
        network_file=net_file,
        od_pairs=req.od_pairs,
        window_minutes=req.window_minutes,
        mesosim=req.mesosim,
    )

    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message", "Simulation failed"))

    return result


@router.post("/api/studio/save-scenario")
async def save_studio_scenario_endpoint(req: SaveScenarioRequest):
    """Save the customized scenario."""
    try:
        net_file, _ = resolve_source_paths(req.source_type, req.source_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    saved = save_studio_scenario(
        scenario_id=req.scenario_id,
        network_file=net_file,
        od_pairs=req.od_pairs,
        window_minutes=req.window_minutes,
        name=req.name or req.scenario_id,
        source_ref=f"{req.source_type}:{req.source_id}",
        last_test_stats=req.last_test_stats,
    )
    return saved


@router.post("/api/studio/fetch-osm")
async def fetch_osm_endpoint(req: FetchOSMRequest):
    """Fetch OpenStreetMap data and build a blank studio scenario network."""
    bbox_tuple = (float(req.bbox[0]), float(req.bbox[1]), float(req.bbox[2]), float(req.bbox[3]))
    try:
        result = await fetch_osm_and_build_studio_network(bbox=bbox_tuple, name=req.name)
        return {
            "status": "success",
            "scenario_id": result["scenario_id"],
            "scenario_name": result["scenario_name"],
            "path": result["path"],
        }
    except Exception as e:
        logger.exception("Error in fetch_osm_endpoint")
        raise HTTPException(status_code=500, detail=str(e))
