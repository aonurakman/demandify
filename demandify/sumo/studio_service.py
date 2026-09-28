"""
Demand Studio service layer: handles network loading, coordinate conversion,
demand manipulation, background SUMO test simulation, and scenario persistence.
"""
from datetime import datetime
import json
import logging
import math
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from demandify.sumo.network import SUMONetwork, convert_osm_to_sumo
from demandify.sumo.matching import get_network_projection
from demandify.sumo.departure_schedule import (
    sequential_departure_times,
    format_departure_time,
    GOLDEN_RATIO_CONJUGATE,
)
from demandify.sumo.simulation import SUMOSimulation
from demandify.offline_data import get_offline_dataset_catalog, resolve_offline_dataset
from demandify.providers.osm import OSMFetcher

logger = logging.getLogger(__name__)

# Try to import pyproj
try:
    from pyproj import Transformer, CRS
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False
    logger.warning("pyproj not available in studio_service")


STUDIO_SCENARIOS_DIR = Path("demandify_scenarios")


def get_scenarios_root() -> Path:
    """Return root directory for saved scenarios."""
    STUDIO_SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
    return STUDIO_SCENARIOS_DIR


def list_studio_sources() -> Dict[str, List[Dict[str, Any]]]:
    """
    Catalog all sources available for Demand Studio:
    1. Offline Datasets (bundled and local)
    2. Completed or in-progress Calibration Runs
    3. Custom Saved Scenarios
    """
    sources: Dict[str, List[Dict[str, Any]]] = {
        "datasets": [],
        "runs": [],
        "scenarios": [],
    }

    # 1. Offline Datasets
    try:
        catalog = get_offline_dataset_catalog()
        for ds in catalog:
            try:
                resolved = resolve_offline_dataset(ds["id"])
                root_path = Path(resolved.root)
                net_path = root_path / "sumo" / "network.net.xml"
                demand_path = root_path / "data" / "demand.csv"
                sources["datasets"].append({
                    "id": ds["id"],
                    "name": ds["name"],
                    "source": ds.get("source", "offline"),
                    "type": "dataset",
                    "path": str(root_path),
                    "has_network": net_path.exists(),
                    "has_demand": demand_path.exists(),
                    "quality_label": ds.get("quality_label"),
                    "quality_score": ds.get("quality_score"),
                    "bbox": ds.get("bbox"),
                })
            except Exception as e:
                logger.warning(f"Error resolving dataset {ds.get('id')}: {e}")
                continue
    except Exception as e:
        logger.warning(f"Error cataloging offline datasets for studio: {e}")

    # 2. Calibration Runs
    runs_dir = Path("demandify_runs")
    if runs_dir.exists():
        for run_path in sorted(runs_dir.iterdir(), reverse=True):
            if not run_path.is_dir() or not run_path.name.startswith("run_"):
                continue
            net_path = run_path / "sumo" / "network.net.xml"
            if not net_path.exists():
                continue
            demand_path = run_path / "data" / "demand.csv"
            if not demand_path.exists():
                latest_demand = run_path / "latest_selected" / "data" / "demand.csv"
                if latest_demand.exists():
                    demand_path = latest_demand
            meta_path = run_path / "metadata.json"
            meta = {}
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    pass

            run_id = run_path.name.replace("run_", "", 1)
            sources["runs"].append({
                "id": run_id,
                "name": meta.get("run_id", run_id),
                "type": "run",
                "path": str(run_path),
                "has_network": True,
                "has_demand": demand_path.exists(),
                "created_at": meta.get("created_at"),
                "status": meta.get("status", "unknown"),
                "window_minutes": meta.get("window_minutes", 15),
            })

    # 3. Saved Custom Scenarios
    scenarios_dir = get_scenarios_root()
    for scen_path in sorted(scenarios_dir.iterdir(), reverse=True):
        if not scen_path.is_dir():
            continue
        net_path = scen_path / "sumo" / "network.net.xml"
        if not net_path.exists():
            continue
        demand_path = scen_path / "data" / "demand.csv"
        meta_path = scen_path / "scenario_meta.json"
        meta = {}
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                pass

        sources["scenarios"].append({
            "id": scen_path.name,
            "name": meta.get("name", scen_path.name),
            "type": "scenario",
            "path": str(scen_path),
            "has_network": True,
            "has_demand": demand_path.exists(),
            "created_at": meta.get("created_at"),
            "total_ods": meta.get("total_ods", 0),
            "total_demand_vehs_h": meta.get("total_demand_vehs_h", 0),
        })

    return sources


def resolve_source_paths(source_type: str, source_id: str) -> Tuple[Path, Optional[Path]]:
    """
    Given source_type ('dataset', 'run', 'scenario') and source_id,
    return (network_file, demand_csv_file_or_None).
    """
    if source_type == "dataset":
        from demandify.offline_data import resolve_offline_dataset
        ds = resolve_offline_dataset(source_id)
        root_path = Path(ds.root)
        net_file = root_path / "sumo" / "network.net.xml"
        demand_file = root_path / "data" / "demand.csv"
        return net_file, (demand_file if demand_file.exists() else None)

    elif source_type == "run":
        run_path = Path("demandify_runs") / f"run_{source_id}"
        if not run_path.exists():
            run_path = Path("demandify_runs") / source_id
        if not run_path.exists():
            raise FileNotFoundError(f"Run directory not found: {source_id}")
        net_file = run_path / "sumo" / "network.net.xml"
        demand_file = run_path / "data" / "demand.csv"
        if not demand_file.exists():
            latest_demand = run_path / "latest_selected" / "data" / "demand.csv"
            if latest_demand.exists():
                demand_file = latest_demand
        return net_file, (demand_file if demand_file.exists() else None)

    elif source_type == "scenario":
        scen_path = get_scenarios_root() / source_id
        if not scen_path.exists():
            raise FileNotFoundError(f"Scenario directory not found: {source_id}")
        net_file = scen_path / "sumo" / "network.net.xml"
        demand_file = scen_path / "data" / "demand.csv"
        return net_file, (demand_file if demand_file.exists() else None)

    else:
        raise ValueError(f"Unknown source type: {source_type}")


class StudioNetworkManager:
    """Manages spatial parsing, routing, and GeoJSON export for Demand Studio."""

    def __init__(self, network_file: Path):
        self.network_file = Path(network_file)
        if not self.network_file.exists():
            raise FileNotFoundError(f"Network file not found: {self.network_file}")

        self.network = SUMONetwork(self.network_file)
        self.proj_str, self.offset = get_network_projection(self.network_file)
        self._setup_transformer()

        # Cache BFS shortest path edges
        self._adjacency = {
            k: tuple(sorted(v)) for k, v in self.network.adjacency.items()
        }

    def _setup_transformer(self):
        """Set up inverse transformer from SUMO local coords to WGS84 (lon, lat)."""
        self.transformer = None
        if not HAS_PYPROJ:
            return

        if self.proj_str and self.proj_str.strip() and self.proj_str.strip() != "!":
            try:
                target_crs = CRS.from_proj4(self.proj_str)
                # inverse: from network target_crs to EPSG:4326 (lon, lat)
                self.transformer = Transformer.from_crs(target_crs, "EPSG:4326", always_xy=True)
                return
            except Exception as e:
                logger.warning(f"Could not build transformer from projParameter: {e}")

        # If projParameter is plain geo or identity
        self.transformer = None

    def sumo_to_geo(self, x: float, y: float) -> Tuple[float, float]:
        """Convert SUMO network coordinate (x, y) to (lon, lat)."""
        if self.transformer is not None:
            px = x - self.offset[0]
            py = y - self.offset[1]
            lon, lat = self.transformer.transform(px, py)
            return (float(lon), float(lat))
        return (float(x), float(y))

    def get_routable_edges(self) -> List[str]:
        """Return list of valid, non-internal edges allowed for passenger vehicles."""
        valid = []
        for edge_id in self.network.edges:
            if edge_id.startswith(":"):
                continue
            geom = self.network.get_edge_geometry(edge_id)
            if geom is not None and geom.length > 1.0:
                valid.append(edge_id)
        return valid

    def find_shortest_path(self, from_edge: str, to_edge: str) -> List[str]:
        """Compute shortest path between edges using BFS on network topology."""
        if from_edge == to_edge:
            return [from_edge]

        visited: Dict[str, Optional[str]] = {from_edge: None}
        queue = [from_edge]
        idx = 0

        while idx < len(queue):
            curr = queue[idx]
            idx += 1
            if curr == to_edge:
                break
            for nbr in self._adjacency.get(curr, ()):
                if nbr not in visited:
                    visited[nbr] = curr
                    queue.append(nbr)

        if to_edge not in visited:
            return []

        path = []
        curr = to_edge
        while curr is not None:
            path.append(curr)
            curr = visited[curr]
        path.reverse()
        return path

    def get_edge_geojson_coordinates(self, edge_id: str) -> List[List[float]]:
        """Return list of [lon, lat] points for an edge."""
        geom = self.network.get_edge_geometry(edge_id)
        if geom is None:
            return []
        coords = []
        for x, y in geom.coords:
            lon, lat = self.sumo_to_geo(x, y)
            coords.append([round(lon, 6), round(lat, 6)])
        return coords

    def get_path_geojson_coordinates(self, path_edges: List[str]) -> List[List[float]]:
        """Return continuous line coordinates for a sequence of path edges."""
        all_coords = []
        for edge in path_edges:
            edge_coords = self.get_edge_geojson_coordinates(edge)
            if not edge_coords:
                continue
            if not all_coords:
                all_coords.extend(edge_coords)
            else:
                # Avoid duplicating connecting vertex
                if all_coords[-1] == edge_coords[0]:
                    all_coords.extend(edge_coords[1:])
                else:
                    all_coords.extend(edge_coords)
        return all_coords

    def build_network_geojson(
        self,
        edge_flows: Optional[Dict[str, float]] = None,
        edge_congestion: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """
        Build GeoJSON FeatureCollection of all routable passenger edges.
        Includes flow bandwidth and optional congestion metrics.
        """
        edge_flows = edge_flows or {}
        edge_congestion = edge_congestion or {}

        features = []
        lons = []
        lats = []

        for edge_id in self.get_routable_edges():
            coords = self.get_edge_geojson_coordinates(edge_id)
            if len(coords) < 2:
                continue

            for lon, lat in coords:
                lons.append(lon)
                lats.append(lat)

            attrs = self.network.get_edge_attributes(edge_id)
            flow = float(edge_flows.get(edge_id, 0.0))
            cong = edge_congestion.get(edge_id, {})

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords,
                },
                "properties": {
                    "id": edge_id,
                    "speed_limit_kmh": round(float(attrs.get("speed", 13.89)) * 3.6, 1),
                    "lanes": int(attrs.get("numLanes", 1)),
                    "type": str(attrs.get("type", "")),
                    "flow": round(flow, 1),
                    "flow_vehs_h": round(flow, 1),
                    "flow_vehs_min": round(flow / 60.0, 2),
                    "sim_speed_kmh": round(cong.get("sim_speed", 0.0), 1) if cong else None,
                    "speed_ratio": round(cong.get("ratio", 1.0), 2) if cong else None,
                },
            })

        bounds = None
        if lons and lats:
            bounds = [
                [min(lats), min(lons)],
                [max(lats), max(lons)],
            ]

        return {
            "type": "FeatureCollection",
            "features": features,
            "bounds": bounds,
            "edge_count": len(features),
        }

    def find_nearest_edge(self, target_lat: float, target_lon: float) -> Optional[Dict[str, Any]]:
        """Snap a clicked (lat, lon) to the nearest routable SUMO edge."""
        best_edge = None
        best_dist_sq = float("inf")
        best_centroid = None

        for edge_id in self.get_routable_edges():
            coords = self.get_edge_geojson_coordinates(edge_id)
            if not coords:
                continue
            # Distance to midpoint / vertices
            for lon, lat in coords:
                d2 = (lat - target_lat) ** 2 + (lon - target_lon) ** 2
                if d2 < best_dist_sq:
                    best_dist_sq = d2
                    best_edge = edge_id
                    best_centroid = [lon, lat]

        if best_edge:
            attrs = self.network.get_edge_attributes(best_edge)
            return {
                "edge_id": best_edge,
                "speed_kmh": round(float(attrs.get("speed", 13.89)) * 3.6, 1),
                "type": str(attrs.get("type", "")),
                "lanes": int(attrs.get("numLanes", 1)),
                "coordinates": self.get_edge_geojson_coordinates(best_edge),
                "centroid": best_centroid,
            }
        return None


def parse_demand_dataframe(demand_df: pd.DataFrame, window_minutes: int = 15) -> List[Dict[str, Any]]:
    """
    Parse a demand.csv DataFrame into an aggregated list of OD pairs:
    returns [{"id": "...", "origin": "...", "destination": "...", "vehs_h": X, "vehs_min": Y, "trips_count": N}]
    """
    if demand_df.empty or "origin link id" not in demand_df.columns:
        return []

    # Group by origin and destination
    grouped = demand_df.groupby(["origin link id", "destination link id"]).size().reset_index(name="count")
    scale_to_hourly = 60.0 / max(1.0, float(window_minutes))

    od_list = []
    for idx, row in grouped.iterrows():
        orig = str(row["origin link id"])
        dest = str(row["destination link id"])
        count = int(row["count"])
        vehs_h = round(count * scale_to_hourly)
        vehs_min = round(vehs_h / 60.0, 2)
        od_list.append({
            "id": f"od_{idx}",
            "origin": orig,
            "destination": dest,
            "trips_in_window": count,
            "vehs_per_hour": vehs_h,
            "vehs_per_min": vehs_min,
            "base_vehs_per_hour": vehs_h,
        })

    # Sort descending by volume
    od_list.sort(key=lambda x: x["vehs_per_hour"], reverse=True)
    return od_list


def compute_edge_flows(manager: StudioNetworkManager, od_pairs: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute aggregate flow (vehs/h) across edges from all active OD pairs."""
    flows: Dict[str, float] = {}
    for od in od_pairs:
        path = manager.find_shortest_path(od["origin"], od["destination"])
        od["path"] = path
        vol = float(od.get("vehs_per_hour", 0.0))
        if vol <= 0:
            continue
        for edge in path:
            flows[edge] = flows.get(edge, 0.0) + vol
    return flows


def apply_multiplier(
    od_pairs: List[Dict[str, Any]],
    factor: float,
    preserve_min_one: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Scale OD flows by a multiplier factor (e.g. 0.1 to 2.0).
    Uses 'base_vehs_per_hour' as the anchor to prevent compounding drift.
    """
    factor = max(0.05, min(3.0, float(factor)))
    updated = []
    total_orig = 0
    total_scaled = 0

    for od in od_pairs:
        base_h = float(od.get("base_vehs_per_hour", od.get("vehs_per_hour", 0)))
        total_orig += int(round(base_h))

        if base_h <= 0:
            scaled_h = 0
        else:
            scaled_h = int(round(base_h * factor))
            if preserve_min_one and scaled_h == 0 and factor > 0:
                scaled_h = 1

        total_scaled += scaled_h
        item = dict(od)
        item["vehs_per_hour"] = scaled_h
        item["vehs_per_min"] = round(scaled_h / 60.0, 2)
        item["base_vehs_per_hour"] = int(round(base_h))
        updated.append(item)

    delta_pct = 0.0
    if total_orig > 0:
        delta_pct = round(((total_scaled - total_orig) / total_orig) * 100.0, 1)

    summary = {
        "multiplier": factor,
        "total_vehs_h_original": total_orig,
        "total_vehs_h_scaled": total_scaled,
        "delta_pct": delta_pct,
        "total_ods": len(updated),
        "active_ods": sum(1 for x in updated if x["vehs_per_hour"] > 0),
    }

    return updated, summary


def generate_studio_trips_df(
    od_pairs: List[Dict[str, Any]],
    window_minutes: int = 15,
) -> pd.DataFrame:
    """Generate deterministic trips dataframe from OD insertion rates."""
    window_seconds = int(window_minutes * 60)
    trips = []
    trip_id = 0
    num_ods = len(od_pairs)
    stagger = num_ods > 1

    for od_idx, od in enumerate(od_pairs):
        orig = str(od["origin"])
        dest = str(od["destination"])
        vehs_h = float(od.get("vehs_per_hour", 0.0))
        if vehs_h <= 0:
            continue

        # Trips count for this simulation window
        count = int(round(vehs_h * (window_minutes / 60.0)))
        if count <= 0 and vehs_h > 0:
            count = 1

        phase_offset = (((od_idx + 1) * GOLDEN_RATIO_CONJUGATE) % 1.0) if stagger else None
        dep_times = sequential_departure_times(0, window_seconds, count, phase_offset=phase_offset)

        for t in dep_times:
            trips.append({
                "ID": f"studio_trip_{trip_id}",
                "origin link id": orig,
                "destination link id": dest,
                "departure timestep": float(t),
            })
            trip_id += 1

    df = pd.DataFrame(trips, columns=["ID", "origin link id", "destination link id", "departure timestep"])
    if not df.empty:
        df = df.sort_values(by=["departure timestep", "origin link id", "destination link id", "ID"]).reset_index(drop=True)
    return df


def run_studio_test_simulation(
    network_file: Path,
    od_pairs: List[Dict[str, Any]],
    window_minutes: int = 15,
    warmup_minutes: int = 3,
    mesosim: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Execute a rapid background SUMO simulation (using mesosim) to evaluate
    the current demand and produce an edge congestion heatmap.
    """
    # Filter out any unroutable OD pairs to ensure valid simulation
    mgr = StudioNetworkManager(network_file)
    od_pairs = [od for od in od_pairs if mgr.find_shortest_path(str(od["origin"]), str(od["destination"]))]

    trips_df = generate_studio_trips_df(od_pairs, window_minutes=window_minutes)
    if trips_df.empty:
        return {
            "status": "error",
            "message": "No active trips in current demand scenario. Add or increase OD volume.",
        }

    temp_dir = Path(tempfile.mkdtemp(prefix="demandify_studio_test_"))
    try:
        demand_csv = temp_dir / "demand.csv"
        trips_df.to_csv(demand_csv, index=False)

        # Write trips.xml
        trips_xml = temp_dir / "trips.xml"
        import xml.etree.ElementTree as ET
        root = ET.Element("routes")
        for _, row in trips_df.iterrows():
            trip = ET.SubElement(root, "trip")
            trip.set("id", str(row["ID"]))
            trip.set("depart", format_departure_time(row["departure timestep"]))
            trip.set("from", str(row["origin link id"]))
            trip.set("to", str(row["destination link id"]))
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(trips_xml, encoding="utf-8", xml_declaration=True)

        sim = SUMOSimulation(
            network_file=network_file,
            vehicle_file=trips_xml,
            step_length=1.0,
            warmup_time=int(warmup_minutes * 60),
            simulation_time=int(window_minutes * 60),
            seed=seed,
            use_dynamic_routing=True,
            mesosim=mesosim,
        )

        edge_speeds, trip_stats = sim.run()

        # Build congestion map
        net = SUMONetwork(network_file)
        edge_congestion: Dict[str, Dict[str, float]] = {}

        for edge_id, sim_speed in edge_speeds.items():
            attrs = net.get_edge_attributes(edge_id)
            ff_speed = float(attrs.get("speed", 13.89)) * 3.6
            ratio = min(1.0, max(0.0, sim_speed / max(ff_speed, 1.0)))
            edge_congestion[edge_id] = {
                "sim_speed": round(sim_speed, 1),
                "freeflow_speed": round(ff_speed, 1),
                "ratio": round(ratio, 2),
            }

        total_vehicles = len(trips_df)
        completed = trip_stats.get("completed_trips", 0)
        teleports = trip_stats.get("teleports", 0)
        avg_speed = 0.0
        if edge_speeds:
            avg_speed = round(sum(edge_speeds.values()) / len(edge_speeds), 1)

        return {
            "status": "success",
            "edge_congestion": edge_congestion,
            "stats": {
                "total_vehicles_inserted": total_vehicles,
                "completed_trips": completed,
                "completion_rate_pct": round((completed / max(1, total_vehicles)) * 100.0, 1),
                "teleports": teleports,
                "mean_edge_speed_kmh": avg_speed,
                "avg_duration_s": round(trip_stats.get("avg_duration", 0.0), 1),
                "avg_waiting_time_s": round(trip_stats.get("avg_waiting_time", 0.0), 1),
                "simulated_edges_count": len(edge_speeds),
            },
        }

    except Exception as e:
        logger.exception("Error running studio test simulation")
        return {
            "status": "error",
            "message": str(e),
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def save_studio_scenario(
    scenario_id: str,
    network_file: Path,
    od_pairs: List[Dict[str, Any]],
    window_minutes: int = 15,
    name: Optional[str] = None,
    source_ref: Optional[str] = None,
    last_test_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Save the current demand studio state as a complete, standalone scenario.
    Directory structure:
      demandify_scenarios/<scenario_id>/
        sumo/network.net.xml
        sumo/trips.xml
        sumo/scenario.sumocfg
        data/demand.csv
        scenario_meta.json
    """
    slug = re.sub(r"[^a-zA-Z0-9_\-]", "_", scenario_id).strip("_").lower()
    if not slug:
        slug = f"scenario_{int(datetime.now().timestamp())}"

    dest_dir = get_scenarios_root() / slug
    dest_dir.mkdir(parents=True, exist_ok=True)
    sumo_dir = dest_dir / "sumo"
    data_dir = dest_dir / "data"
    sumo_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy network
    dest_net = sumo_dir / "network.net.xml"
    shutil.copy2(network_file, dest_net)

    # Copy network meta if present
    src_meta = network_file.with_suffix(".meta.json")
    if src_meta.exists():
        shutil.copy2(src_meta, dest_net.with_suffix(".meta.json"))

    # 2. Save demand.csv (filtering any unroutable ODs)
    mgr = StudioNetworkManager(network_file)
    od_pairs = [od for od in od_pairs if mgr.find_shortest_path(str(od["origin"]), str(od["destination"]))]

    trips_df = generate_studio_trips_df(od_pairs, window_minutes=window_minutes)
    dest_demand = data_dir / "demand.csv"
    trips_df.to_csv(dest_demand, index=False)

    # 3. Generate trips.xml
    dest_trips = sumo_dir / "trips.xml"
    import xml.etree.ElementTree as ET
    root = ET.Element("routes")
    for _, row in trips_df.iterrows():
        trip = ET.SubElement(root, "trip")
        trip.set("id", str(row["ID"]))
        trip.set("depart", format_departure_time(row["departure timestep"]))
        trip.set("from", str(row["origin link id"]))
        trip.set("to", str(row["destination link id"]))
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(dest_trips, encoding="utf-8", xml_declaration=True)

    # 4. Write scenario.sumocfg
    sumocfg_file = sumo_dir / "scenario.sumocfg"
    cfg_root = ET.Element("configuration")
    input_elem = ET.SubElement(cfg_root, "input")
    ET.SubElement(input_elem, "net-file", value="network.net.xml")
    ET.SubElement(input_elem, "route-files", value="trips.xml")
    time_elem = ET.SubElement(cfg_root, "time")
    ET.SubElement(time_elem, "begin", value="0")
    ET.SubElement(time_elem, "end", value=str(int(window_minutes * 60)))
    ET.SubElement(time_elem, "step-length", value="1.0")
    proc_elem = ET.SubElement(cfg_root, "processing")
    ET.SubElement(proc_elem, "ignore-route-errors", value="true")
    ET.SubElement(proc_elem, "device.rerouting.adaptation-interval", value="10")
    cfg_tree = ET.ElementTree(cfg_root)
    ET.indent(cfg_tree, space="  ")
    cfg_tree.write(sumocfg_file, encoding="utf-8", xml_declaration=True)

    # 5. Metadata
    total_vehs_h = sum(int(x.get("vehs_per_hour", 0)) for x in od_pairs)
    meta = {
        "id": slug,
        "name": name or slug,
        "created_at": datetime.now().isoformat(),
        "source_ref": source_ref,
        "window_minutes": window_minutes,
        "total_ods": len(od_pairs),
        "active_ods": sum(1 for x in od_pairs if x.get("vehs_per_hour", 0) > 0),
        "total_demand_vehs_h": total_vehs_h,
        "total_demand_vehs_min": round(total_vehs_h / 60.0, 2),
        "total_trips_in_window": len(trips_df),
        "last_test_stats": last_test_stats,
    }
    with open(dest_dir / "scenario_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "scenario_id": slug,
        "scenario_name": name or slug,
        "path": str(dest_dir),
        "metadata": meta,
    }


async def fetch_osm_and_build_studio_network(
    bbox: Tuple[float, float, float, float],
    name: str,
) -> Dict[str, Any]:
    """
    Fetch raw OSM via Overpass API for a given bounding box, convert to SUMO
    network, and create a blank studio scenario ready for custom demand population.
    """
    slug = re.sub(r"[^a-zA-Z0-9_\-]", "_", name).strip("_").lower()
    if not slug:
        slug = f"osm_{int(datetime.now().timestamp())}"

    dest_dir = get_scenarios_root() / slug
    dest_dir.mkdir(parents=True, exist_ok=True)
    sumo_dir = dest_dir / "sumo"
    data_dir = dest_dir / "data"
    sumo_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    osm_file = data_dir / "map.osm"
    net_file = sumo_dir / "network.net.xml"

    # Fetch OSM data
    fetcher = OSMFetcher(timeout=180)
    logger.info(f"Fetching OSM for bbox {bbox} -> {osm_file}")
    await fetcher.fetch_bbox(bbox, osm_file)

    # Convert to SUMO
    logger.info(f"Converting OSM to SUMO -> {net_file}")
    convert_osm_to_sumo(osm_file, net_file, car_only=True)

    # Write blank demand.csv
    blank_df = pd.DataFrame(columns=["ID", "origin link id", "destination link id", "departure timestep"])
    blank_df.to_csv(data_dir / "demand.csv", index=False)

    # Write initial metadata
    meta = {
        "id": slug,
        "name": name,
        "created_at": datetime.now().isoformat(),
        "source_ref": f"osm_bbox:{bbox}",
        "bbox": {"west": bbox[0], "south": bbox[1], "east": bbox[2], "north": bbox[3]},
        "window_minutes": 15,
        "total_ods": 0,
        "active_ods": 0,
        "total_demand_vehs_h": 0,
        "total_demand_vehs_min": 0.0,
    }
    with open(dest_dir / "scenario_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "scenario_id": slug,
        "scenario_name": name,
        "path": str(dest_dir),
        "net_file": str(net_file),
    }
