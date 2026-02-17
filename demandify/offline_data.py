"""
Shared helpers for discovering and using offline calibration datasets.
"""

import json
import logging
import re
import shutil
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from demandify.utils.validation import validate_bbox

logger = logging.getLogger(__name__)

GENERATED_DATASETS_ROOT = Path.cwd() / "demandify_datasets"
PACKAGED_DATASETS_ROOT = Path(__file__).resolve().parent / "offline_datasets"
DATASET_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")

REQUIRED_DATASET_FILES = (
    "data/traffic_data_raw.csv",
    "data/observed_edges.csv",
    "sumo/network.net.xml",
    "dataset_meta.json",
)


@dataclass
class OfflineDatasetResolved:
    """Resolved offline dataset with source metadata and filesystem handle."""

    dataset_id: str
    name: str
    source: str
    bbox: Dict[str, float]
    created_at: Optional[str]
    quality_label: Optional[str]
    quality_score: Optional[float]
    provider: Dict[str, Any]
    meta: Dict[str, Any]
    root: Any  # pathlib.Path or importlib.resources.abc.Traversable


def _packaged_datasets_root():
    """Return packaged offline dataset root as Traversable."""
    return files("demandify").joinpath("offline_datasets")


def normalize_offline_dataset_name(dataset_name: str) -> str:
    """Validate and normalize dataset names for filesystem-safe storage."""
    normalized = (dataset_name or "").strip()
    if not normalized:
        raise ValueError("Dataset name is required")
    if not DATASET_NAME_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Dataset name must use only letters, numbers, underscores, and hyphens"
        )
    return normalized


def offline_dataset_name_exists(
    dataset_name: str,
    include_generated: bool = True,
    include_packaged: bool = True,
) -> bool:
    """Return whether an offline dataset name already exists in discovered catalogs."""
    normalized = normalize_offline_dataset_name(dataset_name)
    catalog = _collect_internal_catalog(
        include_generated=include_generated,
        include_packaged=include_packaged,
    )
    return any(item["name"] == normalized for item in catalog)


def _is_directory_writable(path: Path) -> bool:
    """Best-effort writable check for a directory path."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".demandify_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def get_writable_offline_datasets_root() -> Path:
    """
    Pick a writable root for storing offline dataset bundles.

    Preference:
    1) package-local `demandify/offline_datasets` (source checkouts)
    2) cwd-local `demandify_datasets` fallback (pip installs / read-only site-packages)
    """
    candidates = (PACKAGED_DATASETS_ROOT, GENERATED_DATASETS_ROOT)
    for candidate in candidates:
        if _is_directory_writable(candidate):
            return candidate
    raise RuntimeError(
        "No writable directory available for offline datasets. "
        f"Tried: {', '.join(str(p) for p in candidates)}"
    )


def _parse_bbox(meta: Dict[str, Any]) -> Optional[Dict[str, float]]:
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

    valid, _ = validate_bbox(west, south, east, north)
    if not valid:
        return None

    return {"west": west, "south": south, "east": east, "north": north}


def _read_json_resource(resource: Any) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(resource.read_text(encoding="utf-8"))
    except Exception:
        return None


def _resource_exists(resource: Any) -> bool:
    try:
        return resource.exists()
    except Exception:
        return False


def _collect_catalog_from_root(root: Any, source: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    try:
        children = list(root.iterdir())
    except Exception:
        return items

    for child in children:
        try:
            if not child.is_dir():
                continue
        except Exception:
            continue

        meta_path = child.joinpath("dataset_meta.json")
        if not _resource_exists(meta_path):
            continue

        meta = _read_json_resource(meta_path)
        if not meta:
            logger.warning("Skipping offline dataset %s: invalid dataset_meta.json", child)
            continue

        bbox = _parse_bbox(meta)
        if bbox is None:
            logger.warning("Skipping offline dataset %s: missing/invalid bbox", child)
            continue

        quality = meta.get("quality") if isinstance(meta.get("quality"), dict) else {}
        provider = meta.get("provider") if isinstance(meta.get("provider"), dict) else {}
        name = getattr(child, "name", str(child))

        items.append(
            {
                "id": f"{source}:{name}",
                "name": name,
                "source": source,
                "bbox": bbox,
                "created_at": meta.get("created_at"),
                "quality_label": quality.get("label"),
                "quality_score": quality.get("score"),
                "provider": provider,
                "meta": meta,
                "_root": child,
            }
        )
    return items


def get_offline_dataset_catalog(
    include_generated: bool = True,
    include_packaged: bool = True,
) -> List[Dict[str, Any]]:
    """
    Discover offline datasets dynamically from configured sources.

    Returns public (JSON-safe) dictionaries suitable for UI/API payloads.
    """
    all_items: List[Dict[str, Any]] = []

    if include_generated:
        all_items.extend(_collect_catalog_from_root(GENERATED_DATASETS_ROOT, "generated"))
    if include_packaged:
        all_items.extend(_collect_catalog_from_root(_packaged_datasets_root(), "packaged"))

    all_items.sort(key=lambda x: (x["name"], x["source"]))

    # Remove internal handles before exposing externally.
    public_items = []
    for item in all_items:
        public_items.append(
            {
                "id": item["id"],
                "name": item["name"],
                "source": item["source"],
                "bbox": item["bbox"],
                "created_at": item.get("created_at"),
                "quality_label": item.get("quality_label"),
                "quality_score": item.get("quality_score"),
            }
        )

    return public_items


def _collect_internal_catalog(
    include_generated: bool = True,
    include_packaged: bool = True,
) -> List[Dict[str, Any]]:
    all_items: List[Dict[str, Any]] = []

    if include_generated:
        all_items.extend(_collect_catalog_from_root(GENERATED_DATASETS_ROOT, "generated"))
    if include_packaged:
        all_items.extend(_collect_catalog_from_root(_packaged_datasets_root(), "packaged"))

    all_items.sort(key=lambda x: (x["name"], x["source"]))
    return all_items


def resolve_offline_dataset(
    dataset_ref: str,
    include_generated: bool = True,
    include_packaged: bool = True,
) -> OfflineDatasetResolved:
    """
    Resolve an offline dataset by id (`source:name`) or by unique name.
    """
    if not dataset_ref or not dataset_ref.strip():
        raise ValueError("Offline dataset name is required")

    dataset_ref = dataset_ref.strip()
    catalog = _collect_internal_catalog(
        include_generated=include_generated, include_packaged=include_packaged
    )

    matches: List[Dict[str, Any]]
    if ":" in dataset_ref:
        matches = [item for item in catalog if item["id"] == dataset_ref]
    else:
        matches = [item for item in catalog if item["name"] == dataset_ref]

    if not matches:
        available = [item["name"] for item in catalog]
        raise ValueError(
            f"Offline dataset '{dataset_ref}' not found. "
            f"Available datasets: {', '.join(sorted(set(available))) if available else 'none'}"
        )

    if len(matches) > 1:
        ids = ", ".join(item["id"] for item in matches)
        raise ValueError(
            f"Offline dataset name '{dataset_ref}' is ambiguous. "
            f"Use one of: {ids}"
        )

    item = matches[0]
    return OfflineDatasetResolved(
        dataset_id=item["id"],
        name=item["name"],
        source=item["source"],
        bbox=item["bbox"],
        created_at=item.get("created_at"),
        quality_label=item.get("quality_label"),
        quality_score=item.get("quality_score"),
        provider=item.get("provider") or {},
        meta=item.get("meta") or {},
        root=item["_root"],
    )


def _copy_resource_file(src: Any, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as fsrc, open(dst, "wb") as fdst:
        shutil.copyfileobj(fsrc, fdst)


def _get_existing_source_files(dataset_root: Any, rel_paths: Sequence[str]) -> List[str]:
    existing = []
    for rel in rel_paths:
        source_resource = dataset_root.joinpath(rel)
        if _resource_exists(source_resource):
            existing.append(rel)
    return existing


def copy_offline_dataset_to_output(
    dataset: OfflineDatasetResolved,
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Copy offline dataset artifacts into a run output directory.
    """
    dataset_root = dataset.root

    missing_required = []
    for rel in REQUIRED_DATASET_FILES:
        if not _resource_exists(dataset_root.joinpath(rel)):
            missing_required.append(rel)
    if missing_required:
        raise FileNotFoundError(
            f"Offline dataset '{dataset.dataset_id}' is missing required files: "
            f"{', '.join(missing_required)}"
        )

    copy_plan = _get_existing_source_files(
        dataset_root,
        (
            "data/traffic_data_raw.csv",
            "data/observed_edges.csv",
            "sumo/network.net.xml",
            "data/map.osm",
            "plots/network.png",
            "dataset_meta.json",
        ),
    )

    copied_paths: Dict[str, Path] = {}
    for rel in copy_plan:
        src = dataset_root.joinpath(rel)
        dst = output_dir / rel
        _copy_resource_file(src, dst)
        copied_paths[rel] = dst

    # Keep source metadata for provenance in output bundle.
    source_meta_copy = output_dir / "data" / "imported_dataset_meta.json"
    if "dataset_meta.json" in copied_paths:
        _copy_resource_file(dataset_root.joinpath("dataset_meta.json"), source_meta_copy)
        copied_paths["data/imported_dataset_meta.json"] = source_meta_copy

    return copied_paths
