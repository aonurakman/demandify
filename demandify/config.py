"""
Configuration management for demandify.
Handles API keys, cache paths, and persistent settings.
"""
from pathlib import Path
from typing import Any, Dict, Optional
import os
import json
from copy import deepcopy
from importlib.resources import files

from pydantic import Field
from pydantic_settings import BaseSettings


_RUN_DEFAULTS_FALLBACK: Dict[str, Any] = {
    "window_options_minutes": [15, 30, 60],
    "window_minutes": 15,
    "seed": 42,
    "warmup_minutes": 5,
    "step_length": 1.0,
    "traffic_tile_zoom": 12,
    "ga_population": 50,
    "ga_generations": 20,
    "parallel_workers": "auto",
    "ga_elitism": 2,
    "ga_mutation_rate": 0.5,
    "ga_crossover_rate": 0.7,
    "initial_population": 1000,
    "num_origins": 10,
    "num_destinations": 10,
    "max_od_pairs": 50,
    "bin_minutes": 5,
    "ga_mutation_sigma": 20,
    "ga_mutation_indpb": 0.3,
    "ga_immigrant_rate": 0.03,
    "ga_elite_top_pct": 0.1,
    "ga_magnitude_penalty_weight": 0.001,
    "ga_stagnation_patience": 20,
    "ga_stagnation_boost": 1.5,
    "ga_assortative_mating": True,
    "ga_deterministic_crowding": True,
}


def _as_bool(value: Any, default: bool) -> bool:
    """Coerce various JSON-like values to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(default)


def _resolve_parallel_workers(value: Any) -> int:
    """Resolve workers value from JSON (supports 'auto')."""
    auto_workers = max(1, os.cpu_count() or 1)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "auto":
            return auto_workers
        try:
            return max(1, int(lowered))
        except ValueError:
            return auto_workers
    if value is None:
        return auto_workers
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return auto_workers


def _normalize_run_defaults(raw: Any) -> Dict[str, Any]:
    """Merge loaded JSON defaults with robust fallbacks and type coercion."""
    merged = deepcopy(_RUN_DEFAULTS_FALLBACK)
    if isinstance(raw, dict):
        merged.update(raw)

    options = merged.get("window_options_minutes", _RUN_DEFAULTS_FALLBACK["window_options_minutes"])
    if not isinstance(options, list) or not options:
        options = _RUN_DEFAULTS_FALLBACK["window_options_minutes"]
    normalized_options = sorted({max(1, int(v)) for v in options})
    merged["window_options_minutes"] = normalized_options

    merged["window_minutes"] = int(merged.get("window_minutes", _RUN_DEFAULTS_FALLBACK["window_minutes"]))
    if merged["window_minutes"] not in merged["window_options_minutes"]:
        merged["window_options_minutes"].append(merged["window_minutes"])
        merged["window_options_minutes"] = sorted(set(merged["window_options_minutes"]))

    merged["seed"] = int(merged.get("seed", _RUN_DEFAULTS_FALLBACK["seed"]))
    merged["warmup_minutes"] = int(
        merged.get("warmup_minutes", _RUN_DEFAULTS_FALLBACK["warmup_minutes"])
    )
    merged["step_length"] = float(merged.get("step_length", _RUN_DEFAULTS_FALLBACK["step_length"]))
    merged["traffic_tile_zoom"] = int(
        merged.get("traffic_tile_zoom", _RUN_DEFAULTS_FALLBACK["traffic_tile_zoom"])
    )
    merged["ga_population"] = int(
        merged.get("ga_population", _RUN_DEFAULTS_FALLBACK["ga_population"])
    )
    merged["ga_generations"] = int(
        merged.get("ga_generations", _RUN_DEFAULTS_FALLBACK["ga_generations"])
    )
    merged["parallel_workers"] = _resolve_parallel_workers(merged.get("parallel_workers"))
    merged["ga_elitism"] = int(merged.get("ga_elitism", _RUN_DEFAULTS_FALLBACK["ga_elitism"]))
    merged["ga_mutation_rate"] = float(
        merged.get("ga_mutation_rate", _RUN_DEFAULTS_FALLBACK["ga_mutation_rate"])
    )
    merged["ga_crossover_rate"] = float(
        merged.get("ga_crossover_rate", _RUN_DEFAULTS_FALLBACK["ga_crossover_rate"])
    )
    merged["initial_population"] = int(
        merged.get("initial_population", _RUN_DEFAULTS_FALLBACK["initial_population"])
    )
    merged["num_origins"] = int(merged.get("num_origins", _RUN_DEFAULTS_FALLBACK["num_origins"]))
    merged["num_destinations"] = int(
        merged.get("num_destinations", _RUN_DEFAULTS_FALLBACK["num_destinations"])
    )
    merged["max_od_pairs"] = int(merged.get("max_od_pairs", _RUN_DEFAULTS_FALLBACK["max_od_pairs"]))
    merged["bin_minutes"] = int(merged.get("bin_minutes", _RUN_DEFAULTS_FALLBACK["bin_minutes"]))
    merged["ga_mutation_sigma"] = int(
        merged.get("ga_mutation_sigma", _RUN_DEFAULTS_FALLBACK["ga_mutation_sigma"])
    )
    merged["ga_mutation_indpb"] = float(
        merged.get("ga_mutation_indpb", _RUN_DEFAULTS_FALLBACK["ga_mutation_indpb"])
    )
    merged["ga_immigrant_rate"] = float(
        merged.get("ga_immigrant_rate", _RUN_DEFAULTS_FALLBACK["ga_immigrant_rate"])
    )
    merged["ga_elite_top_pct"] = float(
        merged.get("ga_elite_top_pct", _RUN_DEFAULTS_FALLBACK["ga_elite_top_pct"])
    )
    merged["ga_magnitude_penalty_weight"] = float(
        merged.get(
            "ga_magnitude_penalty_weight",
            _RUN_DEFAULTS_FALLBACK["ga_magnitude_penalty_weight"],
        )
    )
    merged["ga_stagnation_patience"] = int(
        merged.get("ga_stagnation_patience", _RUN_DEFAULTS_FALLBACK["ga_stagnation_patience"])
    )
    merged["ga_stagnation_boost"] = float(
        merged.get("ga_stagnation_boost", _RUN_DEFAULTS_FALLBACK["ga_stagnation_boost"])
    )
    merged["ga_assortative_mating"] = _as_bool(
        merged.get("ga_assortative_mating"),
        _RUN_DEFAULTS_FALLBACK["ga_assortative_mating"],
    )
    merged["ga_deterministic_crowding"] = _as_bool(
        merged.get("ga_deterministic_crowding"),
        _RUN_DEFAULTS_FALLBACK["ga_deterministic_crowding"],
    )

    return merged


def _load_packaged_run_defaults() -> Dict[str, Any]:
    """Load packaged run defaults JSON with fallback behavior."""
    raw_defaults: Any = {}
    try:
        resource = files("demandify.defaults").joinpath("run_defaults.json")
        raw_defaults = json.loads(resource.read_text(encoding="utf-8"))
    except Exception:
        raw_defaults = {}
    return _normalize_run_defaults(raw_defaults)


_RUN_DEFAULTS = _load_packaged_run_defaults()


def get_run_defaults() -> Dict[str, Any]:
    """Return a copy of normalized run defaults shared by UI and CLI."""
    return deepcopy(_RUN_DEFAULTS)


class DemandifyConfig(BaseSettings):
    """Main configuration for demandify."""
    
    # API Keys
    tomtom_api_key: Optional[str] = Field(default=None, env="TOMTOM_API_KEY")
    
    # Paths
    cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".demandify" / "cache"
    )
    
    # Server settings
    host: str = "127.0.0.1"
    port: int = 8000
    
    # Simulation defaults
    default_window_minutes: int = _RUN_DEFAULTS["window_minutes"]
    default_warmup_minutes: int = _RUN_DEFAULTS["warmup_minutes"]
    default_step_length: float = _RUN_DEFAULTS["step_length"]
    default_traffic_tile_zoom: int = _RUN_DEFAULTS["traffic_tile_zoom"]
    
    # Calibration defaults
    default_ga_population: int = _RUN_DEFAULTS["ga_population"]
    default_ga_generations: int = _RUN_DEFAULTS["ga_generations"]
    default_parallel_workers: int = _RUN_DEFAULTS["parallel_workers"]
    
    # Limits
    max_bbox_area_km2: float = 25.0  # Warn above this
    max_observed_edges: int = 2000
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


_config_instance: Optional[DemandifyConfig] = None


def get_config() -> DemandifyConfig:
    """Get or create the global config instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = DemandifyConfig()
        
        # Ensure cache directory exists
        _config_instance.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load persistent config if exists
        _load_persistent_config(_config_instance)
    
    return _config_instance


def save_api_key(service: str, key: str):
    """Save an API key to persistent storage."""
    config = get_config()
    config_file = config.cache_dir.parent / "config.json"
    
    # Load existing config
    persistent = {}
    if config_file.exists():
        with open(config_file, "r") as f:
            persistent = json.load(f)
    
    # Update API keys
    if "api_keys" not in persistent:
        persistent["api_keys"] = {}
    
    persistent["api_keys"][service] = key
    
    # Save back
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, "w") as f:
        json.dump(persistent, f, indent=2)
    
    # Update current config
    if service == "tomtom":
        config.tomtom_api_key = key


def _load_persistent_config(config: DemandifyConfig):
    """Load persistent configuration from disk."""
    config_file = config.cache_dir.parent / "config.json"
    
    if not config_file.exists():
        return
    
    try:
        with open(config_file, "r") as f:
            persistent = json.load(f)
        
        # Load API keys
        api_keys = persistent.get("api_keys", {})
        if "tomtom" in api_keys and not config.tomtom_api_key:
            config.tomtom_api_key = api_keys["tomtom"]
    
    except Exception as e:
        print(f"Warning: Could not load persistent config: {e}")


def get_api_key(service: str) -> Optional[str]:
    """Get an API key for a service."""
    config = get_config()
    
    if service == "tomtom":
        return config.tomtom_api_key
    
    return None
