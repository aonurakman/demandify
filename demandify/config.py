"""
Configuration management for demandify.
Handles API keys, cache paths, and persistent settings.
"""
from pathlib import Path
from typing import Optional
import os
import json

from pydantic import Field
from pydantic_settings import BaseSettings


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
    default_window_minutes: int = 15
    default_warmup_minutes: int = 5
    default_step_length: float = 1.0
    
    # Calibration defaults
    default_ga_population: int = 50
    default_ga_generations: int = 20
    default_parallel_workers: int = Field(
        default_factory=lambda: max(1, os.cpu_count() or 1)
    )
    
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
