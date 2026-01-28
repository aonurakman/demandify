"""
Cache manager for content-addressed artifact storage.
"""
from pathlib import Path
from typing import Optional, Any
import json
import shutil
import logging

logger = logging.getLogger(__name__)


class CacheManager:
    """Manage cached artifacts."""
    
    def __init__(self, cache_dir: Path):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Root cache directory
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        self.osm_dir = self.cache_dir / "osm"
        self.networks_dir = self.cache_dir / "networks"
        self.traffic_dir = self.cache_dir / "traffic"
        self.matching_dir = self.cache_dir / "matching"
        
        for d in [self.osm_dir, self.networks_dir, self.traffic_dir, self.matching_dir]:
            d.mkdir(exist_ok=True)
    
    def get_osm_path(self, cache_key: str) -> Path:
        """Get path for cached OSM file."""
        return self.osm_dir / f"{cache_key}.osm.xml"
    
    def get_network_path(self, cache_key: str) -> Path:
        """Get path for cached SUMO network."""
        return self.networks_dir / f"{cache_key}.net.xml"
    
    def get_traffic_path(self, cache_key: str) -> Path:
        """Get path for cached traffic data."""
        return self.traffic_dir / f"{cache_key}.pkl"
    
    def get_matching_path(self, cache_key: str) -> Path:
        """Get path for cached map matching results."""
        return self.matching_dir / f"{cache_key}.csv"
    
    def exists(self, path: Path) -> bool:
        """Check if cached file exists."""
        return path.exists()
    
    def save_metadata(self, path: Path, metadata: dict):
        """Save metadata alongside a cached file."""
        meta_path = path.with_suffix(path.suffix + '.meta.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_metadata(self, path: Path) -> Optional[dict]:
        """Load metadata for a cached file."""
        meta_path = path.with_suffix(path.suffix + '.meta.json')
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                return json.load(f)
        return None
    
    def clear(self):
        """Clear all cached data."""
        logger.info(f"Clearing cache: {self.cache_dir}")
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            for d in [self.osm_dir, self.networks_dir, self.traffic_dir, self.matching_dir]:
                d.mkdir(exist_ok=True)
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        stats = {
            'total_size_mb': 0,
            'osm_files': 0,
            'network_files': 0,
            'traffic_files': 0,
            'matching_files': 0
        }
        
        for category, directory in [
            ('osm_files', self.osm_dir),
            ('network_files', self.networks_dir),
            ('traffic_files', self.traffic_dir),
            ('matching_files', self.matching_dir)
        ]:
            if directory.exists():
                files = list(directory.glob('*'))
                stats[category] = len([f for f in files if not f.name.endswith('.meta.json')])
                stats['total_size_mb'] += sum(f.stat().st_size for f in files) / (1024 * 1024)
        
        return stats
