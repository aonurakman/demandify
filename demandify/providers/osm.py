"""
OpenStreetMap data fetching via Overpass API.
"""
from typing import Tuple
from pathlib import Path
import httpx
import logging
import asyncio

logger = logging.getLogger(__name__)


class OSMFetcher:
    """Fetch OSM data using Overpass API with retry logic and fallbacks."""
    
    # Multiple Overpass instances for redundancy
    OVERPASS_INSTANCES = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.openstreetmap.ru/api/interpreter",
    ]
    
    def __init__(self, timeout: int = 180):
        """
        Initialize OSM fetcher.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
    
    async def fetch_osm_data(
        self,
        bbox: Tuple[float, float, float, float],
        output_file: Path,
        max_retries: int = 3
    ) -> Path:
        """
        Fetch OSM data for a bounding box with retry logic.
        
        Args:
            bbox: (west, south, east, north)
            output_file: Path to save OSM XML
            max_retries: Maximum number of retries per instance
        
        Returns:
            Path to saved OSM file
        """
        west, south, east, north = bbox
        
        # Build Overpass query
        query = f"""
        [bbox:{south},{west},{north},{east}]
        [timeout:180];
        (
          way["highway"];
          node(w);
        );
        out body;
        """
        
        logger.info(f"Fetching OSM data for bbox {bbox}")
        
        # Try each instance with retries
        last_error = None
        for instance_url in self.OVERPASS_INSTANCES:
            logger.info(f"Trying Overpass instance: {instance_url}")
            
            for attempt in range(max_retries):
                try:
                    response = await self.client.post(
                        instance_url,
                        data={"data": query}
                    )
                    
                    if response.status_code == 200:
                        # Success!
                        output_file.parent.mkdir(parents=True, exist_ok=True)
                        with open(output_file, 'wb') as f:
                            f.write(response.content)
                        
                        logger.info(f"OSM data saved to {output_file} ({len(response.content)} bytes)")
                        return output_file
                    
                    elif response.status_code == 429:
                        # Rate limited - wait and retry
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limited (429), waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    elif response.status_code == 504:
                        # Gateway timeout - try next instance
                        logger.warning(f"Gateway timeout (504) on attempt {attempt + 1}/{max_retries}")
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            break  # Try next instance
                    
                    else:
                        response.raise_for_status()
                
                except httpx.HTTPStatusError as e:
                    last_error = e
                    logger.error(f"HTTP error on {instance_url}: {e}")
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.info(f"Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        break  # Try next instance
                
                except Exception as e:
                    last_error = e
                    logger.error(f"Error fetching from {instance_url}: {e}")
                    break  # Try next instance
        
        # All instances failed
        error_msg = f"Failed to fetch OSM data from all Overpass instances. Last error: {last_error}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
