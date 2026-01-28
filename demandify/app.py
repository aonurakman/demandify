"""
FastAPI application setup.
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from demandify import __version__
from demandify.web import routes
from demandify.utils import logging  # Setup logging


# Get base directory
BASE_DIR = Path(__file__).parent

# Create FastAPI app
app = FastAPI(
    title="demandify",
    description="Calibrate SUMO traffic simulations against real-world congestion data",
    version=__version__
)

# Mount static files
static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Setup templates
templates_dir = BASE_DIR / "templates"
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))

# Include routes
app.include_router(routes.router)


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    from demandify.config import get_config
    config = get_config()
    print(f"Cache directory: {config.cache_dir}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": __version__}
