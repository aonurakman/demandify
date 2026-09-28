"""
FastAPI application setup.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from demandify import __version__
from demandify.web import dataset_routes, routes, studio_routes
from demandify.utils import logging  # Setup logging


# Get base directory
BASE_DIR = Path(__file__).parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize on startup, clean up on shutdown."""
    from demandify.config import get_config
    config = get_config()
    print(f"Cache directory: {config.cache_dir}")
    yield


# Create FastAPI app
app = FastAPI(
    title="demandify",
    description="Calibrate SUMO traffic simulations against real-world congestion data",
    version=__version__,
    lifespan=lifespan,
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
app.include_router(dataset_routes.router)
app.include_router(studio_routes.router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": __version__}
