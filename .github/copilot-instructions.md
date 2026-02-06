# Copilot Instructions for demandify

## Repository Overview

**demandify** is a Python-based tool for calibrating SUMO (Simulation of Urban MObility) traffic simulations against real-world congestion data. It fetches live traffic data from TomTom API, converts OpenStreetMap road networks to SUMO format, and uses a genetic algorithm to optimize vehicle demand patterns to match observed traffic conditions.

- **Size**: ~13MB, 35 Python files
- **Languages**: Python 3.9+
- **Framework**: FastAPI web application with CLI interface
- **Key Dependencies**: SUMO (external), TomTom Traffic API, FastAPI, pandas, numpy, DEAP (genetic algorithms)
- **Package Manager**: pip with pyproject.toml (setuptools build system)

## Build and Test Instructions

### Installation

**Always install the package before making changes**:
```bash
pip install -e .
```

For development with testing tools:
```bash
pip install -e ".[dev]"
```

### Required External Dependency

**SUMO must be installed separately** - it's not a pip package. Users install it from https://eclipse.dev/sumo/. The `demandify doctor` command checks if SUMO is available.

### Testing

Run tests with pytest:
```bash
pytest
```

Run specific test:
```bash
pytest tests/test_cache_keys.py
```

Test with coverage:
```bash
pytest --cov=demandify
```

**Note**: Some tests may require SUMO to be installed on the system.

### Code Quality

**Always run these before committing**:

Format code:
```bash
black demandify/
```

Lint code:
```bash
ruff check demandify/
```

Type checking:
```bash
mypy demandify/
```

### Running the Application

Start web server:
```bash
demandify
```
Opens at http://127.0.0.1:8000

Run headless calibration:
```bash
demandify run "west,south,east,north" --name test_run
```

Check system requirements:
```bash
demandify doctor
```

## Project Layout and Architecture

### Directory Structure

```
/home/runner/work/demandify/demandify/
├── demandify/              # Main package
│   ├── __main__.py         # CLI entry point
│   ├── cli.py              # CLI command definitions
│   ├── app.py              # FastAPI application
│   ├── pipeline.py         # Main calibration pipeline orchestrator
│   ├── config.py           # Configuration management
│   ├── cache/              # Content-addressed caching system
│   ├── calibration/        # Genetic algorithm optimization
│   │   ├── objective.py    # Fitness function
│   │   ├── optimizer.py    # GA implementation
│   │   └── worker.py       # Parallel simulation workers
│   ├── providers/          # External data fetchers
│   │   ├── tomtom.py       # TomTom Traffic API client
│   │   └── osm.py          # OpenStreetMap data fetcher
│   ├── sumo/               # SUMO integration
│   │   ├── network.py      # OSM to SUMO conversion
│   │   ├── matching.py     # Traffic segment to SUMO edge matching
│   │   ├── demand.py       # Demand generation (OD pairs, time bins)
│   │   └── simulation.py   # SUMO simulation runner
│   ├── export/             # Output generation
│   │   ├── exporter.py     # Scenario file export
│   │   ├── report.py       # HTML report generation
│   │   └── custom_formats.py # Additional export formats
│   ├── web/                # Web UI
│   ├── templates/          # Jinja2 HTML templates
│   ├── static/             # Static assets (CSS, JS, images)
│   └── utils/              # Utility functions
├── tests/                  # Test suite
├── scripts/                # Utility scripts
│   ├── verify_parallel.py  # Parallel execution verification
│   └── verify_timeline.py  # Timeline verification
├── static/                 # Repository static assets (banner, screenshots)
├── pyproject.toml          # Project configuration
├── README.md               # Comprehensive documentation
├── .env.example            # Environment variable template
└── .gitignore             # Git ignore rules
```

### Key Configuration Files

- **pyproject.toml**: Project metadata, dependencies, and tool configuration (black, ruff, pytest, mypy)
- **MANIFEST.in**: Specifies additional files to include in package distribution
- **.env.example**: Template for environment variables (TomTom API key)

### Configuration

Tool configuration lives in pyproject.toml:
- Black: line-length = 100, target Python 3.10
- Ruff: line-length = 100, select = ["E", "F", "I", "N", "W"], ignore = ["E501"]
- Pytest: testpaths = ["tests"]
- Mypy: python_version = "3.9"

### Pipeline Architecture

The calibration pipeline (pipeline.py - CalibrationPipeline class) executes 8 stages:

1. **Validate inputs**: Check bbox, parameters, API key
2. **Fetch traffic snapshot**: Get real-time speeds from TomTom (cached by 5-min bucket)
3. **Fetch OSM extract**: Download road network data (cached by bbox)
4. **Build SUMO network**: Convert OSM to car-only SUMO .net.xml (cached)
5. **Map matching**: Match traffic segments to SUMO edges using spatial index (cached)
6. **Initialize demand**: Select routable OD pairs and time bins
7. **Calibrate demand**: Run genetic algorithm to optimize vehicle counts
8. **Export scenario**: Generate demand.csv, trips.xml, config, HTML report

### Caching System

Content-addressed caching in `~/.demandify/cache/`:
- OSM extracts: by bbox
- SUMO networks: by bbox + conversion params
- Traffic snapshots: by bbox + provider + style + tile zoom + 5-minute timestamp bucket
- Map matching results: by bbox + network key + provider + timestamp bucket

Cache keys defined in `demandify/cache/keys.py`.

### Important Implementation Details

1. **Seeding**: Random seed controls OD selection and GA evolution for reproducibility. Traffic snapshots bucketed to 5-minute windows ensure same data for same time bucket.

2. **Parallel Processing**: GA uses multiprocessing workers (configurable via --workers) for parallel fitness evaluation.

3. **SUMO Integration**: 
   - Uses subprocess calls to SUMO binaries (netconvert, sumo-gui, sumo)
   - Network conversion filters to car-only edges
   - Simulation runs with TraCI for programmatic control

4. **API Integration**: TomTom Traffic Flow API uses vector tiles. Free tier: 2,500 requests/day.

5. **Error Handling**: Scenario config includes `--ignore-route-errors` by default to handle routing failures gracefully.

## Development Workflow

### Making Changes

1. Install in editable mode: `pip install -e ".[dev]"`
2. Make changes to code
3. Format: `black demandify/`
4. Lint: `ruff check demandify/`
5. Test: `pytest`
6. Run manually if touching CLI/web: `demandify` or `demandify run ...`

### Adding Dependencies

- Add to `dependencies` array in pyproject.toml
- For dev-only dependencies, add to `[project.optional-dependencies].dev`
- Reinstall: `pip install -e ".[dev]"`

### Common Pitfalls

1. **SUMO not installed**: Many features require SUMO binaries (netconvert, sumo, sumo-gui) in PATH. Test with `demandify doctor`.

2. **TomTom API key**: Required for traffic data. Set via environment variable `TOMTOM_API_KEY`, `.env` file, or `demandify set-key`.

3. **Cache location**: Cache stored in `~/.demandify/`, not in repo. Clear with `demandify cache clear`.

4. **Import paths**: Package is "demandify", so imports are `from demandify.module import ...`

5. **Output directory**: Calibration runs create folders in `demandify_runs/` by default (gitignored).

6. **Line length**: Both black and ruff use 100 characters, not the default 88.

## Testing Notes

- Test suite in `tests/` directory
- Tests cover: cache key generation, demand conversion, network routing, SUMO doctor check, worker seeding
- Some tests may require SUMO installation
- Run individual tests for faster iteration during development

## CLI Commands Reference

```bash
demandify                    # Start web server (default)
demandify run "bbox"         # Headless calibration
demandify doctor             # Check SUMO installation
demandify set-key <key>      # Set TomTom API key
demandify cache clear        # Clear cache
demandify --version          # Show version
```

## File Types

- `.py`: Python source files
- `.net.xml`: SUMO network files
- `.rou.xml`, `.trips.xml`: SUMO route/trip files
- `.sumocfg`: SUMO configuration files
- `.md`: Markdown documentation
- `.html`: Jinja2 templates (in demandify/templates/)
- `.csv`: Data exports (demand.csv, observed_edges.csv)
- `.json`: Configuration and metadata files

## Notes for Code Changes

- **Respect existing style**: 100-char lines, type hints where present, docstrings for public APIs
- **Maintain caching**: When changing data fetching or processing, ensure cache keys remain consistent or update cache versioning
- **Test with real data**: If possible, test changes with actual TomTom API calls and SUMO simulations
- **Web UI changes**: Templates in `demandify/templates/`, static files in `demandify/static/`
- **Parallel safety**: Changes to calibration workers must be multiprocessing-safe (no shared state)
