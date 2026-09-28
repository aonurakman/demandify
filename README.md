![demandify](https://github.com/aonurakman/demandify/blob/main/static/banner.png?raw=true)

[![PyPI version](https://badge.fury.io/py/demandify.svg)](https://pypi.org/project/demandify/)
[![DOI](https://zenodo.org/badge/1144266508.svg)](https://doi.org/10.5281/zenodo.18698977)
[![Reproducibility Check](https://github.com/aonurakman/demandify/actions/workflows/reproducibility.yml/badge.svg)](https://github.com/aonurakman/demandify/actions/workflows/reproducibility.yml)


# Welcome to demandify!

**Turn real-world traffic data into agent-based SUMO traffic scenarios.**

Do you want to recreate real-world city traffic but don't have access to precious driver trip data? **demandify** solves that.

Pick a spot on the map and demandify will:
1.  Fetch real-time congestion data from TomTom 🗺️
2.  Build a clean SUMO network 🛣️
3.  Use the Genetic Algorithm to figure out the demand pattern to match that traffic 🧬
4. Produces a ready-to-run SUMO scenario in agent-level precision that allows you to test your urban routing policies, even for your CAVs! ([wink](https://github.com/COeXISTENCE-PROJECT/URB) [wink](https://github.com/COeXISTENCE-PROJECT/RouteRL)).

Three integrated web tools are accessible via the top navigation bar:
- 🚦 **Scenario Calibrator** (`/`): Calibrate realistic vehicle demand against live or offline traffic observations.
- 💾 **Dataset Builder** (`/dataset-builder`): Prepare and bundle reusable offline traffic snapshots and networks.
- 🛠️ **Demand Studio** (`/demand-studio`): Inspect, edit, scale, test, and design custom OD traffic scenarios interactively.

![Workflow](https://github.com/aonurakman/demandify/blob/main/static/schema.png?raw=true)

## Features

- 🌍 **Real-world calibration**: Uses TomTom Traffic Flow API for live congestion data
- 📦 **Offline calibration import**: Run from bundled/offline traffic+network snapshots
- 🛠️ **Demand Studio**: Interactive visual editor to inspect, create, scale, and test traffic demand directly in your browser
- 🧭 **Unified navigation**: Easily switch between Scenario Calibrator, Dataset Builder, and Demand Studio
- 🎯 **Seeded & reproducible**: Same seed = identical results for same congestion and bbox
- 🚗 **Car-only SUMO networks**: Automatic OSM → SUMO conversion with car filtering, clean networks
- 🧬 **Genetic algorithm**: Calibrates demand against observed congestion with intervalwise MAE scoring, MAE-elite Pareto selection, teleport filtering, immigrants, assortative mating, deterministic crowding, and adaptive mutation boost
- 💾 **Smart caching**: Content-addressed caching for fast re-runs (traffic snapshots bucketed to 5-minute windows)
- 📊 **Beautiful reports**: HTML reports with visualizations and statistics
- ⌨️ **CLI native**: Live in the terminal? No problem.
- 🖥️ **Clean web UI**: Modern dark theme, interactive Leaflet maps, live metrics, and real-time logs
- ✅ **Data quality labeling**: Feasibility check reports data quality scores and potential risk flags before running

![GUI Screenshot](https://github.com/aonurakman/demandify/blob/main/static/gui.png?raw=true)

## Quickstart

### 1. Install demandify

```bash
# Install from PyPI (Recommended)
pip install demandify
```

If you want to contribute or install from source:
```bash
git clone https://github.com/aonurakman/demandify.git
cd demandify
pip install -e .
```

### 2. Install SUMO 🚦

**demandify** requires SUMO (Simulation of Urban MObility) to power its simulations.

> [!IMPORTANT] 
> demandify is developed and tested with SUMO version 1.26.0. Ensure that your SUMO version is up to date.

👉 **[Download SUMO from the official website](https://eclipse.dev/sumo/)**

Once installed, verify it's working:
```bash
demandify doctor
```

### 3. Get a TomTom API Key

1. Sign up at [https://developer.tomtom.com/](https://developer.tomtom.com/)
2. Create a new app and copy the API key
3. The free tier includes 2,500 requests/day

### 4. Run demandify

```bash
demandify
```

This starts the web server at [http://127.0.0.1:8000](http://127.0.0.1:8000)

### 5. Calibrate a scenario

1. **Choose mode** at the top:
   - `Create`: live TomTom + OSM fetch
   - `Import`: select existing offline dataset (bbox auto-loaded and locked)
2. **Draw a bounding box** on the map (Create mode only)
3. **Configure parameters** (defaults work well):
   - Time window: 15, 30, or 60 minutes
   - Seed: any integer for reproducibility
   - Warmup: a few minutes to populate the network
   - GA population/generations: controls quality vs speed
4. **Paste your API key** (Create mode only; one-time, stored locally)
5. **Click "Start Calibration"**
6. **Watch the progress** through 8 stages
7. **Download your scenario** with `demand.csv`, SUMO network, and report

Before calibration starts, demandify runs a preparation feasibility check and reports:
- fetched traffic segments
- matched observed edges
- total network edges
- data quality label + score + risk flags
   
### 6. Run Headless (Optional) 🤖

You can run the full calibration pipeline directly from the command line, ideal for automation or remote servers.

```bash
# Basic usage (defaults: window=15, pop=50, gen=20)
demandify run "2.2961,48.8469,2.3071,48.8532" --name Paris_Test_01

# Advanced usage with custom parameters
demandify run "2.2961,48.8469,2.3071,48.8532" \
  --name paris_v1 \
  --window 30 \
  --seed 123 \
  --pop 100 \
  --gen 50 \
  --mutation 0.5 \
  --elitism 2

# With advanced GA dynamics
demandify run "2.2961,48.8469,2.3071,48.8532" \
  --name paris_v2 \
  --pop 100 \
  --gen 100 \
  --immigrant-rate 0.05 \
  --stagnation-patience 15

# Fully non-interactive (automation/CI)
demandify run "2.2961,48.8469,2.3071,48.8532" \
  --name paris_v3 \
  --non-interactive

# Import existing offline dataset (no live TomTom/OSM fetch)
demandify run --import krakow_v1 --name krakow_remote
```

> **Note:** By default, the CLI pauses after fetching/matching data and asks for confirmation, then asks whether to run another calibration. Pass `--non-interactive` to auto-approve and exit immediately after pipeline completion.

#### Calibration CLI Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `bbox` | String | Req* | Bounding box (`west,south,east,north`) |
| `--import` | String | None | Use an offline dataset by name (or `source:name`) |
| `--name` | String | Auto | Custom Run ID/Name |
| `--non-interactive` | Flag | off | Disable prompts (auto-approve and exit when pipeline completes) |
| `--window` | Int | 15 | Simulation duration (min) |
| `--warmup` | Int | 5 | Warmup duration before scoring (min) |
| `--seed` | Int | 42 | Random seed |
| `--step-length`| Float | 1.0 | SUMO step length (seconds) |
| `--workers` | Int | Auto (CPU count) | Parallel GA workers |
| `--tile-zoom` | Int | 12 | TomTom vector flow tile zoom |
| `--pop` | Int | 50 | GA Population size |
| `--gen` | Int | 20 | GA Generations |
| `--mutation`| Float | 0.5 | Mutation rate (per individual) |
| `--crossover`| Float| 0.7 | Crossover rate |
| `--elitism` | Int | 2 | Top individuals to keep |
| `--sigma` | Int | 20 | Mutation magnitude (step size) |
| `--indpb` | Float | 0.3 | Mutation probability (per gene) |
| `--max-ods` | Int | 50 | Max OD pairs to generate |
| `--min-connection-paths` | Int | 1 | Minimum number of distinct simple routes required for an OD pair to be eligible during sampling |
| `--initial-population` | Int | 1000 | Target initial number of vehicles (controls sparse initialization) |
| `--capacity-factor` | Float | 1.0 | Effective road-capacity factor (derating for mixed traffic friction, e.g. 0.85–0.90) |
| `--mesosim` / `--no-mesosim` | Flag | on | Mesoscopic simulation during GA candidate evaluation (final run is microscopic) |
| `--topology-guidance` / `--no-topology-guidance` | Flag | on | Use network topology and speed discrepancies to guide GA mutations |
| `--sensor-coverage-od` / `--no-sensor-coverage-od` | Flag | on | Greedy set-cover during OD selection to maximize sensor edge coverage |

\* `bbox` is required in create mode. In import mode, use `--import` and do not pass `bbox`.

`Import` mode constraints:
- positional `bbox` is rejected
- `--tile-zoom` is rejected
- all calibration controls (seed, GA params, warmup/window, etc.) remain available

#### Advanced GA Dynamics

These parameters control diversity mechanisms and adaptive behavior in the genetic algorithm, addressing local optima stagnation and trip count explosion.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--immigrant-rate` | Float | 0.03 | Fraction of random individuals injected per generation (0–1) |
| `--elite-top-pct` | Float | 0.1 | Defines the size of the top-by-MAE elite slice per generation: `n=max(1, elite_top_pct * population)` |
| `--stagnation-patience` | Int | 20 | Generations without improvement before mutation boost activates |
| `--stagnation-boost` | Float | 1.5 | Multiplier for mutation sigma and rate during stagnation |
| `--checkpoint-interval` | Int | 10 | Save best-individual checkpoint artifacts every N generations |
| `--assortative-mating` | Flag | off | Explicitly enable assortative mating |
| `--no-assortative-mating` | Flag | off | Disable assortative mating (dissimilar parent pairing, on by default) |
| `--deterministic-crowding` | Flag | off | Explicitly enable deterministic crowding |
| `--no-deterministic-crowding` | Flag | off | Disable deterministic crowding (diversity-preserving replacement, on by default) |
| `--early-stopping` / `--no-early-stopping` | Flag | off | Stop calibration early if stagnation persists after mutation boost |

All advanced dynamics are **enabled by default** with conservative values. For most use cases, the defaults work well. You can disable features via the corresponding `--no-*` flags or explicitly force-enable them with `--assortative-mating` / `--deterministic-crowding`.

### 7. Build Offline Dataset 💾

If you want a reusable prep bundle (for future no-key workflows), open:

- [http://127.0.0.1:8000/dataset-builder](http://127.0.0.1:8000/dataset-builder)

This dedicated page is separate from calibration runs. It executes preparation only (traffic snapshot + OSM + SUMO network + map matching) and stores files under:

- `demandify_datasets/<dataset_name>/`

Each dataset includes `data/traffic_data_raw.csv`, `data/observed_edges.csv`, `data/map.osm`, `sumo/network.net.xml`, and `dataset_meta.json`.

`dataset_meta.json` now includes a computed data quality block (`score`, `label`, `recommendation`, and metrics) to help decide whether a dataset is strong enough for offline calibration.

Bundled snapshot previews:

| Den Haag (`den_haag_v1`) | Krakow (`krakow_v1`) | Eskisehir (`eskisehir_v1`) |
|---|---|---|
| ![Den Haag offline network](https://github.com/aonurakman/demandify/blob/main/demandify/offline_datasets/den_haag_v1/plots/network.png?raw=true) | ![Krakow offline network](https://github.com/aonurakman/demandify/blob/main/demandify/offline_datasets/krakow_v1/plots/network.png?raw=true) | ![Eskisehir offline network](https://github.com/aonurakman/demandify/blob/main/demandify/offline_datasets/eskisehir_v1/plots/network.png?raw=true) |

### 8. Demand Studio (Interactive Scenario Editor) 🛠️

Want to inspect your trips, try "what-if" traffic experiments, or design a custom traffic scenario from scratch? **Demand Studio** provides an interactive visual workspace directly inside your browser:

- [http://127.0.0.1:8000/demand-studio](http://127.0.0.1:8000/demand-studio) (or click **Studio** in the top navigation bar).

![Demand Studio Screenshot](https://github.com/aonurakman/demandify/blob/main/static/demand_studio.png?raw=true)

#### What you can do:

- 🔍 **Inspect and Edit Origin-Destination (OD) Pairs**:
  - Load trips from any previous calibration run (`demandify_runs/`), offline dataset (`demandify_datasets/` or bundled cities), or saved scenario.
  - Browse every OD pair in the sidebar, filter or search by road ID, and see the exact shortest path drawn on the map.
  - Tweak vehicle flow rates directly or delete unwanted pairs with one click.

- 🛣️ **Interactive Route & Pair Selection**:
  - Click any road on the map to set it as an **Origin** (marked with a green circle).
  - The map highlights all existing destinations linked to that origin, along with their routes.
  - Click an existing destination to view and edit its volume, or click any unserved road to stage a brand-new OD pair.
  - **Smart connectivity guard**: If two selected roads cannot connect (for example, due to one-way streets or separated ramps), demandify lets you know right away and prevents creating broken trips.

- 📈 **Global Demand Multiplier**:
  - Test lighter off-peak hours or heavy rush-hour conditions with a single slider.
  - Scale total network demand from **0.1x** to **2.0x**.
  - All OD pairs are smoothly adjusted to whole vehicle counts while keeping active routes alive.

- 🚦 **Fast In-Browser SUMO Testing**:
  - Click **"SUMO Test"** to run a quick simulation right in the browser (15, 30, or 60 minute test windows).
  - Check live metrics on the sidebar:
    - **Inserted vehicles**: total vehicles added to the network
    - **Completion rate**: percentage of trips that reached their destination
    - **Teleports**: count of jammed vehicles (lower is better!)
    - **Average speed**: overall network speed in km/h
  - Toggle between **Flow Bandwidth** (line thickness shows vehicle volumes) and **Congestion Heatmap** (green = free flow, orange/red = congested).

- 🗺️ **Fetch Clean Networks from OpenStreetMap**:
  - Want to build demand from scratch without needing a TomTom key?
  - Select **"Fetch from OpenStreetMap"** in the scenario dropdown.
  - Enter a bounding box and a name: demandify downloads the OSM map and builds a clean SUMO road network ready for you to place vehicles.

- 💾 **Export Ready-to-Run Scenarios**:
  - Hit **"Save Scenario"** to export your modified scenario under `demandify_scenarios/<scenario_name>/`.
  - It generates all ready-to-run SUMO files: `network.net.xml`, `trips.xml`, `scenario.sumocfg`, and `demand.csv`.

## How It Works

demandify follows a multi-stage pipeline:

1. **Validate inputs** - Check mode/parameters and feasibility
2. **Preparation**:
   - `Create`: fetch traffic + OSM, build network, match edges
   - `Import`: load/copy network + observed traffic files from offline dataset
3. **Initialize demand** - Select routable OD pairs (lane-permission aware)
4. **Calibrate demand** - Run GA to optimize per-OD vehicle insertion rates against observed edge-speed error
5. **Export scenario** - Generate `demand.csv`, `trips.xml`, config, and report

### Advanced GA Dynamics

The genetic algorithm includes several mechanisms to avoid common pitfalls like local optima stagnation and trip count explosion:

- **MAE-elite Pareto parent selection**: Individuals are first ordered by `mae`, and the top slice (`n=max(1, elite_top_pct * population)`) becomes the elite pool. If that pool contains any zero-teleport candidates, teleporting candidates are discarded. The remaining elite is Pareto-ranked on `(failure_rate, missing_edges, magnitude)` when teleports are all zero, or on `(teleports, failure_rate, missing_edges, magnitude)` otherwise.
- **Random immigrants**: A small fraction of completely random individuals is injected each generation to maintain genetic diversity and escape local optima.
- **Assortative mating**: Parents are paired by dissimilarity (by genome magnitude) for crossover, promoting exploration of the search space.
- **Deterministic crowding**: Offspring compete with similar parents for population slots, preserving niche diversity.
- **Adaptive mutation boost**: If the best fitness stagnates for K generations, mutation sigma and rate are temporarily increased by a configurable multiplier. They reset automatically when improvement resumes.

Parent choice, survival elitism, per-generation representatives, and the final returned solution all follow that same **MAE-elite Pareto rule** across generations, so the returned individual still comes from the strongest MAE frontier while preferring lower teleports, lower failure rate, fewer missing edges, and lower total demand inside that frontier.

The primary loss itself is still MAE, but it is computed interval-by-interval across the post-warmup measurement windows. For each observed edge and measurement interval, demandify compares the simulated speed to the observed speed; if an observed edge has no simulated speed in that interval, it falls back to the matched SUMO edge free-flow speed.

The calibration report includes plots for **genotypic diversity** (mean pairwise L2 distance) and **phenotypic diversity** (σ of fitness values) across generations, along with markers indicating when mutation boost was active.

### Variability & Consistency
      
While demandify uses seeding (random seed) for all internal stochastic operations (OD selection, GA evolution), **perfect reproducibility is not guaranteed** due to the inherently chaotic nature of traffic microsimulation (SUMO) and real-time data inputs.
      
Seeding ensures *consistency* (runs look similar), but small timing differences in OS scheduling or dynamic routing decisions can lead to divergent outcomes. Traffic snapshots are cached in 5-minute buckets; using the same seed, bbox, and time bucket will reproduce demand.csv and SUMO randomness.
      
### Caching
      
demandify caches:
- OSM extracts (by bbox)
- SUMO networks (by bbox + conversion params)
- Traffic snapshots (by bbox + provider + style + tile zoom + 5-minute timestamp bucket)
- Map matching results (by bbox + network key + provider + timestamp bucket)
      
Cache location: `~/.demandify/cache/`

Clear cache: `demandify cache clear`

## CLI Commands

```bash
# Start web server (default)
demandify

# Run headless calibration
demandify run "west,south,east,north"

# Run headless from bundled offline dataset
demandify run --import krakow_v1

# Check system requirements
demandify doctor

# Set TomTom API key (CLI)
demandify set-key YOUR_KEY_HERE

# Clear cache
demandify cache clear

# Show version
demandify --version
```

## Output Files & Directories

### Calibration Runs (`demandify_runs/run_<timestamp>/`)

Each calibration run creates a folder with:

- **`demand.csv`** - Travel demand with exact schema: `ID`, `origin link id`, `destination link id`, `departure timestep`
- **`trips.xml`** - SUMO trips file
- **`network.net.xml`** - SUMO network
- **`scenario.sumocfg`** - Ready-to-run SUMO configuration file (configured with default route resilience)
- **`observed_edges.csv`** - Speed observations mapped to SUMO edge IDs
- **`run_meta.json`** - Complete run metadata with fitness scores and best-candidate diagnostics
- **`report.html`** - Standalone HTML calibration report with interactive charts and metrics
- **`latest_selected/`** - Rolling recovery export kept up-to-date across generations
- **`<run_id>/`** - URB/RouteRL-compatible export bundle

Run the calibrated scenario:
```bash
cd demandify_runs/run_<timestamp>/sumo
sumo-gui -c scenario.sumocfg
```

### Demand Studio Scenarios (`demandify_scenarios/<scenario_name>/`)

When you save a scenario from Demand Studio, it exports a standalone bundle ready to run:

- **`sumo/network.net.xml`** - Road network geometry
- **`sumo/trips.xml`** - Vehicle departure trips
- **`sumo/scenario.sumocfg`** - Ready-to-run SUMO scenario configuration
- **`data/demand.csv`** - Complete OD demand table
- **`scenario_meta.json`** - Scenario summary, vehicle counts, and test telemetry

### Offline Datasets (`demandify_datasets/<dataset_name>/`)

Datasets generated via the Dataset Builder store:

- **`data/traffic_data_raw.csv`** - Raw TomTom congestion observations
- **`data/observed_edges.csv`** - Map-matched edge speeds
- **`data/map.osm`** - Raw OpenStreetMap road data
- **`sumo/network.net.xml`** - Converted car-only SUMO network
- **`dataset_meta.json`** - Feasibility metrics, data quality score, and bounding box metadata

## Configuration

### API Keys

Three ways to provide your TomTom API key:

1. **Web UI**: Paste in the form (saved to `~/.demandify/config.json`)
2. **Environment variable**: `export TOMTOM_API_KEY=your_key`
3. **`.env` file**: Copy `.env.example` to `.env` and add your key
4. **CLI**: `demandify set-key YOUR_KEY` stores it in `~/.demandify/config.json`

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black demandify/

# Lint
ruff check demandify/
```

## License

MIT

## Acknowledgments

- **SUMO**: [Eclipse SUMO](https://eclipse.dev/sumo/)
- **TomTom**: [Traffic Flow API](https://developer.tomtom.com/traffic-api)
- **OpenStreetMap**: [© OpenStreetMap contributors](https://www.openstreetmap.org/copyright)

## Citation

If you use this software for your research, please consider using the citation below.
The canonical metadata for GitHub's "Cite this repository" is in `CITATION.cff`.

```bibtex
@software{demandify_2026,
  author       = {{Ahmet Onur Akman}},
  title        = {{demandify: Calibrate SUMO traffic scenarios against real-world congestion using genetic algorithms}},
  year         = {2026},
  version      = {0.0.6},
  publisher    = {PyPI},
  url          = {https://pypi.org/project/demandify/},
  repository   = {https://github.com/aonurakman/demandify},
  doi          = {10.5281/zenodo.19050050}
}
```
