![demandify](static/banner.png)

# Welcome to demandify!

> [!CAUTION]
> **BETA VERSION**: This project is currently in early beta (0.0.1). Expect frequent breaking changes and instability.
> Use at your own risk.

**Turn real-world traffic data into accurate SUMO simulations.**

Have you ever wanted to create a realistic traffic simulation but got stuck figuring out where the cars should go? **demandify** solves that.

It's simple: Pick a spot on the map and demandify will:
1.  Fetch real-time congestion data from TomTom 🗺️
2.  Build a clean SUMO network 🛣️
3.  Use the Genetic Algorithm to figure out the demand pattern to match that traffic 🧬

The result? A ready-to-run SUMO scenario that allows you to test your urban routing policies, even for your CAVs! ([wink](https://github.com/COeXISTENCE-PROJECT/URB) [wink](https://github.com/COeXISTENCE-PROJECT/RouteRL)).

## Features

- 🌍 **Real-world calibration**: Uses TomTom Traffic Flow API for live congestion data
- 🎯 **Seeded & reproducible**: Same seed = identical results for same congestion and bbox
- 🚗 **Car-only SUMO networks**: Automatic OSM → SUMO conversion with car filtering
- 🧬 **Genetic algorithm**: Optimizes demand to match observed speeds
- 💾 **Smart caching**: Content-addressed caching for fast re-runs
- 📊 **Beautiful reports**: HTML reports with visualizations and statistics
- 🖥️ **Clean web UI**: Leaflet map, real-time progress stepper, log console

## Quickstart

### 1. Install demandify

```bash
# Clone the repository
git clone <repository-url>
cd demandify

# Install in development mode
pip install -e .
```

### 2. Install SUMO 🚦

**demandify** requires SUMO (Simulation of Urban MObility) to power its simulations.

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

1. **Draw a bounding box** on the map
2. **Configure parameters** (defaults work well):
   - Time window: 15 or 30 minutes
   - Seed: any integer for reproducibility
   - GA population/generations: controls quality vs speed
3. **Paste your API key** (one-time, stored locally)
4. **Click "Start Calibration"**
5. **Watch the progress** through 8 stages
6. **Download your scenario** with `demand.csv`, SUMO network, and report
   
### 6. Run Headless (Optional) 🤖

You can run the full calibration pipeline directly from the command line, ideal for automation or remote servers.

```bash
# Basic usage (defaults: window=15, pop=50, gen=20)
demandify run "2.2961,48.8469,2.3071,48.8532" --name Paris_Test_01

# Advanced usage with custom parameters
demandify run "2.2961,48.8469,2.3071,48.8532" \
  --name Paris_Optimization \
  --window 30 \
  --seed 123 \
  --pop 100 \
  --gen 50 \
  --mutation 0.5 \
  --elitism 2
```

> **Note:** The CLI will pause after fetching/matching data to show matching statistics and ask for confirmation before starting the intensive calibration.

#### Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `bbox` | String | (Req) | Bounding box (`west,south,east,north`) |
| `--name` | String | Auto | Custom Run ID/Name |
| `--window` | Int | 15 | Simulation duration (min) |
| `--seed` | Int | 42 | Random seed |
| `--pop` | Int | 50 | GA Population size |
| `--gen` | Int | 20 | GA Generations |
| `--mutation`| Float | 0.5 | Mutation rate (per individual) |
| `--crossover`| Float| 0.7 | Crossover rate |
| `--elitism` | Int | 2 | Top individuals to keep |
| `--sigma` | Int | 20 | Mutation magnitude (step size) |
| `--indpb` | Float | 0.3 | Mutation probability (per gene) |
| `--origins` | Int | 10 | Number of origin candidates |
| `--destinations` | Int | 10 | Number of destination candidates |
| `--max-ods` | Int | 50 | Max OD pairs to generate |
| `--bin-size` | Int | 5 | Time bin size in minutes |
| `--initial-population` | Int | 1000 | Target initial number of vehicles (controls sparse initialization) |

## How It Works

demandify follows an 8-stage pipeline:

1. **Validate inputs** - Check bbox, parameters, API key
2. **Fetch traffic snapshot** - Get real-time speeds from TomTom
3. **Fetch OSM extract** - Download road network data
4. **Build SUMO network** - Convert OSM to car-only SUMO `.net.xml`
5. **Map matching** - Match traffic segments to SUMO edges
6. **Initialize demand** - Select origin/destination pairs and time bins
7. **Calibrate demand** - Run GA to optimize vehicle counts
8. **Export scenario** - Generate `demand.csv`, routes, config, and report

### Variability & Consistency
      
While demandify uses seeding (random seed) for all internal stochastic operations (OD selection, GA evolution), **perfect reproducibility is not guaranteed** due to the inherently chaotic nature of traffic microsimulation (SUMO) and real-time data inputs.
      
Seeding ensures *consistency* (runs look similar), but small timing differences in OS scheduling or dynamic routing decisions can lead to divergent outcomes.
      
### Caching
      
demandify caches:
- OSM extracts (by bbox)
- SUMO networks (by bbox + conversion params)
- Traffic snapshots (by bbox + timestamp bucket)
- Map matching results
      
Cache location: `~/.demandify/cache/`

Clear cache: `demandify cache clear`

## CLI Commands

```bash
# Start web server (default)
demandify

# Run headless calibration
demandify run "west,south,east,north"

# Check system requirements
demandify doctor

# Clear cache
demandify cache clear

# Show version
demandify --version
```

## Output Files

Each run creates a folder with:

- **`demand.csv`** - Travel demand with exact schema:
  - `ID`, `origin link id`, `destination link id`, `departure timestep`
- **`trips.xml`** - SUMO trips file
- **`routes.rou.xml`** - Routed SUMO routes
- **`network.net.xml`** - SUMO network
- **`scenario.sumocfg`** - SUMO configuration (ready to run)
- **`observed_edges.csv`** - Observed traffic speeds
- **`run_meta.json`** - Complete run metadata
- **`report.html`** - Calibration report with visualizations

Run the scenario:
```bash
cd demandify_runs/run_<timestamp>
sumo-gui -c scenario.sumocfg
```

## Configuration

### API Keys

Three ways to provide your TomTom API key:

1. **Web UI**: Paste in the form (saved to `~/.demandify/config.json`)
2. **Environment variable**: `export TOMTOM_API_KEY=your_key`
3. **`.env` file**: Copy `.env.example` to `.env` and add your key

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
