"""
Main calibration pipeline orchestrator.
Ties together all components to execute the full workflow.
"""
from pathlib import Path
from typing import Tuple, Dict, List
from datetime import datetime
import pandas as pd
import numpy as np
import logging

from demandify.utils.logger import setup_logging

from demandify.config import get_config
from demandify.providers.tomtom import TomTomProvider
from demandify.providers.osm import OSMFetcher
from demandify.sumo.network import convert_osm_to_sumo, SUMONetwork
from demandify.sumo.matching import EdgeMatcher
from demandify.sumo.demand import DemandGenerator
from demandify.sumo.simulation import SUMOSimulation
from demandify.calibration.objective import EdgeSpeedObjective
from demandify.calibration.optimizer import GeneticAlgorithm
from demandify.calibration.worker import run_simulation_worker, SimulationConfig, evaluate_for_ga
from functools import partial
from demandify.cache.manager import CacheManager
from demandify.cache.keys import bbox_key, osm_key, network_key, traffic_key
from demandify.export.exporter import ScenarioExporter
from demandify.export.report import ReportGenerator
from demandify.utils.visualization import plot_network_geometry
from demandify.export.custom_formats import URBDataExporter
import shutil

logger = logging.getLogger(__name__)


class CalibrationPipeline:
    """Main calibration pipeline."""
    
    def __init__(
        self,
        bbox: Tuple[float, float, float, float],
        window_minutes: int,
        seed: int,
        ga_population: int = 50,
        ga_generations: int = 20,
        ga_mutation_rate: float = 0.5,
        ga_crossover_rate: float = 0.7,
        ga_elitism: int = 2,
        ga_mutation_sigma: int = 20,
        ga_mutation_indpb: float = 0.3,
        num_origins: int = 10,
        num_destinations: int = 10,
        max_od_pairs: int = 1000,
        bin_minutes: float = 1.0,
        initial_population: int = 1000,
        output_dir: Path = None,
        run_id: str = None,
        progress_callback: callable = None
    ):
        """
        Initialize pipeline.
        
        Args:
            bbox: (west, south, east, north)
            window_minutes: Simulation window in minutes
            seed: Random seed
            ga_population: GA population size
            ga_generations: GA generations
            num_origins: Number of origin candidates
            num_destinations: Number of destination candidates
            max_od_pairs: Maximum number of OD pairs to generate
            bin_minutes: Duration of each demand time bin in minutes
            initial_population: Target initial number of vehicles (controls GA init bounds)
            output_dir: Output directory for results
            run_id: Optional custom identifier for the run
            progress_callback: Optional callable(stage, name, msg, level) for UI updates
        """
        self.bbox = bbox
        self.window_minutes = window_minutes
        self.warmup_minutes = 5
        self.seed = seed
        self.ga_population = ga_population
        self.ga_generations = ga_generations
        self.ga_mutation_rate = ga_mutation_rate
        self.ga_crossover_rate = ga_crossover_rate
        self.ga_elitism = ga_elitism
        self.ga_mutation_sigma = ga_mutation_sigma
        self.ga_mutation_indpb = ga_mutation_indpb
        self.num_origins = num_origins
        self.num_destinations = num_destinations
        self.max_od_pairs = max_od_pairs
        self.bin_minutes = bin_minutes
        self.initial_population = initial_population
        self.run_id = run_id
        
        if output_dir is None:
            if run_id:
                # Use provided run ID
                run_name = f"run_{run_id}"
            else:
                # Generate timestamp-based ID
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = f"run_{timestamp}"
                self.run_id = timestamp
            
            output_dir = Path.cwd() / "demandify_runs" / run_name
        
        self.min_trip_distance = None  # Will be calculated adaptively
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "sumo").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        
        # Setup file logging for this run
        self._setup_run_logging()
        
        # Cache and config
        self.config = get_config()
        self.cache_manager = CacheManager(self.config.cache_dir)
        
        self.progress_callback = progress_callback
        
        logger.info(f"Pipeline initialized: bbox={bbox}, seed={seed}, run_id={self.run_id}")
    
    def _report_progress(self, stage: int, name: str, msg: str, level: str = "info"):
        """Report progress via callback if available."""
        if self.progress_callback:
            try:
                self.progress_callback(stage, name, msg, level)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
        
        # Also log key events
        # The logger is now configured to output to console gracefully
        if level == "info":
            logger.info(f"[{stage}] {name}: {msg}")
        elif level == "error":
            logger.error(f"[{stage}] {name}: {msg}")
        elif level == "warning":
            logger.warning(f"[{stage}] {name}: {msg}")
    
    def _setup_run_logging(self):
        """Setup file logging for this specific run."""
        log_file = "pipeline.log"
        setup_logging(
            run_dir=self.output_dir / "logs", 
            log_file=log_file,
            level=logging.INFO
        )
        
        # Copy to latest.log for easy tailing
        try:
            latest_link = Path("demandify_runs/latest.log")
            if latest_link.exists():
                latest_link.unlink()
            # On Windows, symlinks require admin, so we skip or copy
            # On Mac/Linux:
            # latest_link.symlink_to(log_file)
        except Exception:
            pass
    
    async def prepare(self) -> Dict:
        """
        Run preparation stages (1-4): Fetch data and match to network.
        
        Returns:
            Context dictionary required for calibration
        """
        self._report_progress(0, "Initializing", "Starting preparation phase")
        
        # Stage 1: Fetch traffic data
        self._report_progress(1, "Fetching Traffic Data", "Connecting to TomTom API...")
        traffic_df = await self._fetch_traffic_data()
        
        # Save raw traffic data
        traffic_data_file = self.output_dir / "data" / "traffic_data_raw.csv"
        traffic_df.to_csv(traffic_data_file, index=False)
        self._report_progress(1, "Fetching Traffic Data", f"✓ Fetched {len(traffic_df)} traffic segments")
        
        # Stage 2: Fetch OSM data
        self._report_progress(2, "Download OSM", "Downloading OpenStreetMap data...")
        osm_cache_path = await self._fetch_osm_data()
        osm_file = self.output_dir / "data" / "map.osm"
        shutil.copy2(osm_cache_path, osm_file)
        self._report_progress(2, "Download OSM", "✓ OSM data downloaded")
        
        # Stage 3: Build SUMO network
        self._report_progress(3, "Building Network", "Converting map to SUMO network...")
        # Build/Get from cache using original OSM path (key based)
        network_cache_path = self._build_sumo_network(osm_cache_path)
        
        # Copy to run folder
        network_file = self.output_dir / "sumo" / "network.net.xml"
        shutil.copy2(network_cache_path, network_file)
        
        # Count edges and plot map
        net = SUMONetwork(network_file)
        total_edges = len(net.edges)
        plot_network_geometry(network_file, self.output_dir / "plots" / "network.png")
        self._report_progress(3, "Building Network", "✓ SUMO network created")
        
        # Cleanup OSM file to save space
        try:
            if osm_file.exists():
                osm_file.unlink()
                logger.debug(f"Deleted {osm_file} to save space")
        except Exception as e:
            logger.warning(f"Failed to delete OSM file: {e}")
        
        # Stage 4: Map matching
        self._report_progress(4, "Matching Traffic", "Matching traffic data to road network...")
        observed_edges = self._match_traffic_to_edges(traffic_df, network_file)
        
        # Save observed edges
        observed_edges_file = self.output_dir / "data" / "observed_edges.csv"
        observed_edges.to_csv(observed_edges_file, index=False)
        self._report_progress(4, "Matching Traffic", f"✓ Matched {len(observed_edges)} traffic segments to SUMO edges")
        
        return {
            "traffic_df": traffic_df,
            "osm_file": osm_file,
            "network_file": network_file,
            "observed_edges": observed_edges,
            "traffic_data_file": traffic_data_file,
            "observed_edges_file": observed_edges_file,
            "total_edges": total_edges
        }

    def calibrate(self, context: Dict) -> Dict:
        """
        Run calibration stages (5-8) using preparation context.
        
        Args:
            context: Dictionary returned by prepare()
            
        Returns:
            Metadata dict with results
        """
        self._report_progress(5, "Init Demand", "Starting calibration phase")
        
        # Unpack context
        network_file = context["network_file"]
        observed_edges = context["observed_edges"]
        traffic_data_file = context["traffic_data_file"]
        observed_edges_file = context["observed_edges_file"]
        
        # Abort if no edges matched (critical check)
        if len(observed_edges) == 0:
            error_msg = "No traffic sensors matched in this area. Cannot calibrate demand."
            self._report_progress(5, "No Observed Edges", error_msg, level="error")
            raise RuntimeError(error_msg)
        
        # Stage 5: Initialize demand model
        self._report_progress(5, "Init Demand", "Initializing demand generation...")
        demand_gen, od_pairs, departure_bins = self._initialize_demand(network_file)
        self._report_progress(5, "Init Demand", f"✓ Created {len(od_pairs)} OD pairs")
        
        # Stage 6: Calibrate demand
        self._report_progress(6, "Calibrating Demand", 
                             f"Running genetic algorithm ({self.ga_generations} generations)...")
        best_genome, best_loss, loss_history = self._calibrate_demand(
            demand_gen, od_pairs, departure_bins, observed_edges, network_file
        )
        self._report_progress(6, "Calibrating Demand", f"✓ Complete: loss={best_loss:.2f} km/h")
        
        # Stage 7: Generate final demand files
        self._report_progress(7, "Generating Demand", "Creating final trip files...")
        demand_csv, trips_file = self._generate_final_demand(
            demand_gen, best_genome, od_pairs, departure_bins, network_file
        )
        self._report_progress(7, "Generating Demand", "✓ Files created")
        
        # Stage 8: Export scenario and report
        self._report_progress(8, "Exporting Results", "Running final simulation and generating reports...")
        simulated_speeds = self._run_final_simulation(network_file, trips_file)
        
        # Calculate final metrics
        if len(observed_edges) > 0:
            objective = EdgeSpeedObjective(observed_edges)
            quality_metrics = objective.calculate_metrics(simulated_speeds)
            
            # Log observed edge coverage
            observed_edge_ids = set(observed_edges['edge_id'])
            simulated_edge_ids = set(simulated_speeds.keys())
            matched = observed_edge_ids & simulated_edge_ids
            missing = observed_edge_ids - simulated_edge_ids
            
            logger.debug(f"📊 Edge coverage: {len(matched)}/{len(observed_edge_ids)} observed edges have traffic")
            if missing:
                logger.warning(f"⚠️  Missing traffic on observed edges: {sorted(missing)}")
                
        else:
            quality_metrics = {
                'mae': None, 'mse': None, 'matched_edges': 0,
                'missing_edges': 0, 'total_edges': 0
            }
        
        
        metadata = self._export_results(
            network_file, demand_csv, trips_file,
            observed_edges_file, traffic_data_file, observed_edges,
            simulated_speeds, best_loss, loss_history, quality_metrics
        )
        
        # Note: With dynamic routing, we don't generate routes.rou.xml
        # SUMO routes trips on-the-fly, so no cleanup of route files needed
        logger.debug("Using dynamic routing - no intermediate route files to clean up")
            
        logger.info("Pipeline complete!")
        
        return metadata

    async def run(self, confirm_callback=None) -> Dict:
        """
        Run the full calibration pipeline.
        
        Args:
            confirm_callback: Optional function(stats) -> bool. 
                              Called after preparation. If returns False, aborts run.
        
        Returns:
            Metadata dict with results or None if aborted
        """
        # Phase 1: Prepare
        context = await self.prepare()
        
        # Confirmation hook
        if confirm_callback:
            traffic_count = len(context["traffic_df"])
            matched_count = len(context["observed_edges"])
            
            stats = {
                "fetched_segments": traffic_count,
                "matched_edges": matched_count,
                "total_network_edges": context.get("total_edges", 0)
            }
            
            should_proceed = confirm_callback(stats)
            if not should_proceed:
                logger.info("🚫 Run aborted by user.")
                return None
                
        # Phase 2: Calibrate
        return self.calibrate(context)
    
    async def _fetch_traffic_data(self) -> pd.DataFrame:
        """Fetch traffic data from TomTom."""
        # Check if already fetched (idempotency for UI confirmation flow)
        traffic_data_file = self.output_dir / "data" / "traffic_data_raw.csv"
        if traffic_data_file.exists():
            logger.info("Using existing traffic data from run directory")
            df = pd.read_csv(traffic_data_file)
            # Ensure geometry column is evaluated if string
            if 'geometry' in df.columns and isinstance(df.iloc[0]['geometry'], str):
                import ast
                df['geometry'] = df['geometry'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            return df

        if not self.config.tomtom_api_key:
            raise RuntimeError("TomTom API key not configured")
        
        provider = TomTomProvider(self.config.tomtom_api_key)
        
        try:
            traffic_df = await provider.fetch_traffic_snapshot(self.bbox)
            logger.debug(f"Fetched {len(traffic_df)} traffic segments")
            return traffic_df
        finally:
            await provider.close()
    
    async def _fetch_osm_data(self) -> Path:
        """Fetch OSM data."""
        cache_key_bbox = bbox_key(*self.bbox)
        cache_key_osm = osm_key(cache_key_bbox)
        osm_file = self.cache_manager.get_osm_path(cache_key_osm)
        
        if self.cache_manager.exists(osm_file):
            logger.debug(f"Using cached OSM data: {osm_file}")
            return osm_file
        
        fetcher = OSMFetcher()
        
        try:
            await fetcher.fetch_osm_data(self.bbox, osm_file)
            return osm_file
        finally:
            await fetcher.close()
    
    def _build_sumo_network(self, osm_file: Path) -> Path:
        """Build SUMO network from OSM."""
        cache_key_bbox = bbox_key(*self.bbox)
        cache_key_osm = osm_key(cache_key_bbox)
        cache_key_net = network_key(cache_key_osm, car_only=True, seed=self.seed)
        network_file = self.cache_manager.get_network_path(cache_key_net)
        
        if self.cache_manager.exists(network_file):
            logger.debug(f"Using cached SUMO network: {network_file}")
            return network_file
        
        network_file, metadata = convert_osm_to_sumo(
            osm_file, network_file, car_only=True, seed=self.seed
        )
        
        return network_file
    
    def _match_traffic_to_edges(
        self, traffic_df: pd.DataFrame, network_file: Path
    ) -> pd.DataFrame:
        """Match traffic segments to SUMO edges."""
        # Check if already matches (idempotency)
        observed_edges_file = self.output_dir / "data" / "observed_edges.csv"
        if observed_edges_file.exists():
            logger.info("Using existing observed edges from run directory")
            return pd.read_csv(observed_edges_file)
            
        network = SUMONetwork(network_file)
        matcher = EdgeMatcher(network, network_file)  # Pass file for projection info
        
        observed_edges = matcher.match_traffic_data(traffic_df, min_confidence=0.1)
        
        logger.debug(f"Matched {len(observed_edges)} edges")
        
        return observed_edges
    
    def _initialize_demand(
        self, network_file: Path
    ) -> Tuple[DemandGenerator, List[Tuple[str, str]], List[Tuple[int, int]]]:
        """Initialize demand generator and select OD pairs."""
        network = SUMONetwork(network_file)
        demand_gen = DemandGenerator(network, seed=self.seed)
        
        # Calculate adaptive minimum trip distance
        # Heuristic: 10% of the bounding box diagonal
        # This prevents picking origin/dest that are practically neighbors
        w, s, e, n = self.bbox
        # Very rough approximation of meters (lat/lon degrees to meters)
        # Using 111km per degree lat, and ~75km per degree lon at 48N
        dx = (e - w) * 75000.0
        dy = (n - s) * 111000.0
        diag = (dx*dx + dy*dy)**0.5
        
        # Adaptive min distance: max(200m, 10% of diagonal)
        # But cap it at 2km for very large maps to avoid filtering too much
        self.min_trip_distance = min(2000.0, max(200.0, diag * 0.10))
        
        logger.info(f"Network diagonal ~{int(diag)}m. Using min_trip_distance={int(self.min_trip_distance)}m")
        
        # Select OD pairs (now validates each pair individually)
        origins, destinations = demand_gen.select_od_candidates(
            num_origins=self.num_origins,
            num_destinations=self.num_destinations,
            max_od_pairs=self.max_od_pairs,  # Pass target to ensure even distribution
            min_trip_distance=self.min_trip_distance
        )
        
        # Create OD pairs from validated origins/destinations
        # Note: origins/destinations now contain only edges used in validated pairs
        od_pairs = [(o, d) for o in origins for d in destinations if o != d]
        
        # Limit if needed (should rarely exceed since we controlled it in select_od_candidates)
        if len(od_pairs) > self.max_od_pairs:
            # We shuffle first to avoid biasing towards the first found pairs
            np.random.RandomState(self.seed).shuffle(od_pairs)
            od_pairs = od_pairs[:self.max_od_pairs]
            logger.info(f"Limited to {self.max_od_pairs} OD pairs from {len(origins)}×{len(destinations)} combinations")
        
        # Create departure bins - cover ENTIRE duration (warmup + window)
        # We start from t=0 to populate the network during warmup
        warmup_sec = self.warmup_minutes * 60
        window_sec = self.window_minutes * 60
        total_duration = warmup_sec + window_sec
        
        # Calculate bins based on bin_minutes (supporting floats)
        target_bin_duration = int(self.bin_minutes * 60)
        if target_bin_duration < 1:
            target_bin_duration = 1
            
        num_bins = max(1, int(round(total_duration / target_bin_duration)))
        
        departure_bins = []
        for i in range(num_bins):
            start = i * target_bin_duration
            end = i * target_bin_duration + target_bin_duration
            # Adjust last bin to match exactly
            if i == num_bins - 1:
                end = total_duration
            departure_bins.append((start, end))
        
        logger.info(f"Created {len(od_pairs)} OD pairs and {len(departure_bins)} departure bins (duration={target_bin_duration}s)")
        
        return demand_gen, od_pairs, departure_bins
    


    def _calibrate_demand(
        self,
        demand_gen: DemandGenerator,
        od_pairs: List[Tuple[str, str]],
        departure_bins: List[Tuple[int, int]],
        observed_edges: pd.DataFrame,
        network_file: Path
    ) -> Tuple[np.ndarray, float, List[float]]:
        """Calibrate demand using GA."""
        
        # Handle case where no edges were matched
        if len(observed_edges) == 0:
            logger.warning("No observed edges matched - skipping calibration, using random demand")
            genome_size = len(od_pairs) * len(departure_bins)
            random_genome = np.random.RandomState(self.seed).randint(0, 10, size=genome_size)
            return random_genome, float('inf'), [float('inf')]
        
        # Create SimulationConfig for the worker
        sim_config = SimulationConfig(
            run_id=self.run_id,
            network_file=network_file,
            od_pairs=od_pairs,
            departure_bins=departure_bins,
            observed_edges=observed_edges,
            warmup_time=self.warmup_minutes * 60,
            simulation_time=(self.warmup_minutes + self.window_minutes) * 60,
            step_length=1.0,
            debug=False,  # Can be exposed via config
            output_base_dir=self.output_dir / "temp_eval"
        )
        
        # Create picklable evaluation function using partial
        # This binds 'config' to run_simulation_worker, leaving 'genome' as the only required arg
        evaluate_func = partial(run_simulation_worker, config=sim_config)
        
        # Note: We rely on the worker to handle demand generation and simulation.
        # The worker returns (loss, metrics). 
        # But GA expects a function that returns float loss.
        # run_simulation_worker returns Tuple[float, Dict].
        # We need an adapter that returns just float, but partial only binds args.
        # We need a top-level wrapper or the worker itself to return float.
        # 
        # Wait! The worker returns (loss, metrics). GA expects float.
        # Code in optimizer.py:
        #   def fitness_wrapper(ind):
        #       return (evaluate_func(np.array(ind)),)
        # 
        # If evaluate_func returns (loss, metrics), then fitness_wrapper returns ((loss, metrics),)
        # This will break DEAP which expects (float,).
        #
        # FIX: The evaluate_func passed to GA must return float.
        # Since I can't use a lambda (not picklable), I need a wrapper function defined at module level 
        # or verify if worker returns just loss.
        # My worker returns (loss, metrics).
        # I should assume the worker returns just loss, OR use a wrapper.
        #
        # Let's import a wrapper from worker.py if possible, or define one here.
        # Actually, let's use the `evaluate_genome_wrapper` matching signature.
        # 
        # BETTER: Modify worker.py to have a function `evaluate_loss_only` for the GA.
        # OR simply change run_simulation_worker to return just loss? No, we want metrics.
        #
        # Strategy: Define `evaluate_adapter` top-level in `pipeline.py` or import it.
        # Defining it top-level in pipeline.py is tricky if pipeline.py imports worker.
        # Let's use `run_simulation_worker` but I need to discard the metrics.
        #
        # I'll create `evaluate_genome_for_ga` in `pipeline.py` (top-level) or `worker.py`.
        # Putting it in `worker.py` is cleanest. 
        # I'll edit `worker.py` in a moment to add `evaluate_for_ga`.
        # For now, let's assume `run_simulation_worker` returns        # Define parameter bounds (trips per OD pair per bin)
        # 0 to 3 trips per OD pair per 1-minute bin
        # Initialization with (0, 100) creates ~500k trips -> Gridlock
        
    # ... continuing with valid code ...
    
    # Run GA
        genome_size = len(od_pairs) * len(departure_bins)
        # Dynamic Bounds & Init Prob Logic
        avg_trips_per_gene = self.initial_population / max(1, genome_size)
        upper_bound = max(1, int(avg_trips_per_gene * 2))
        bounds = (0, upper_bound)
        
        avg_val_if_active = (bounds[0] + bounds[1]) / 2.0
        init_prob = None
        if avg_trips_per_gene < avg_val_if_active:
             init_prob = avg_trips_per_gene / max(0.1, avg_val_if_active)
             init_prob = min(1.0, max(0.001, init_prob)) # Clamp
        
        dynamic_sigma = max(1, int(upper_bound * 0.2))
        
        logger.info(f"Dynamic GA Initialization: Target {self.initial_population} vehicles -> Bounds {bounds} (Avg {avg_trips_per_gene:.2f}/gene)")

        ga = GeneticAlgorithm(
            genome_size=genome_size,
            seed=self.seed,
            bounds=bounds,
            population_size=self.ga_population,
            num_generations=self.ga_generations,
            mutation_rate=self.ga_mutation_rate,
            crossover_rate=self.ga_crossover_rate,
            elitism=self.ga_elitism,
            mutation_sigma=dynamic_sigma,
            mutation_indpb=self.ga_mutation_indpb,
            num_workers=self.config.default_parallel_workers,
            init_prob=init_prob
        )
        
        # Progress callback
        def progress_callback(gen: int, best_loss: float, mean_loss: float):
            msg = f"  🔄 GA Gen {gen}/{self.ga_generations}: best_loss={best_loss:.2f} km/h, mean={mean_loss:.2f}"
            logger.info(msg)
            # print(msg, flush=True) # logger handles this now
        
        # Start optimization
        from demandify.calibration.worker import evaluate_for_ga
        evaluate_func_clean = partial(evaluate_for_ga, config=sim_config)
        
        best_genome, best_loss, loss_history = ga.optimize(
            evaluate_func_clean, 
            progress_callback=progress_callback
        )
        
        logger.info(f"✅ Calibration complete: loss={best_loss:.2f} km/h, vehicles={int(best_genome.sum())}")
        
        return best_genome, best_loss, loss_history
    
    def _generate_final_demand(
        self,
        demand_gen: DemandGenerator,
        genome: np.ndarray,
        od_pairs: List[Tuple[str, str]],
        departure_bins: List[Tuple[int, int]],
        network_file: Path
    ) -> Tuple[Path, Path]:
        """Generate final demand files (demand.csv and trips.xml)."""
        demand_csv = self.output_dir / "data" / "demand.csv"
        trips_file = self.output_dir / "sumo" / "trips.xml"
        
        demand_gen.genome_to_demand_csv(genome, od_pairs, departure_bins, demand_csv)
        demand_gen.demand_csv_to_trips_xml(demand_csv, trips_file)
        # Skip routing - SUMO will route trips dynamically
        
        return demand_csv, trips_file
    
    def _run_final_simulation(
        self, network_file: Path, trips_file: Path
    ) -> Dict[str, float]:
        """Run final simulation to get edge speeds with dynamic routing."""
        sim = SUMOSimulation(
            network_file, 
            trips_file,  # Pass trips.xml
            step_length=1.0,
            warmup_time=self.warmup_minutes * 60,
            simulation_time=(self.warmup_minutes + self.window_minutes) * 60,
            seed=self.seed,
            use_dynamic_routing=True
        )
        
        # Run in 'sumo' dir to generate and keep simulation.sumocfg there
        sumo_dir = self.output_dir / "sumo"
        sumo_dir.mkdir(parents=True, exist_ok=True)
        
        simulated_speeds, _ = sim.run(output_dir=sumo_dir)  # Ignore routing_failures for final sim
        return simulated_speeds
    
    def _export_results(
        self,
        network_file: Path,
        demand_csv: Path,
        trips_file: Path,
        # routes_file removed - using dynamic routing
        observed_edges_file: Path,
        traffic_data_file: Path,
        observed_edges: pd.DataFrame,
        simulated_speeds: Dict[str, float],
        best_loss: float,
        loss_history: List[float],
        quality_metrics: Dict
    ) -> Dict:
        """Export scenario and generate report with comprehensive metadata."""
        # Comprehensive metadata per user request
        metadata = {
            'run_info': {
                'timestamp': datetime.now().isoformat(),
                'bbox_coordinates': {
                    'west': self.bbox[0],
                    'south': self.bbox[1],
                    'east': self.bbox[2],
                    'north': self.bbox[3]
                },
                'seed': self.seed,
                'routing_mode': 'dynamic'  # NEW: Indicate dynamic routing
            },
            'simulation_config': {
                'window_minutes': self.window_minutes,
                'warmup_minutes': self.warmup_minutes,
                'step_length_seconds': 1.0
            },
            'calibration_config': {
                'ga_population': self.ga_population,
                'ga_generations': self.ga_generations,
                'ga_mutation_rate': self.ga_mutation_rate,
                'ga_crossover_rate': self.ga_crossover_rate,
                'ga_elitism': self.ga_elitism,
                'ga_mutation_sigma': self.ga_mutation_sigma,
                'ga_mutation_indpb': self.ga_mutation_indpb,
                'num_workers': self.config.default_parallel_workers
            },
            'demand_config': {
                'num_origins': self.num_origins,
                'num_destinations': self.num_destinations,
                'max_od_pairs': self.max_od_pairs,
                'bin_minutes': self.bin_minutes
            },
            'results': {
                'final_loss_mae_kmh': round(best_loss, 2) if best_loss != float('inf') else None,
                'loss_history': [round(x, 2) for x in loss_history] if loss_history[0] != float('inf') else [],
                'quality_metrics': {
                    'mae_kmh': round(quality_metrics['mae'], 2) if quality_metrics['mae'] and quality_metrics['mae'] != float('inf') else None,
                    'mse_kmh2': round(quality_metrics['mse'], 2) if quality_metrics['mse'] and quality_metrics['mse'] != float('inf') else None,
                    'matched_edges': quality_metrics['matched_edges'],
                    'missing_edges': quality_metrics['missing_edges'],
                    'total_observed_edges': quality_metrics['total_edges'],
                    'match_rate_percent': round(100 * quality_metrics['matched_edges'] / quality_metrics['total_edges'], 1) if quality_metrics['total_edges'] > 0 else 0,
                    'description': 'MAE (Mean Absolute Error) shows average speed error in km/h - lower is better. Match rate shows % of observed segments successfully matched to SUMO edges.'
                }
            },
            'data_sources': {
                'traffic_provider': 'TomTom Flow Segment Data API',
                'traffic_snapshot_timestamp': datetime.now().isoformat(),
                'osm_source': 'Overpass API',
                'traffic_segments_fetched': len(pd.read_csv(traffic_data_file)) if traffic_data_file.exists() else 0
            },
            'output_files': {
                'demand_csv': 'data/demand.csv',
                'trips_xml': 'sumo/trips.xml',
                'routes_xml': 'sumo/routes.rou.xml',
                'network_xml': f'sumo/{network_file.name}',
                'scenario_config': 'sumo/scenario.sumocfg',
                'observed_edges_csv': 'data/observed_edges.csv',
                'traffic_data_raw_csv': 'data/traffic_data_raw.csv',
                'report_html': 'report.html',
                'metadata_json': 'run_meta.json',
                'plots_dir': 'plots',
                'logs_dir': 'logs'
            }
        }
        
        # Export scenario
        exporter = ScenarioExporter(self.output_dir)
        exporter.export(
            network_file, demand_csv, trips_file,  # No routes_file - using dynamic routing
            observed_edges_file, metadata
        )
        
        # Export Custom Formats (URB Project)
        try:
            run_id = self.output_dir.name.replace("run_", "")
            custom_exporter = URBDataExporter(run_id, self.output_dir)
            custom_exporter.export(network_file, trips_file)  # Pass trips.xml
        except Exception as e:
            logger.warning(f"URB export failed: {e}")
        
        # Generate report
        report_gen = ReportGenerator(self.output_dir)
        report_gen.generate(observed_edges, simulated_speeds, loss_history, metadata)
        
        # Visualize network (per user request)
        from demandify.export.visualize import visualize_network
        network_viz_file = self.output_dir / "plots" / "network.png"
        try:
            visualize_network(network_file, network_viz_file)
            logger.info(f"Network visualization saved: {network_viz_file}")
        except Exception as e:
            logger.warning(f"Failed to create network visualization: {e}")
        
        return metadata
