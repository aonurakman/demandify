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

from demandify.config import get_config
from demandify.providers.tomtom import TomTomProvider
from demandify.providers.osm import OSMFetcher
from demandify.sumo.network import convert_osm_to_sumo, SUMONetwork
from demandify.sumo.matching import EdgeMatcher
from demandify.sumo.demand import DemandGenerator
from demandify.sumo.simulation import SUMOSimulation
from demandify.calibration.objective import EdgeSpeedObjective
from demandify.calibration.optimizer import GeneticAlgorithm
from demandify.cache.manager import CacheManager
from demandify.cache.keys import bbox_key, osm_key, network_key, traffic_key
from demandify.export.exporter import ScenarioExporter
from demandify.export.report import ReportGenerator

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
        output_dir: Path = None
    ):
        """
        Initialize pipeline.
        
        Args:
            bbox: (west, south, east, north)
            window_minutes: Simulation window in minutes
            seed: Random seed
            ga_population: GA population size
            ga_generations: GA generations
            output_dir: Output directory for results
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
        
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path.cwd() / "demandify_runs" / f"run_{timestamp}"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logging for this run
        self._setup_run_logging()
        
        # Cache and config
        self.config = get_config()
        self.cache_manager = CacheManager(self.config.cache_dir)
        
        logger.info(f"Pipeline initialized: bbox={bbox}, seed={seed}")
    
    def _setup_run_logging(self):
        """Setup file logging for this specific run."""
        log_file = self.output_dir / "pipeline.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        
        # Use same format as console
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # Add to root logger (captures all demandify.* loggers)
        root_logger = logging.getLogger('demandify')
        root_logger.addHandler(file_handler)
        
        # Store handler reference for cleanup
        self.log_handler = file_handler
        
        logger.info(f"Pipeline logs will be saved to {log_file}")
    
    async def run(self) -> Dict:
        """
        Run the full calibration pipeline.
        
        Returns:
            Metadata dict with results
        """
        logger.info("Starting calibration pipeline")
        
        # Stage 1: Fetch traffic data
        logger.info("Stage 1/8: Fetching traffic data")
        traffic_df = await self._fetch_traffic_data()
        
        # Save raw traffic data
        traffic_data_file = self.output_dir / "traffic_data_raw.csv"
        traffic_df.to_csv(traffic_data_file, index=False)
        logger.debug(f"Saved {len(traffic_df)} traffic segments to {traffic_data_file}")
        
        # Stage 2: Fetch OSM data
        logger.info("Stage 2/8: Fetching OSM data")
        osm_file = await self._fetch_osm_data()
        
        # Stage 3: Build SUMO network
        logger.info("Stage 3/8: Building SUMO network")
        network_file = self._build_sumo_network(osm_file)
        
        # Stage 4: Map matching
        logger.info("Stage 4/8: Matching traffic to SUMO edges")
        observed_edges = self._match_traffic_to_edges(traffic_df, network_file)
        
        # Save observed edges
        observed_edges_file = self.output_dir / "observed_edges.csv"
        observed_edges.to_csv(observed_edges_file, index=False)
        logger.debug(f"Matched {len(observed_edges)} traffic segments to SUMO edges")
        
        # Abort if no edges matched (critical for CLI/User feedback)
        if len(observed_edges) == 0:
            error_msg = "No traffic sensors matches in this area. Cannot calibrate demand without ground truth data."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Stage 5: Initialize demand model
        logger.info("Stage 5/8: Initializing demand model")
        demand_gen, od_pairs, departure_bins = self._initialize_demand(network_file)
        
        # Stage 6: Calibrate demand
        logger.info("Stage 6/8: Calibrating demand (GA)")
        best_genome, best_loss, loss_history = self._calibrate_demand(
            demand_gen, od_pairs, departure_bins, observed_edges, network_file
        )
        
        # Stage 7: Generate final demand files
        logger.info("Stage 7/8: Generating final demand files")
        demand_csv, trips_file, routes_file = self._generate_final_demand(
            demand_gen, best_genome, od_pairs, departure_bins, network_file
        )
        
        # Stage 8: Export scenario and report
        logger.info("Stage 8/8: Exporting scenario and report")
        simulated_speeds = self._run_final_simulation(network_file, routes_file)
        
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
                'mae': None,
                'mse': None,
                'matched_edges': 0,
                'missing_edges': 0,
                'total_edges': 0
            }
        
        metadata = self._export_results(
            network_file, demand_csv, trips_file, routes_file,
            observed_edges_file, traffic_data_file, observed_edges,
            simulated_speeds, best_loss, loss_history, quality_metrics
        )
        
        logger.info("Pipeline complete!")
        
        return metadata
    
    async def _fetch_traffic_data(self) -> pd.DataFrame:
        """Fetch traffic data from TomTom."""
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
        
        # Select OD pairs
        origins, destinations = demand_gen.select_od_candidates(
            num_origins=10, num_destinations=10
        )
        
        # Create OD pairs (all combinations)
        od_pairs = [(o, d) for o in origins for d in destinations if o != d]
        od_pairs = od_pairs[:50]  # Limit to 50 OD pairs
        
        # Create departure bins - cover ENTIRE duration (warmup + window)
        # We start from t=0 to populate the network during warmup
        warmup_sec = self.warmup_minutes * 60
        window_sec = self.window_minutes * 60
        total_duration = warmup_sec + window_sec
        
        # Use approximately 5-minute bins (300s)
        target_bin_duration = 300
        num_bins = max(4, int(round(total_duration / target_bin_duration)))
        bin_duration = total_duration // num_bins
        
        departure_bins = []
        for i in range(num_bins):
            start = i * bin_duration
            end = i * bin_duration + bin_duration
            # Adjust last bin to match exactly
            if i == num_bins - 1:
                end = total_duration
            departure_bins.append((start, end))
        
        logger.info(f"Created {len(od_pairs)} OD pairs and {len(departure_bins)} departure bins")
        
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
        
        # Create objective
        objective = EdgeSpeedObjective(observed_edges)
        
        # Evaluation function
        def evaluate(genome: np.ndarray) -> float:
            # Generate demand files
            temp_dir = self.output_dir / "temp_eval"
            temp_dir.mkdir(exist_ok=True)
            
            demand_csv = temp_dir / "demand.csv"
            trips_file = temp_dir / "trips.xml"
            routes_file = temp_dir / "routes.rou.xml"
            
            demand_gen.genome_to_demand_csv(genome, od_pairs, departure_bins, demand_csv)
            demand_gen.demand_csv_to_trips_xml(demand_csv, trips_file)
            demand_gen.route_trips(network_file, trips_file, routes_file)
            
            # Run simulation
            sim = SUMOSimulation(
                network_file, routes_file,
                step_length=1.0,
                warmup_time=self.warmup_minutes * 60,
                simulation_time=(self.warmup_minutes + self.window_minutes) * 60
            )
            
            simulated_speeds = sim.run()
            
            # Calculate loss
            loss = objective.calculate_loss(simulated_speeds)
            
            return loss
        
        # Run GA
        genome_size = len(od_pairs) * len(departure_bins)
        ga = GeneticAlgorithm(
            genome_size=genome_size,
            seed=self.seed,
            bounds=(0, 200),  # Increased to ensure sufficient demand
            population_size=self.ga_population,
            num_generations=self.ga_generations,
            mutation_rate=self.ga_mutation_rate,
            crossover_rate=self.ga_crossover_rate,
            elitism=self.ga_elitism,
            mutation_sigma=self.ga_mutation_sigma,
            mutation_indpb=self.ga_mutation_indpb,
            num_workers=self.config.default_parallel_workers
        )
        
        # Progress callback for UI updates
        def progress_callback(gen: int, best_loss: float, mean_loss: float):
            # logger.info(f"🔄 GA Gen{gen}/{self.ga_generations}: best={best_loss:.2f} km/h, mean={mean_loss:.2f} km/h")
            pass
        
        best_genome, best_loss, loss_history = ga.optimize(evaluate, progress_callback=progress_callback)
        
        logger.info(f"✅ Calibration complete: loss={best_loss:.2f} km/h, vehicles={int(best_genome.sum())}")
        
        return best_genome, best_loss, loss_history
    
    def _generate_final_demand(
        self,
        demand_gen: DemandGenerator,
        genome: np.ndarray,
        od_pairs: List[Tuple[str, str]],
        departure_bins: List[Tuple[int, int]],
        network_file: Path
    ) -> Tuple[Path, Path, Path]:
        """Generate final demand files."""
        demand_csv = self.output_dir / "demand.csv"
        trips_file = self.output_dir / "trips.xml"
        routes_file = self.output_dir / "routes.rou.xml"
        
        demand_gen.genome_to_demand_csv(genome, od_pairs, departure_bins, demand_csv)
        demand_gen.demand_csv_to_trips_xml(demand_csv, trips_file)
        demand_gen.route_trips(network_file, trips_file, routes_file)
        
        return demand_csv, trips_file, routes_file
    
    def _run_final_simulation(
        self, network_file: Path, routes_file: Path
    ) -> Dict[str, float]:
        """Run final simulation to get edge speeds."""
        sim = SUMOSimulation(
            network_file, routes_file,
            step_length=1.0,
            warmup_time=self.warmup_minutes * 60,
            simulation_time=(self.warmup_minutes + self.window_minutes) * 60
        )
        
        return sim.run()
    
    def _export_results(
        self,
        network_file: Path,
        demand_csv: Path,
        trips_file: Path,
        routes_file: Path,
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
                'seed': self.seed
            },
            'simulation_config': {
                'window_minutes': self.window_minutes,
                'warmup_minutes': self.warmup_minutes,
                'step_length_seconds': 1.0
            },
            'calibration_config': {
                'ga_population': self.ga_population,
                'ga_generations': self.ga_generations,
                'num_workers': self.config.default_parallel_workers
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
                'demand_csv': 'demand.csv',
                'trips_xml': 'trips.xml',
                'routes_xml': 'routes.rou.xml',
                'network_xml': network_file.name,
                'scenario_config': 'scenario.sumocfg',
                'observed_edges_csv': 'observed_edges.csv',
                'traffic_data_raw_csv': 'traffic_data_raw.csv',
                'report_html': 'report.html',
                'metadata_json': 'run_meta.json'
            }
        }
        
        # Export scenario
        exporter = ScenarioExporter(self.output_dir)
        exporter.export(
            network_file, demand_csv, trips_file, routes_file,
            observed_edges_file, metadata
        )
        
        # Generate report
        report_gen = ReportGenerator(self.output_dir)
        report_gen.generate(observed_edges, simulated_speeds, loss_history, metadata)
        
        # Visualize network (per user request)
        from demandify.export.visualize import visualize_network
        network_viz_file = self.output_dir / "network_graph.png"
        try:
            visualize_network(network_file, network_viz_file)
            logger.info(f"Network visualization saved: {network_viz_file}")
        except Exception as e:
            logger.warning(f"Failed to create network visualization: {e}")
        
        return metadata
