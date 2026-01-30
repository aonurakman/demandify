"""
Genetic algorithm for demand calibration.
Fully seeded for reproducibility with parallel evaluation.
"""
from typing import List, Tuple, Callable
import numpy as np
import logging
from deap import base, creator, tools
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

logger = logging.getLogger(__name__)


class GeneticAlgorithm:
    """Seeded genetic algorithm for demand optimization with parallel evaluation."""
    
    def __init__(
        self,
        genome_size: int,
        seed: int,
        bounds: Tuple[int, int] = (0, 100),
        population_size: int = 50,
        num_generations: int = 20,
        mutation_rate: float = 0.5,
        crossover_rate: float = 0.7,
        elitism: int = 2,
        mutation_sigma: int = 20,
        mutation_indpb: float = 0.3,
        num_workers: int = None,
        init_prob: float = None
    ):
        """
        Initialize GA.
        
        Args:
            genome_size: Size of genome (num_od_pairs * num_bins)
            seed: Random seed
            bounds: (min, max) values for genome elements
            population_size: Population size
            num_generations: Number of generations
            mutation_rate: Mutation probability (per individual)
            crossover_rate: Crossover probability
            elitism: Number of best individuals to keep
            mutation_sigma: Mutation step size (Gaussian sigma)
            mutation_indpb: Mutation probability (per gene)
            num_workers: Number of parallel workers (None = cpu_count)
        """
        self.genome_size = genome_size
        self.seed = seed
        self.bounds = bounds
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.mutation_sigma = mutation_sigma
        self.mutation_indpb = mutation_indpb
        self.num_workers = num_workers or max(1, cpu_count() - 1)
        self.init_prob = init_prob
        
        # Seeded RNG
        self.rng = np.random.RandomState(seed)
        
        # Setup DEAP
        self._setup_deap()
    
    def _setup_deap(self):
        """Setup DEAP creator and toolbox."""
        # Create fitness and individual classes
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)
        
        self.toolbox = base.Toolbox()
        
        # Attribute generator (seeded)
        def gen_attr():
            if self.init_prob is not None:
                if self.rng.random() < self.init_prob:
                    return self.rng.randint(self.bounds[0], self.bounds[1] + 1)
                else:
                    return 0
            else:
                return self.rng.randint(self.bounds[0], self.bounds[1] + 1)

        self.toolbox.register("attr_int", gen_attr)
        
        # Individual and population
        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            self.toolbox.attr_int,
            n=self.genome_size
        )
        
        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual
        )
    
    def _make_evaluator(self, evaluate_func):
        """Create a picklable evaluator wrapper."""
        def evaluator(individual):
            return (evaluate_func(np.array(individual)),)
        return evaluator
    
    def optimize(
        self,
        evaluate_func: Callable[[np.ndarray], float],
        early_stopping_patience: int = 5,
        early_stopping_epsilon: float = 0.1,
        progress_callback: Callable[[int, float, float], None] = None
    ) -> Tuple[np.ndarray, float, List[float]]:
        """
        Run GA optimization with parallel evaluation.
        
        Args:
            evaluate_func: Function that takes a genome and returns loss. 
                           MUST be picklable (e.g. partial of top-level func).
            early_stopping_patience: Stop if no improvement for N generations
            early_stopping_epsilon: Minimum improvement threshold
        
        Returns:
            (best_genome, best_loss, loss_history)
        """
        logger.info(f"Starting GA optimization (pop={self.population_size}, gen={self.num_generations}, workers={self.num_workers})")
        
        # Helper to unpack single-element tuple return from evaluate if needed
        # But evaluate_func is expected to return float
        def fitness_wrapper(ind):
            return (evaluate_func(np.array(ind)),)

        self.toolbox.register("evaluate", fitness_wrapper)
        
        # Genetic operators
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register(
            "mutate",
            self._bounded_mutation,
            mu=0,
            sigma=self.mutation_sigma,
            indpb=self.mutation_indpb
        )
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Create population
        population = self.toolbox.population(n=self.population_size)
        
        # Track stats
        loss_history = []
        best_loss = float('inf')
        generations_without_improvement = 0
        
        # Context manager for Pool ensures cleanup
        # We use map_async or imap for better control
        with Pool(processes=self.num_workers) as pool:
            
            # Helper for parallel evaluation
            def parallel_evaluate(individuals):
                arrays = [np.array(ind) for ind in individuals]
                
                results = []
                # pool.imap allows return of any object
                for res in tqdm(pool.imap(evaluate_func, arrays), total=len(individuals), desc="  Evaluating", leave=False):
                    results.append(res)
                return results

            # Initial Evaluation
            logger.info("Evaluating initial population...")
            results = parallel_evaluate(population)
            for ind, res in zip(population, results):
                if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], dict):
                    loss, metrics = res
                    ind.fitness.values = (loss,)
                    ind.metrics = metrics
                else:
                    # Fallback for pure float return
                    loss = res[0] if isinstance(res, tuple) else res
                    ind.fitness.values = (loss,)
                    ind.metrics = {}
            
            # Evolution loop
            for gen in range(self.num_generations):
                # Select
                offspring = self.toolbox.select(population, len(population))
                offspring = list(map(self.toolbox.clone, offspring))
                
                # Crossover
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if self.rng.random() < self.crossover_rate:
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values
                        if hasattr(child1, 'metrics'): del child1.metrics
                        if hasattr(child2, 'metrics'): del child2.metrics
                
                # Mutation
                for mutant in offspring:
                    if self.rng.random() < self.mutation_rate:
                        self.toolbox.mutate(mutant)
                        del mutant.fitness.values
                        if hasattr(mutant, 'metrics'): del mutant.metrics
                
                # Identify invalid (new) individuals
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                
                # Evaluate (Parallel)
                if invalid_ind:
                    logger.info(f"🧬 Generation {gen+1}/{self.num_generations}: evaluating {len(invalid_ind)} individuals...")
                    results = parallel_evaluate(invalid_ind)
                    for ind, res in zip(invalid_ind, results):
                        if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], dict):
                            loss, metrics = res
                            ind.fitness.values = (loss,)
                            ind.metrics = metrics
                        else:
                            loss = res[0] if isinstance(res, tuple) else res
                            ind.fitness.values = (loss,)
                            ind.metrics = {}
                
                # Elitism
                population = tools.selBest(population, self.elitism) + offspring[:-self.elitism]
            
                # Stats
                fits = [ind.fitness.values[0] for ind in population]
                current_best = min(fits)
                current_mean = np.mean(fits)
                
                # Aggregate metrics for best individual
                best_ind_gen = tools.selBest(population, 1)[0]
                best_metrics = getattr(best_ind_gen, 'metrics', {})
                
                loss_history.append(current_best)
                
                # Log stats with metrics if available
                metric_str = ""
                if best_metrics:
                    zero_flow = best_metrics.get('zero_flow_edges', '?')
                    avg_dur = best_metrics.get('avg_trip_duration', 0.0)
                    trip_fail = best_metrics.get('routing_failures', 0)
                    metric_str = f" | ZeroFlow={zero_flow}, AvgDur={avg_dur:.1f}s, Fail={trip_fail}"
                
                logger.info(f"✅ Gen {gen+1} Stats: Best Loss={current_best:.2f}, Mean={current_mean:.2f}{metric_str}")
                
                if progress_callback:
                    progress_callback(gen + 1, current_best, current_mean)
                
                if current_best < best_loss - early_stopping_epsilon:
                    best_loss = current_best
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1
        
        # Get best individual
        best_ind = tools.selBest(population, 1)[0]
        best_genome = np.array(best_ind)
        best_loss = best_ind.fitness.values[0]
        
        logger.info(f"GA complete: best loss = {best_loss:.2f}")
        
        return best_genome, best_loss, loss_history
    
    def _bounded_mutation(self, individual, mu, sigma, indpb):
        """Gaussian mutation with bounds."""
        for i in range(len(individual)):
            if self.rng.random() < indpb:
                individual[i] += int(self.rng.normal(mu, sigma))
                # Enforce bounds
                individual[i] = max(self.bounds[0], min(self.bounds[1], individual[i]))
        return individual,
