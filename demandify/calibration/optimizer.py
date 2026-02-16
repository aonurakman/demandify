"""
Genetic algorithm for demand calibration.
Fully seeded for reproducibility with parallel evaluation.

Advanced features:
- Magnitude penalty: penalize excessive trip counts among top individuals
- Random immigrants: inject random individuals each generation to maintain diversity
- Assortative mating: prefer crossover between dissimilar parents
- Deterministic crowding: offspring replace most similar parents
- Adaptive mutation boost: increase mutation on stagnation
- Elite re-ranking: secondary sort top x% by magnitude (fewer trips preferred)
- Diversity tracking: genotypic (L2) and phenotypic diversity per generation
"""

import logging
from multiprocessing import Pool, cpu_count
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
from deap import base, creator, tools
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
        init_prob: float = None,
        immigrant_rate: float = 0.03,
        elite_top_pct: float = 0.1,
        magnitude_penalty_weight: float = 0.001,
        stagnation_patience: int = 20,
        stagnation_boost: float = 1.5,
        assortative_mating: bool = True,
        deterministic_crowding: bool = True,
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
            immigrant_rate: Fraction of population replaced by random immigrants (0-1)
            elite_top_pct: Top percentage for secondary sorting by magnitude (0-1)
            magnitude_penalty_weight: Weight for magnitude penalty in fitness
            stagnation_patience: Generations without improvement before mutation boost
            stagnation_boost: Multiplier for mutation sigma/rate on stagnation
            assortative_mating: Prefer crossover between dissimilar parents
            deterministic_crowding: Offspring replace most similar parents
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

        # Advanced GA parameters
        self.immigrant_rate = immigrant_rate
        self.elite_top_pct = elite_top_pct
        self.magnitude_penalty_weight = magnitude_penalty_weight
        self.stagnation_patience = stagnation_patience
        self.stagnation_boost = stagnation_boost
        self.assortative_mating = assortative_mating
        self.deterministic_crowding = deterministic_crowding

        # Track base mutation params for adaptive boost
        self._base_mutation_sigma = mutation_sigma
        self._base_mutation_rate = mutation_rate
        self._mutation_boosted = False

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
            n=self.genome_size,
        )

        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

    def _make_evaluator(self, evaluate_func):
        """Create a picklable evaluator wrapper."""

        def evaluator(individual):
            return (evaluate_func(np.array(individual)),)

        return evaluator

    def _compute_genotypic_diversity(self, population) -> float:
        """Compute mean pairwise L2 distance across the population."""
        if len(population) < 2:
            return 0.0
        arrays = [np.array(ind, dtype=float) for ind in population]
        # Sample pairs for efficiency (cap at 200 pairs)
        n = len(arrays)
        max_pairs = min(200, n * (n - 1) // 2)
        total_dist = 0.0
        count = 0
        indices = list(range(n))
        self.rng.shuffle(indices)
        for i in range(n):
            for j in range(i + 1, n):
                total_dist += np.linalg.norm(arrays[indices[i]] - arrays[indices[j]])
                count += 1
                if count >= max_pairs:
                    return total_dist / count
        return total_dist / max(1, count)

    def _compute_phenotypic_diversity(self, population) -> float:
        """Compute standard deviation of fitness values (phenotypic diversity)."""
        fits = [ind.fitness.values[0] for ind in population if ind.fitness.valid]
        if len(fits) < 2:
            return 0.0
        return float(np.std(fits))

    def _create_immigrant(self) -> list:
        """Create a random immigrant individual within bounds."""
        ind = creator.Individual(
            [
                int(self.rng.randint(self.bounds[0], self.bounds[1] + 1))
                for _ in range(self.genome_size)
            ]
        )
        return ind

    def _assortative_mate_pairs(self, offspring):
        """Pair dissimilar individuals for crossover (assortative mating)."""
        if len(offspring) < 2:
            return []
        # Sort by genome sum (magnitude) as proxy for dissimilarity
        indexed = list(enumerate(offspring))
        indexed.sort(key=lambda x: sum(x[1]))
        # Pair first half with second half (most dissimilar)
        n = len(indexed)
        half = n // 2
        pairs = []
        for k in range(half):
            i1 = indexed[k][0]
            i2 = indexed[n - 1 - k][0]
            pairs.append((i1, i2))
        return pairs

    def _apply_magnitude_penalty(self, population):
        """Apply magnitude penalty: among top elite_top_pct, prefer fewer trips.

        Stores current raw loss as ``ind.raw_loss`` for every valid individual.
        This is refreshed on each call to avoid stale values after cloning/mutation.
        """
        # Always refresh the current (raw) loss before any penalty.
        for ind in population:
            if ind.fitness.valid:
                ind.raw_loss = ind.fitness.values[0]

        if self.magnitude_penalty_weight <= 0 or self.elite_top_pct <= 0:
            return

        sorted_pop = sorted(population, key=lambda ind: ind.raw_loss)
        top_n = max(1, int(len(sorted_pop) * self.elite_top_pct))
        top_individuals = sorted_pop[:top_n]

        for ind in top_individuals:
            magnitude = sum(ind)
            penalty = magnitude * self.magnitude_penalty_weight
            ind.fitness.values = (ind.raw_loss + penalty,)

    @staticmethod
    def _restore_raw_fitness(population):
        """Restore fitness values to their raw (un-penalized) loss."""
        for ind in population:
            if hasattr(ind, "raw_loss"):
                ind.fitness.values = (ind.raw_loss,)

    @staticmethod
    def _invalidate_individual(individual):
        """Invalidate fitness and clear derived attributes after variation."""
        if individual.fitness.valid:
            del individual.fitness.values
        if hasattr(individual, "metrics"):
            del individual.metrics
        if hasattr(individual, "raw_loss"):
            del individual.raw_loss

    def optimize(
        self,
        evaluate_func: Callable[[np.ndarray], Union[float, Tuple[float, Dict[str, Any]]]],
        early_stopping_patience: int = 5,
        early_stopping_epsilon: float = 0.1,
        progress_callback: Callable[[int, float, float], None] = None,
    ) -> Tuple[np.ndarray, float, List[float], List[dict]]:
        """
        Run GA optimization with parallel evaluation.

        Args:
            evaluate_func: Function that takes a genome and returns either a float loss
                           or a (loss, metrics_dict) tuple.
                           MUST be picklable (e.g. partial of top-level func).
            early_stopping_patience: Stop if no improvement for N generations
            early_stopping_epsilon: Minimum improvement threshold

        Returns:
            (best_genome, best_loss, loss_history, generation_stats)
        """
        logger.info(
            f"Starting GA optimization (pop={self.population_size}, gen={self.num_generations}, workers={self.num_workers})"
        )
        logger.info(
            f"Advanced GA: immigrants={self.immigrant_rate:.0%}, elite_top={self.elite_top_pct:.0%}, "
            f"mag_penalty={self.magnitude_penalty_weight}, stagnation_K={self.stagnation_patience}, "
            f"assortative={self.assortative_mating}, crowding={self.deterministic_crowding}"
        )

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
            indpb=self.mutation_indpb,
        )
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        # Create population
        population = self.toolbox.population(n=self.population_size)

        # Track stats
        loss_history = []
        generation_stats = []
        best_loss = float("inf")
        generations_without_improvement = 0
        # Track the actual best individual across all generations (on raw loss only)
        overall_best_ind = None
        overall_best_loss = float("inf")

        # Context manager for Pool ensures cleanup
        # We use map_async or imap for better control
        with Pool(processes=self.num_workers) as pool:

            # Helper for parallel evaluation
            def parallel_evaluate(individuals):
                arrays = [np.array(ind) for ind in individuals]

                results = []
                # pool.imap allows return of any object
                for res in tqdm(
                    pool.imap(evaluate_func, arrays),
                    total=len(individuals),
                    desc="  Evaluating",
                    leave=False,
                ):
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
                # --- Adaptive mutation boost on stagnation ---
                if generations_without_improvement >= self.stagnation_patience:
                    if not self._mutation_boosted:
                        self.mutation_sigma = int(self._base_mutation_sigma * self.stagnation_boost)
                        self.mutation_rate = min(
                            1.0, self._base_mutation_rate * self.stagnation_boost
                        )
                        self._mutation_boosted = True
                        logger.info(
                            f"🔥 Stagnation detected at gen {gen+1}: boosting mutation "
                            f"(sigma={self.mutation_sigma}, rate={self.mutation_rate:.2f})"
                        )
                elif self._mutation_boosted:
                    # Reset mutation back to base values on improvement
                    self.mutation_sigma = self._base_mutation_sigma
                    self.mutation_rate = self._base_mutation_rate
                    self._mutation_boosted = False
                    logger.info("✨ Improvement found: resetting mutation to base values")

                # Update mutate operator with current sigma
                self.toolbox.register(
                    "mutate",
                    self._bounded_mutation,
                    mu=0,
                    sigma=self.mutation_sigma,
                    indpb=self.mutation_indpb,
                )

                # Apply a temporary magnitude re-ranking before parent selection.
                # This only affects mate selection pressure for this generation.
                self._apply_magnitude_penalty(population)

                # Select (using temporarily penalized fitness)
                offspring = self.toolbox.select(population, len(population))
                offspring = list(map(self.toolbox.clone, offspring))

                # Return both parent and cloned offspring fitness to raw values.
                self._restore_raw_fitness(population)
                self._restore_raw_fitness(offspring)

                # --- Crossover (with optional assortative mating) ---
                if self.assortative_mating:
                    pairs = self._assortative_mate_pairs(offspring)
                    for i1, i2 in pairs:
                        if self.rng.random() < self.crossover_rate:
                            self.toolbox.mate(offspring[i1], offspring[i2])
                            self._invalidate_individual(offspring[i1])
                            self._invalidate_individual(offspring[i2])
                else:
                    for child1, child2 in zip(offspring[::2], offspring[1::2]):
                        if self.rng.random() < self.crossover_rate:
                            self.toolbox.mate(child1, child2)
                            self._invalidate_individual(child1)
                            self._invalidate_individual(child2)

                # Mutation
                for mutant in offspring:
                    if self.rng.random() < self.mutation_rate:
                        self.toolbox.mutate(mutant)
                        self._invalidate_individual(mutant)

                # Identify invalid (new) individuals
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

                # --- Inject random immigrants ---
                num_immigrants = max(0, int(self.population_size * self.immigrant_rate))
                immigrants = []
                if num_immigrants > 0:
                    for _ in range(num_immigrants):
                        imm = self._create_immigrant()
                        immigrants.append(imm)
                    invalid_ind.extend(immigrants)

                # Evaluate (Parallel)
                if invalid_ind:
                    logger.info(
                        f"🧬 Generation {gen+1}/{self.num_generations}: evaluating {len(invalid_ind)} individuals"
                        f" ({num_immigrants} immigrants)..."
                    )
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

                # --- Replacement: deterministic crowding or standard elitism ---
                elites = tools.selBest(population, self.elitism)

                if self.deterministic_crowding:
                    # Similarity-based replacement: each offspring replaces the
                    # most similar member of the *non-elite* population if it is
                    # fitter, preserving niche diversity.
                    remaining = [ind for ind in population if ind not in elites]
                    for child in offspring:
                        if not child.fitness.valid:
                            continue
                        if not remaining:
                            break
                        # Find the most similar individual in remaining (L2)
                        child_arr = np.array(child, dtype=float)
                        best_idx = 0
                        best_dist = float("inf")
                        for idx, parent in enumerate(remaining):
                            dist = float(np.linalg.norm(child_arr - np.array(parent, dtype=float)))
                            if dist < best_dist:
                                best_dist = dist
                                best_idx = idx
                        # Replace if child is fitter
                        if child.fitness.values[0] < remaining[best_idx].fitness.values[0]:
                            remaining[best_idx] = child
                    population = elites + remaining
                else:
                    # Standard elitism
                    population = elites + offspring[: self.population_size - self.elitism]

                # --- Inject immigrants by replacing worst individuals ---
                if num_immigrants > 0 and immigrants:
                    # Sort population by fitness (worst last), replace tail
                    population.sort(key=lambda ind: ind.fitness.values[0])
                    for i, imm in enumerate(immigrants):
                        if imm.fitness.valid:
                            population[-(i + 1)] = imm

                # Ensure population size is maintained
                population = population[: self.population_size]

                # --- Track overall best on RAW loss (before magnitude penalty) ---
                raw_fits = [ind.fitness.values[0] for ind in population]
                for ind, raw_loss in zip(population, raw_fits):
                    if raw_loss < overall_best_loss:
                        overall_best_loss = raw_loss
                        overall_best_ind = self.toolbox.clone(ind)
                        overall_best_ind.fitness.values = (raw_loss,)
                        if hasattr(ind, "metrics"):
                            overall_best_ind.metrics = ind.metrics

                # Stats (use raw losses for reporting)
                current_best = float(min(raw_fits))
                current_mean = float(np.mean(raw_fits))
                current_std = float(np.std(raw_fits))

                # Aggregate metrics for best individual
                best_ind_gen = min(zip(population, raw_fits), key=lambda x: x[1])[0]
                best_metrics = getattr(best_ind_gen, "metrics", {})

                # Genome magnitude stats
                magnitudes = [sum(ind) for ind in population]
                best_magnitude = sum(best_ind_gen)
                mean_magnitude = np.mean(magnitudes)

                # Diversity metrics
                genotypic_diversity = self._compute_genotypic_diversity(population)
                phenotypic_diversity = self._compute_phenotypic_diversity(population)

                # Aggregate population-level metrics
                pop_zero_flows = []
                pop_failures = []
                for ind in population:
                    m = getattr(ind, "metrics", {})
                    if "zero_flow_edges" in m:
                        pop_zero_flows.append(m["zero_flow_edges"])
                    if "routing_failures" in m:
                        pop_failures.append(m["routing_failures"])

                gen_stat = {
                    "generation": gen + 1,
                    "best_loss": current_best,
                    "mean_loss": current_mean,
                    "std_loss": current_std,
                    "best_magnitude": float(best_magnitude),
                    "mean_magnitude": float(mean_magnitude),
                    "best_zero_flow": best_metrics.get("zero_flow_edges", None),
                    "mean_zero_flow": float(np.mean(pop_zero_flows)) if pop_zero_flows else None,
                    "best_routing_failures": best_metrics.get("routing_failures", None),
                    "mean_routing_failures": float(np.mean(pop_failures)) if pop_failures else None,
                    "genotypic_diversity": float(genotypic_diversity),
                    "phenotypic_diversity": float(phenotypic_diversity),
                    "mutation_boosted": self._mutation_boosted,
                }

                loss_history.append(current_best)
                generation_stats.append(gen_stat)

                # Log stats with metrics if available
                metric_str = ""
                if best_metrics:
                    zero_flow = best_metrics.get("zero_flow_edges", "?")
                    avg_dur = best_metrics.get("avg_trip_duration", 0.0)
                    trip_fail = best_metrics.get("routing_failures", 0)
                    metric_str = f" | ZeroFlow={zero_flow}, AvgDur={avg_dur:.1f}s, Fail={trip_fail}"

                boost_str = " [BOOSTED]" if self._mutation_boosted else ""
                logger.info(
                    f"✅ Gen {gen+1}/{self.num_generations}: best={current_best:.2f}, mean={current_mean:.2f}, "
                    f"div={genotypic_diversity:.1f}{metric_str}{boost_str}"
                )

                if progress_callback:
                    progress_callback(gen + 1, current_best, current_mean)

                if current_best < best_loss - early_stopping_epsilon:
                    best_loss = current_best
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1

        # Ultimate best is solely on the main objective (raw loss, no magnitude penalty)
        if overall_best_ind is not None:
            best_genome = np.array(overall_best_ind)
            best_loss = overall_best_loss
        else:
            best_ind = tools.selBest(population, 1)[0]
            best_genome = np.array(best_ind)
            best_loss = best_ind.fitness.values[0]

        logger.info(f"GA complete: best loss = {best_loss:.2f}")

        return best_genome, best_loss, loss_history, generation_stats

    def _bounded_mutation(self, individual, mu, sigma, indpb):
        """Gaussian mutation with bounds."""
        for i in range(len(individual)):
            if self.rng.random() < indpb:
                individual[i] += int(self.rng.normal(mu, sigma))
                # Enforce bounds
                individual[i] = max(self.bounds[0], min(self.bounds[1], individual[i]))
        return (individual,)
