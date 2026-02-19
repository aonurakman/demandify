"""
Genetic algorithm for demand calibration.
Fully seeded for reproducibility with parallel evaluation.

Advanced features:
- Feasible-elite parent selection with fallback (E -> feasibility -> magnitude)
- Random immigrants: inject random individuals each generation to maintain diversity
- Assortative mating: prefer crossover between dissimilar parents
- Deterministic crowding: offspring replace most similar parents
- Adaptive mutation boost: increase mutation on stagnation
- Diversity tracking: genotypic (L2) and phenotypic diversity per generation
"""

import logging
from multiprocessing import Pool, cpu_count
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
            elite_top_pct: Fraction used to define feasible elite slice size (0-1)
            magnitude_penalty_weight: Weight for magnitude term in elite parent ranking
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
        self.last_best_selection_mode = None
        self.last_best_selection_value = None
        self.last_best_raw_loss = None
        self.last_best_feasible_e_loss = None

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

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Parse numeric value with fallback."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        """Parse integer value with fallback."""
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _has_worker_error(individual) -> bool:
        """Return True when worker reported an explicit evaluation failure."""
        metrics = getattr(individual, "metrics", {}) or {}
        if bool(metrics.get("worker_error", False)):
            return True
        return bool(metrics.get("error"))

    def _individual_e_loss(self, individual) -> float:
        """Get flow-fit error E for an individual (fallback to fitness)."""
        if self._has_worker_error(individual):
            return float("inf")
        metrics = getattr(individual, "metrics", {}) or {}
        e_loss = metrics.get("e_loss")
        if e_loss is None:
            if individual.fitness.valid:
                return self._safe_float(individual.fitness.values[0], default=float("inf"))
            return float("inf")
        return self._safe_float(e_loss, default=float("inf"))

    def _individual_fail_total(self, individual) -> int:
        """Get fail_total (routing failures + teleports) with backward-compatible fallback."""
        metrics = getattr(individual, "metrics", {}) or {}

        fail_total = metrics.get("fail_total")
        if fail_total is not None:
            return self._safe_int(fail_total, default=0)

        routing_raw = metrics.get("routing_failures")
        teleports_raw = metrics.get("teleports")
        routing_failures = self._safe_int(routing_raw, default=0)
        teleports = self._safe_int(teleports_raw, default=0)
        fallback_total = routing_failures + teleports
        if fallback_total == 0 and routing_raw is None and teleports_raw is None and self._has_worker_error(individual):
            return 1
        return fallback_total

    def _is_feasible_individual(self, individual) -> bool:
        """
        Strict feasibility predicate used by parent selection and best tracking.

        Feasible requires:
            - valid finite fitness
            - no explicit worker error marker
            - fail_total == 0 (with backward-compatible fallback)
        """
        if not individual.fitness.valid:
            return False
        raw_fitness = self._safe_float(individual.fitness.values[0], default=float("inf"))
        if not np.isfinite(raw_fitness):
            return False
        if self._has_worker_error(individual):
            return False
        return self._individual_fail_total(individual) == 0

    def _individual_reliability_penalty(self, individual) -> float:
        """Get reliability penalty with fallback to (fitness - e_loss)."""
        if self._has_worker_error(individual):
            return float("inf")
        metrics = getattr(individual, "metrics", {}) or {}
        penalty = metrics.get("reliability_penalty")
        if penalty is not None:
            return self._safe_float(penalty, default=0.0)

        if not individual.fitness.valid:
            return 0.0

        raw_loss = self._safe_float(individual.fitness.values[0], default=float("inf"))
        e_loss = self._individual_e_loss(individual)
        if np.isfinite(raw_loss) and np.isfinite(e_loss):
            return max(0.0, raw_loss - e_loss)
        return 0.0

    def _build_parent_selection_plan(self, population) -> Dict[str, Any]:
        """
        Build a per-generation parent-selection plan.

        Priority:
            1) E (ascending)
            2) Feasibility (fail_total == 0)
            3) Magnitude among feasible elites
        """
        pop_size = len(population)
        if pop_size == 0:
            return {
                "mode": "empty",
                "candidate_pool": [],
                "score_by_id": {},
                "population_size": 0,
                "elite_count": 0,
                "feasible_count": 0,
            }

        elite_count = max(1, int(self.elite_top_pct * pop_size))
        e_sorted = sorted(population, key=self._individual_e_loss)
        feasible_sorted = [ind for ind in e_sorted if self._is_feasible_individual(ind)]
        feasible_count = len(feasible_sorted)

        score_by_id: Dict[int, float] = {}

        if feasible_count >= elite_count:
            denominator = max(1, pop_size - 1)
            e_rank_by_id = {id(ind): idx for idx, ind in enumerate(e_sorted)}
            elite_slice = feasible_sorted[:elite_count]
            for ind in elite_slice:
                magnitude = float(sum(ind))
                e_rank_idx = e_rank_by_id[id(ind)]
                rank_term = e_rank_idx / denominator
                score_by_id[id(ind)] = (self.magnitude_penalty_weight * magnitude) + rank_term

            return {
                "mode": "feasible_elite",
                "candidate_pool": elite_slice,
                "score_by_id": score_by_id,
                "population_size": pop_size,
                "elite_count": elite_count,
                "feasible_count": feasible_count,
            }

        for ind in e_sorted:
            e_loss = self._individual_e_loss(ind)
            reliability_penalty = self._individual_reliability_penalty(ind)
            score_by_id[id(ind)] = e_loss + reliability_penalty

        return {
            "mode": "fallback",
            "candidate_pool": e_sorted,
            "score_by_id": score_by_id,
            "population_size": pop_size,
            "elite_count": elite_count,
            "feasible_count": feasible_count,
        }

    def _tournament_select_by_score(self, candidates, score_by_id, k, tournsize: int = 3):
        """Tournament selection minimizing explicit score (without mutating fitness)."""
        if not candidates:
            return []

        tournsize = max(1, min(int(tournsize), len(candidates)))
        selected = []
        for _ in range(k):
            aspirants = [candidates[int(self.rng.randint(0, len(candidates)))] for _ in range(tournsize)]
            winner = min(aspirants, key=lambda ind: score_by_id.get(id(ind), float("inf")))
            selected.append(winner)
        return selected

    def _select_parents(self, population, tournsize: int = 3):
        """Select parents using feasible-elite mode or fallback mode for this generation."""
        plan = self._build_parent_selection_plan(population)
        parents = self._tournament_select_by_score(
            plan["candidate_pool"],
            plan["score_by_id"],
            len(population),
            tournsize=tournsize,
        )
        return parents, plan

    def _clone_individual_snapshot(self, individual):
        """Clone an individual with stable fitness/metrics snapshot."""
        cloned = self.toolbox.clone(individual)
        if individual.fitness.valid:
            cloned.fitness.values = (float(individual.fitness.values[0]),)
        if hasattr(individual, "metrics"):
            cloned.metrics = dict(individual.metrics)
        return cloned

    def _update_best_trackers(
        self,
        population,
        overall_best_ind,
        overall_best_loss: float,
        overall_best_feasible_ind,
        overall_best_feasible_e: float,
    ):
        """Update best raw and best feasible trackers from a population snapshot."""
        for ind in population:
            if not ind.fitness.valid:
                continue

            raw_loss = self._safe_float(ind.fitness.values[0], default=float("inf"))
            if raw_loss < overall_best_loss:
                overall_best_loss = raw_loss
                overall_best_ind = self._clone_individual_snapshot(ind)

            e_loss = self._individual_e_loss(ind)
            if self._is_feasible_individual(ind) and e_loss < overall_best_feasible_e:
                overall_best_feasible_e = e_loss
                overall_best_feasible_ind = self._clone_individual_snapshot(ind)

        return (
            overall_best_ind,
            overall_best_loss,
            overall_best_feasible_ind,
            overall_best_feasible_e,
        )

    @staticmethod
    def _resolve_return_best(
        population,
        overall_best_ind,
        overall_best_loss: float,
        overall_best_feasible_ind,
        overall_best_feasible_e: float,
    ):
        """
        Resolve final best individual with feasible-first policy.

        Returns:
            (best_individual, best_loss, mode) where mode is "feasible" or "raw".
        """
        if overall_best_feasible_ind is not None:
            return overall_best_feasible_ind, float(overall_best_feasible_e), "feasible"

        if overall_best_ind is not None:
            return overall_best_ind, float(overall_best_loss), "raw"

        best_ind = tools.selBest(population, 1)[0]
        return best_ind, float(best_ind.fitness.values[0]), "raw"

    @staticmethod
    def _invalidate_individual(individual):
        """Invalidate fitness and clear derived attributes after variation."""
        if individual.fitness.valid:
            del individual.fitness.values
        if hasattr(individual, "metrics"):
            del individual.metrics

    def optimize(
        self,
        evaluate_func: Callable[[np.ndarray], Union[float, Tuple[float, Dict[str, Any]]]],
        early_stopping_patience: int = 5,
        early_stopping_epsilon: float = 0.1,
        progress_callback: Callable[[int, float, float], None] = None,
        generation_callback: Optional[
            Callable[[int, np.ndarray, float, Dict[str, Any]], None]
        ] = None,
    ) -> Tuple[np.ndarray, float, List[float], List[dict]]:
        """
        Run GA optimization with parallel evaluation.

        Args:
            evaluate_func: Function that takes a genome and returns either a float loss
                           or a (loss, metrics_dict) tuple.
                           MUST be picklable (e.g. partial of top-level func).
            early_stopping_patience: Stop if no improvement for N generations
            early_stopping_epsilon: Minimum improvement threshold
            generation_callback: Optional callback executed once per generation with
                                (generation_idx, best_genome_snapshot, best_loss, best_metrics).
                                Errors in callback are caught and logged.

        Returns:
            (best_genome, best_loss, loss_history, generation_stats)
        """
        logger.info(
            f"Starting GA optimization (pop={self.population_size}, gen={self.num_generations}, workers={self.num_workers})"
        )
        logger.info(
            f"Advanced GA: immigrants={self.immigrant_rate:.0%}, elite_top={self.elite_top_pct:.0%}, "
            f"elite_mag_weight={self.magnitude_penalty_weight}, stagnation_K={self.stagnation_patience}, "
            f"assortative={self.assortative_mating}, crowding={self.deterministic_crowding}"
        )

        # Genetic operators
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register(
            "mutate",
            self._bounded_mutation,
            mu=0,
            sigma=self.mutation_sigma,
            indpb=self.mutation_indpb,
        )

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
        # Track best feasible individual across all generations (fail_total == 0), ranked by E.
        overall_best_feasible_ind = None
        overall_best_feasible_e = float("inf")
        selection_mode_prev = None

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

            (
                overall_best_ind,
                overall_best_loss,
                overall_best_feasible_ind,
                overall_best_feasible_e,
            ) = self._update_best_trackers(
                population,
                overall_best_ind,
                overall_best_loss,
                overall_best_feasible_ind,
                overall_best_feasible_e,
            )

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

                # Parent selection uses feasible-elite mode with per-generation fallback.
                offspring, selection_plan = self._select_parents(population, tournsize=3)
                offspring = list(map(self.toolbox.clone, offspring))

                if selection_plan["mode"] != selection_mode_prev:
                    if selection_plan["mode"] == "fallback":
                        logger.info(
                            "⚠️ Parent selection fallback active at gen %s: feasible=%s < elite_n=%s",
                            gen + 1,
                            selection_plan["feasible_count"],
                            selection_plan["elite_count"],
                        )
                    elif selection_plan["mode"] == "feasible_elite":
                        logger.info(
                            "✅ Parent selection feasible-elite active at gen %s: feasible=%s, elite_n=%s",
                            gen + 1,
                            selection_plan["feasible_count"],
                            selection_plan["elite_count"],
                        )
                    selection_mode_prev = selection_plan["mode"]

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

                # --- Track overall best (raw and feasible-by-E) ---
                raw_fits = [ind.fitness.values[0] for ind in population]
                (
                    overall_best_ind,
                    overall_best_loss,
                    overall_best_feasible_ind,
                    overall_best_feasible_e,
                ) = self._update_best_trackers(
                    population,
                    overall_best_ind,
                    overall_best_loss,
                    overall_best_feasible_ind,
                    overall_best_feasible_e,
                )

                # Stats (use raw losses for reporting)
                current_best = float(min(raw_fits))
                current_mean = float(np.mean(raw_fits))
                current_std = float(np.std(raw_fits))

                # Aggregate metrics for best individual
                best_ind_gen = min(zip(population, raw_fits), key=lambda x: x[1])[0]
                best_metrics = getattr(best_ind_gen, "metrics", {})
                best_fail_total = self._individual_fail_total(best_ind_gen)

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
                pop_fail_totals = []
                for ind in population:
                    m = getattr(ind, "metrics", {})
                    zero_flow_value = m.get("zero_flow_edges")
                    if zero_flow_value is not None:
                        # Skip non-numeric/invalid values (e.g. worker-error placeholders)
                        zf = self._safe_float(zero_flow_value, default=float("inf"))
                        if np.isfinite(zf):
                            pop_zero_flows.append(zf)
                    if "routing_failures" in m:
                        pop_failures.append(m["routing_failures"])
                    pop_fail_totals.append(self._individual_fail_total(ind))

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
                    "best_fail_total": best_fail_total,
                    "mean_fail_total": float(np.mean(pop_fail_totals)) if pop_fail_totals else None,
                    "genotypic_diversity": float(genotypic_diversity),
                    "phenotypic_diversity": float(phenotypic_diversity),
                    "mutation_boosted": self._mutation_boosted,
                }

                loss_history.append(current_best)
                generation_stats.append(gen_stat)

                # Log stats with metrics if available
                metric_str = f" | Trips={int(best_magnitude)}"
                if best_metrics:
                    zero_flow = best_metrics.get("zero_flow_edges", "?")
                    avg_dur = best_metrics.get("avg_trip_duration", 0.0)
                    metric_str += (
                        f", ZeroFlow={zero_flow}, AvgDur={avg_dur:.1f}s, "
                        f"FailTotal={best_fail_total}"
                    )

                boost_str = " [BOOSTED]" if self._mutation_boosted else ""
                logger.info(
                    f"✅ Gen {gen+1}/{self.num_generations}: best={current_best:.2f}, mean={current_mean:.2f}, "
                    f"div={genotypic_diversity:.1f}{metric_str}{boost_str}"
                )

                if progress_callback:
                    progress_callback(gen + 1, current_best, current_mean)

                if generation_callback:
                    try:
                        generation_callback(
                            gen + 1,
                            np.array(best_ind_gen, dtype=int),
                            current_best,
                            dict(best_metrics) if isinstance(best_metrics, dict) else {},
                        )
                    except Exception as e:
                        logger.warning(
                            "Generation callback failed at gen %s: %s",
                            gen + 1,
                            e,
                        )

                if current_best < best_loss - early_stopping_epsilon:
                    best_loss = current_best
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1

        best_individual, best_loss, best_mode = self._resolve_return_best(
            population,
            overall_best_ind,
            overall_best_loss,
            overall_best_feasible_ind,
            overall_best_feasible_e,
        )
        best_genome = np.array(best_individual)
        if best_mode == "feasible":
            logger.info(f"GA complete: best feasible E = {best_loss:.2f}")
        else:
            logger.warning(
                "GA complete without feasible individual (fail_total == 0); "
                f"returning best raw loss = {best_loss:.2f}"
            )
        self.last_best_selection_mode = best_mode
        self.last_best_selection_value = float(best_loss)
        self.last_best_raw_loss = float(overall_best_loss)
        self.last_best_feasible_e_loss = (
            float(overall_best_feasible_e) if overall_best_feasible_ind is not None else None
        )

        return best_genome, best_loss, loss_history, generation_stats

    def _bounded_mutation(self, individual, mu, sigma, indpb):
        """Gaussian mutation with lower-bound clipping only (no upper cap)."""
        for i in range(len(individual)):
            if self.rng.random() < indpb:
                individual[i] += int(self.rng.normal(mu, sigma))
                # Keep demand non-negative while allowing exploration above init bounds.
                individual[i] = max(self.bounds[0], individual[i])
        return (individual,)
