"""
Genetic algorithm for demand calibration.
Fully seeded for reproducibility with parallel evaluation.

Advanced features:
- MAE-elite lexicographic selection (MAE -> failure rate -> magnitude)
- Random immigrants: inject random individuals each generation to maintain diversity
- Assortative mating: prefer crossover between dissimilar parents
- Deterministic crowding: offspring replace most similar parents
- Adaptive mutation boost: increase mutation on stagnation
- Diversity tracking: genotypic (L2) and phenotypic diversity per generation
"""

import logging
from contextlib import nullcontext
from fractions import Fraction
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
        magnitude_penalty_weight: Optional[float] = None,
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
            elite_top_pct: Fraction used to define the top-MAE elite slice size (0-1)
            magnitude_penalty_weight: Deprecated compatibility parameter (unused)
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
        self.last_best_mae = None
        self.last_best_mae_candidate_mae = None
        self.last_best_mae_candidate_failure_rate = None
        self.last_best_mae_candidate_fail_total = None
        self.last_best_mae_candidate_magnitude = None
        self.last_best_selected_mae = None
        self.last_best_selected_failure_rate = None
        self.last_best_selected_fail_total = None
        self.last_best_selected_magnitude = None
        # Deprecated compatibility attributes. These now alias MAE-based values.
        self.last_best_raw_loss = None
        self.last_best_selected_raw_loss = None
        self.last_best_selected_e_loss = None

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

    def _individual_mae(self, individual) -> float:
        """Get MAE from metrics or fitness with invalid/error protection."""
        if not individual.fitness.valid or self._has_worker_error(individual):
            return float("inf")
        metrics = getattr(individual, "metrics", {}) or {}
        mae = metrics.get("mae")
        if mae is not None:
            return self._safe_float(mae, default=float("inf"))
        return self._safe_float(individual.fitness.values[0], default=float("inf"))

    def _individual_raw_loss(self, individual) -> float:
        """Backward-compatible alias for the MAE fitness value."""
        return self._individual_mae(individual)

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

    @staticmethod
    def _individual_magnitude(individual) -> float:
        """Get genome magnitude for magnitude pressure."""
        return float(sum(individual))

    def _individual_total_vehicles(self, individual) -> int:
        """Get total vehicles with fallback to genome magnitude."""
        metrics = getattr(individual, "metrics", {}) or {}
        total_vehicles = metrics.get("total_vehicles")
        if total_vehicles is not None:
            parsed = self._safe_int(total_vehicles, default=0)
            if parsed >= 0:
                return parsed
        return max(0, self._safe_int(self._individual_magnitude(individual), default=0))

    def _individual_failure_rate_key(self, individual) -> Tuple[int, Fraction]:
        """Return exact sortable failure-rate key using integer arithmetic semantics."""
        if not individual.fitness.valid or self._has_worker_error(individual):
            return (1, Fraction(0, 1))

        fail_total = max(0, self._individual_fail_total(individual))
        total_vehicles = self._individual_total_vehicles(individual)
        if total_vehicles <= 0:
            if fail_total <= 0:
                return (0, Fraction(0, 1))
            return (1, Fraction(0, 1))
        return (0, Fraction(fail_total, total_vehicles))

    def _individual_failure_rate(self, individual) -> float:
        """Get numeric failure rate for reporting."""
        rate_is_inf, rate_fraction = self._individual_failure_rate_key(individual)
        if rate_is_inf:
            return float("inf")
        return float(rate_fraction)

    def _primary_sort_key(self, individual):
        """Primary ordering used to define the MAE elite slice."""
        rate_is_inf, rate_fraction = self._individual_failure_rate_key(individual)
        return (
            self._individual_mae(individual),
            rate_is_inf,
            rate_fraction,
            self._individual_magnitude(individual),
        )

    def _elite_selection_sort_key(self, individual):
        """Deterministic lexicographic ordering inside the MAE elite."""
        rate_is_inf, rate_fraction = self._individual_failure_rate_key(individual)
        return (
            rate_is_inf,
            rate_fraction,
            self._individual_magnitude(individual),
            self._individual_mae(individual),
        )

    def _select_best_candidate(self, candidates, key_by_id):
        """Pick the best MAE-elite candidate under the lexicographic key."""
        if not candidates:
            return None
        return min(candidates, key=lambda ind: key_by_id.get(id(ind), self._elite_selection_sort_key(ind)))

    def _build_parent_selection_plan(self, population) -> Dict[str, Any]:
        """
        Build a per-generation parent-selection plan.

        Priority:
            1) Sort full population by MAE
            2) Keep the top elite slice defined by elite_top_pct
            3) Within that elite, prefer:
               - lower failure_rate
               - lower magnitude
               - lower exact MAE
        """
        pop_size = len(population)
        if pop_size == 0:
            return {
                "mode": "empty",
                "candidate_pool": [],
                "selection_key_by_id": {},
                "population_size": 0,
                "elite_count": 0,
            }

        elite_count = max(1, int(self.elite_top_pct * pop_size))
        mae_sorted = sorted(population, key=self._primary_sort_key)
        elite_slice = mae_sorted[:elite_count]
        selection_key_by_id = {
            id(ind): self._elite_selection_sort_key(ind)
            for ind in elite_slice
        }

        return {
            "mode": "mae_elite_lexicographic",
            "candidate_pool": elite_slice,
            "selection_key_by_id": selection_key_by_id,
            "population_size": pop_size,
            "elite_count": elite_count,
        }

    def _tournament_select_by_key(self, candidates, key_by_id, k, tournsize: int = 3):
        """Tournament selection minimizing explicit lexicographic key."""
        if not candidates:
            return []

        tournsize = max(1, min(int(tournsize), len(candidates)))
        selected = []
        for _ in range(k):
            aspirants = [candidates[int(self.rng.randint(0, len(candidates)))] for _ in range(tournsize)]
            winner = min(
                aspirants,
                key=lambda ind: key_by_id.get(id(ind), self._elite_selection_sort_key(ind)),
            )
            selected.append(winner)
        return selected

    def _select_parents(self, population, tournsize: int = 3):
        """Select parents from the MAE elite using lexicographic preferences."""
        plan = self._build_parent_selection_plan(population)
        parents = self._tournament_select_by_key(
            plan["candidate_pool"],
            plan["selection_key_by_id"],
            len(population),
            tournsize=tournsize,
        )
        return parents, plan

    def _select_survival_elites(self, population, k: int):
        """Keep elites using the same MAE-elite lexicographic ordering as parent selection."""
        if k <= 0 or not population:
            return []

        plan = self._build_parent_selection_plan(population)
        ranked_slice = sorted(
            plan["candidate_pool"],
            key=lambda ind: plan["selection_key_by_id"].get(
                id(ind), self._elite_selection_sort_key(ind)
            ),
        )
        if len(ranked_slice) >= k:
            return ranked_slice[:k]

        selected_ids = {id(ind) for ind in ranked_slice}
        remainder = [
            ind for ind in sorted(population, key=self._primary_sort_key) if id(ind) not in selected_ids
        ]
        return ranked_slice + remainder[: max(0, k - len(ranked_slice))]

    @staticmethod
    def _exclude_by_identity(population, excluded):
        """Return population members excluding only exact object identities."""
        excluded_ids = {id(ind) for ind in excluded}
        return [ind for ind in population if id(ind) not in excluded_ids]

    def _clone_individual_snapshot(self, individual):
        """Clone an individual with stable fitness/metrics snapshot."""
        cloned = self.toolbox.clone(individual)
        if individual.fitness.valid:
            cloned.fitness.values = (float(individual.fitness.values[0]),)
        if hasattr(individual, "metrics"):
            cloned.metrics = dict(individual.metrics)
        return cloned

    def _individual_summary(self, individual, mode: Optional[str] = None) -> Dict[str, float]:
        """Build a compact metrics summary for a tracked individual."""
        summary = {
            "mae": float(self._individual_mae(individual)),
            "failure_rate": float(self._individual_failure_rate(individual)),
            "fail_total": int(self._individual_fail_total(individual)),
            "magnitude": float(self._individual_magnitude(individual)),
        }
        if mode is not None:
            summary["mode"] = mode
        return summary

    def _update_best_trackers(
        self,
        population,
        overall_best_ind,
        overall_best_loss: float,
    ):
        """Update best-MAE tracker from a population snapshot."""
        for ind in population:
            if not ind.fitness.valid:
                continue

            mae = self._individual_mae(ind)
            if mae < overall_best_loss:
                overall_best_loss = mae
                overall_best_ind = self._clone_individual_snapshot(ind)

        return overall_best_ind, overall_best_loss

    def _select_generation_representative(self, population):
        """Select the per-generation representative using the MAE-elite rule."""
        if not population:
            return None, None

        selection_plan = self._build_parent_selection_plan(population)
        selected_ind = self._select_best_candidate(
            selection_plan["candidate_pool"],
            selection_plan["selection_key_by_id"],
        )
        if selected_ind is None:
            return None, None

        return self._clone_individual_snapshot(selected_ind), self._individual_summary(
            selected_ind,
            mode=selection_plan["mode"],
        )

    def _select_best_generation_representative(self, representatives):
        """Apply a global MAE-elite pass over generation representatives."""
        if not representatives:
            return None, None

        elite_count = max(1, int(self.elite_top_pct * len(representatives)))
        mae_sorted = sorted(
            representatives,
            key=lambda item: (
                item[1]["mae"],
                self._individual_failure_rate_key(item[0]),
                item[1]["magnitude"],
            ),
        )
        mae_elite = mae_sorted[:elite_count]
        best_ind, best_state = min(
            mae_elite,
            key=lambda item: (
                self._individual_failure_rate_key(item[0]),
                item[1]["magnitude"],
                item[1]["mae"],
            ),
        )
        return best_ind, dict(best_state)

    def _resolve_return_best(
        self,
        population,
        overall_best_ind,
        overall_best_loss: float,
        generation_representatives,
    ):
        """
        Resolve final best individual with MAE-elite lexicographic policy.

        Returns:
            (best_individual, selected_state)
        """
        selected_ind, selected_state = self._select_best_generation_representative(
            generation_representatives
        )
        if selected_ind is not None and selected_state is not None:
            return selected_ind, selected_state

        if overall_best_ind is not None:
            return overall_best_ind, self._individual_summary(
                overall_best_ind,
                mode="mae_fallback",
            )

        best_ind = tools.selBest(population, 1)[0]
        return best_ind, self._individual_summary(best_ind, mode="mae_fallback")

    @staticmethod
    def _invalidate_individual(individual):
        """Invalidate fitness and clear derived attributes after variation."""
        if individual.fitness.valid:
            del individual.fitness.values
        if hasattr(individual, "metrics"):
            del individual.metrics

    def _cx_two_point_seeded(self, ind1, ind2):
        """
        Deterministic two-point crossover using the GA-local RNG.

        Equivalent in spirit to DEAP's cxTwoPoint, but avoids the global
        `random` module so runs are reproducible for a fixed seed.
        """
        size = min(len(ind1), len(ind2))
        if size < 2:
            return ind1, ind2

        cx1 = int(self.rng.randint(1, size + 1))
        cx2 = int(self.rng.randint(1, size))
        if cx2 >= cx1:
            cx2 += 1
        else:
            cx1, cx2 = cx2, cx1

        ind1[cx1:cx2], ind2[cx1:cx2] = ind2[cx1:cx2], ind1[cx1:cx2]
        return ind1, ind2

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
                                (generation_idx, selected_genome_snapshot,
                                 selected_mae, selected_metrics).
                                Errors in callback are caught and logged.

        Returns:
            (best_genome, best_loss, loss_history, generation_stats)
        """
        logger.info(
            f"Starting GA optimization (pop={self.population_size}, gen={self.num_generations}, workers={self.num_workers})"
        )
        logger.info(
            f"Advanced GA: immigrants={self.immigrant_rate:.0%}, elite_top={self.elite_top_pct:.0%}, "
            "elite_secondary=(failure_rate -> magnitude -> MAE within MAE elite), "
            f"stagnation_K={self.stagnation_patience}, "
            f"assortative={self.assortative_mating}, crowding={self.deterministic_crowding}"
        )

        # Genetic operators
        self.toolbox.register("mate", self._cx_two_point_seeded)
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
        best_loss_for_stagnation = float("inf")
        generations_without_improvement = 0
        # Track the actual best individual across all generations by MAE.
        overall_best_ind = None
        overall_best_loss = float("inf")
        # Track each generation representative so final selection can apply
        # the same staged rule globally across generations.
        generation_representatives = []
        selection_mode_prev = None

        # Use in-process evaluation for workers=1 to maximize reproducibility
        # and avoid multiprocessing spawn/fork side effects.
        if self.num_workers <= 1:
            logger.info("Using single-process evaluation mode (workers=1)")

            class _SerialPool:
                @staticmethod
                def imap(func, iterable):
                    for item in iterable:
                        yield func(item)

            eval_context = nullcontext(_SerialPool())
        else:
            eval_context = Pool(processes=self.num_workers)

        # Context manager ensures worker cleanup in parallel mode.
        with eval_context as pool:

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
            ) = self._update_best_trackers(
                population,
                overall_best_ind,
                overall_best_loss,
            )
            initial_rep_ind, initial_rep_state = self._select_generation_representative(population)
            if initial_rep_ind is not None and initial_rep_state is not None:
                generation_representatives.append((initial_rep_ind, initial_rep_state))

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

                # Parent selection uses a top-MAE elite slice with lexicographic preferences.
                offspring, selection_plan = self._select_parents(population, tournsize=3)
                offspring = list(map(self.toolbox.clone, offspring))

                if selection_plan["mode"] != selection_mode_prev:
                    logger.info(
                        "✅ Parent selection MAE-elite lexicographic active at gen %s: elite_n=%s/%s",
                        gen + 1,
                        selection_plan["elite_count"],
                        selection_plan["population_size"],
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
                elites = self._select_survival_elites(population, self.elitism)

                if self.deterministic_crowding:
                    # Similarity-based replacement: each offspring replaces the
                    # most similar member of the *non-elite* population if it is
                    # fitter, preserving niche diversity.
                    # Use identity-based exclusion so duplicate genomes are retained
                    # unless they are the exact elite objects.
                    remaining = self._exclude_by_identity(population, elites)
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

                # --- Track overall best MAE and best selected candidate ---
                mae_values = [ind.fitness.values[0] for ind in population]
                (
                    overall_best_ind,
                    overall_best_loss,
                ) = self._update_best_trackers(
                    population,
                    overall_best_ind,
                    overall_best_loss,
                )

                # Stats use the selected individual's MAE, while mean/std stay population-wide.
                current_mean = float(np.mean(mae_values))
                current_std = float(np.std(mae_values))

                # Aggregate metrics for the generation's selected individual.
                best_ind_gen, best_gen_state = self._select_generation_representative(population)
                if best_ind_gen is None:
                    best_ind_gen = min(zip(population, mae_values), key=lambda x: x[1])[0]
                    best_gen_state = self._individual_summary(best_ind_gen, mode="mae_fallback")
                if best_ind_gen is not None and best_gen_state is not None:
                    generation_representatives.append(
                        (self._clone_individual_snapshot(best_ind_gen), dict(best_gen_state))
                    )
                best_metrics = getattr(best_ind_gen, "metrics", {})
                best_fail_total = self._individual_fail_total(best_ind_gen)
                best_failure_rate = self._individual_failure_rate(best_ind_gen)
                best_selected_mae = self._individual_mae(best_ind_gen)
                current_best = float(best_selected_mae)

                # Genome magnitude stats
                magnitudes = [sum(ind) for ind in population]
                best_magnitude = sum(best_ind_gen)
                mean_magnitude = np.mean(magnitudes)

                # Diversity metrics
                genotypic_diversity = self._compute_genotypic_diversity(population)
                phenotypic_diversity = self._compute_phenotypic_diversity(population)

                # Aggregate population-level metrics
                pop_zero_flows = []
                pop_failure_rates = []
                pop_fail_totals = []
                for ind in population:
                    m = getattr(ind, "metrics", {})
                    zero_flow_value = m.get("zero_flow_edges")
                    if zero_flow_value is not None:
                        # Skip non-numeric/invalid values (e.g. worker-error placeholders)
                        zf = self._safe_float(zero_flow_value, default=float("inf"))
                        if np.isfinite(zf):
                            pop_zero_flows.append(zf)
                    failure_rate = self._individual_failure_rate(ind)
                    if np.isfinite(failure_rate):
                        pop_failure_rates.append(failure_rate)
                    pop_fail_totals.append(self._individual_fail_total(ind))

                gen_stat = {
                    "generation": gen + 1,
                    "best_loss": current_best,
                    "mean_loss": current_mean,
                    "std_loss": current_std,
                    "best_magnitude": float(best_magnitude),
                    "mean_magnitude": float(mean_magnitude),
                    "best_failure_rate": float(best_failure_rate) if np.isfinite(best_failure_rate) else None,
                    "mean_failure_rate": float(np.mean(pop_failure_rates)) if pop_failure_rates else None,
                    "best_zero_flow": best_metrics.get("zero_flow_edges", None),
                    "mean_zero_flow": float(np.mean(pop_zero_flows)) if pop_zero_flows else None,
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
                    if np.isfinite(best_failure_rate):
                        failure_rate_str = f"{best_failure_rate * 100.0:.1f}%"
                    else:
                        failure_rate_str = "inf"
                    metric_str += (
                        f", ZeroFlow={zero_flow}, AvgDur={avg_dur:.1f}s, "
                        f"FailRate={failure_rate_str}, "
                        f"FailTotal={best_fail_total}"
                    )

                boost_str = " [BOOSTED]" if self._mutation_boosted else ""
                logger.info(
                    f"✅ Gen {gen+1}/{self.num_generations}: mae={current_best:.2f}, mean_mae={current_mean:.2f}, "
                    f"div={genotypic_diversity:.1f}{metric_str}{boost_str}"
                )

                if progress_callback:
                    progress_callback(gen + 1, current_best, current_mean)

                if generation_callback:
                    try:
                        generation_callback(
                            gen + 1,
                            np.array(best_ind_gen, dtype=int),
                            best_selected_mae,
                            dict(best_metrics) if isinstance(best_metrics, dict) else {},
                        )
                    except Exception as e:
                        logger.warning(
                            "Generation callback failed at gen %s: %s",
                            gen + 1,
                            e,
                        )

                # Stagnation boost should reset on any strict improvement, even if the
                # change is smaller than the early-stopping epsilon.
                if current_best < best_loss_for_stagnation:
                    best_loss_for_stagnation = current_best
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1

        best_individual, best_selection = self._resolve_return_best(
            population,
            overall_best_ind,
            overall_best_loss,
            generation_representatives,
        )
        best_loss = float(best_selection["mae"])
        best_genome = np.array(best_individual)
        best_mae_candidate = (
            self._individual_summary(overall_best_ind)
            if overall_best_ind is not None
            else self._individual_summary(best_individual)
        )
        logger.info(
            "GA complete: selected mae=%.2f, failure_rate=%.4f, fail_total=%s, magnitude=%.0f | best_mae_candidate mae=%.2f, failure_rate=%.4f, fail_total=%s, magnitude=%.0f",
            best_selection["mae"],
            best_selection["failure_rate"],
            best_selection["fail_total"],
            best_selection["magnitude"],
            best_mae_candidate["mae"],
            best_mae_candidate["failure_rate"],
            best_mae_candidate["fail_total"],
            best_mae_candidate["magnitude"],
        )
        self.last_best_selection_mode = best_selection["mode"]
        self.last_best_selection_value = float(best_selection["failure_rate"])
        self.last_best_mae = float(overall_best_loss)
        self.last_best_mae_candidate_mae = float(best_mae_candidate["mae"])
        self.last_best_mae_candidate_failure_rate = float(best_mae_candidate["failure_rate"])
        self.last_best_mae_candidate_fail_total = int(best_mae_candidate["fail_total"])
        self.last_best_mae_candidate_magnitude = float(best_mae_candidate["magnitude"])
        self.last_best_selected_mae = float(best_selection["mae"])
        self.last_best_selected_failure_rate = float(best_selection["failure_rate"])
        self.last_best_selected_fail_total = int(best_selection["fail_total"])
        self.last_best_selected_magnitude = float(best_selection["magnitude"])
        # Deprecated compatibility aliases.
        self.last_best_raw_loss = float(overall_best_loss)
        self.last_best_selected_raw_loss = float(best_selection["mae"])
        self.last_best_selected_e_loss = float(best_selection["mae"])

        return best_genome, best_loss, loss_history, generation_stats

    def _bounded_mutation(self, individual, mu, sigma, indpb):
        """Gaussian mutation with lower-bound clipping only (no upper cap)."""
        for i in range(len(individual)):
            if self.rng.random() < indpb:
                individual[i] += int(self.rng.normal(mu, sigma))
                # Keep demand non-negative while allowing exploration above init bounds.
                individual[i] = max(self.bounds[0], individual[i])
        return (individual,)
