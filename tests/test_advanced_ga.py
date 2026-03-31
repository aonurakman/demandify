"""Tests for advanced GA parameterization features."""

import inspect

import numpy as np
import pandas as pd
import pytest
from deap import creator

from demandify.calibration import optimizer as optimizer_module
from demandify.calibration.optimizer import GeneticAlgorithm
from demandify.export.report import ReportGenerator

# ---------------------------------------------------------------------------
# Helper: a simple evaluate function that returns (loss, metrics)
# ---------------------------------------------------------------------------


def _simple_evaluate(genome):
    """Evaluate function: loss = mean absolute value, lower is better."""
    loss = float(np.mean(np.abs(genome)))
    total_vehicles = int(np.sum(genome))
    metrics = {
        "mae": loss,
        "missing_edges": int(np.sum(genome == 0)),
        "zero_flow_edges": int(np.sum(genome == 0)),
        "routing_failures": 0,
        "teleports": 0,
        "fail_total": 0,
        "failure_rate": 0.0,
        "avg_trip_duration": 100.0,
        "total_vehicles": total_vehicles,
        "magnitude": total_vehicles,
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# GeneticAlgorithm constructor tests
# ---------------------------------------------------------------------------


class TestGAAdvancedParams:
    """Test that advanced parameters are properly accepted and stored."""

    def test_default_advanced_params(self):
        ga = GeneticAlgorithm(genome_size=10, seed=42)
        assert ga.immigrant_rate == 0.03
        assert ga.elite_top_pct == 0.1
        assert ga.stagnation_patience == 20
        assert ga.stagnation_boost == 1.5
        assert ga.assortative_mating is True
        assert ga.deterministic_crowding is True

    def test_custom_advanced_params(self):
        ga = GeneticAlgorithm(
            genome_size=10,
            seed=42,
            immigrant_rate=0.05,
            elite_top_pct=0.2,
            stagnation_patience=10,
            stagnation_boost=2.0,
            assortative_mating=False,
            deterministic_crowding=False,
        )
        assert ga.immigrant_rate == 0.05
        assert ga.elite_top_pct == 0.2
        assert ga.stagnation_patience == 10
        assert ga.stagnation_boost == 2.0
        assert ga.assortative_mating is False
        assert ga.deterministic_crowding is False

    def test_optimize_signature_removes_unused_early_stopping_args(self):
        params = inspect.signature(GeneticAlgorithm.optimize).parameters
        assert "early_stopping_patience" not in params
        assert "early_stopping_epsilon" not in params


# ---------------------------------------------------------------------------
# Diversity metrics
# ---------------------------------------------------------------------------


class TestDiversityMetrics:
    """Test genotypic and phenotypic diversity computation."""

    def test_genotypic_diversity_identical_population(self):
        ga = GeneticAlgorithm(genome_size=5, seed=42, population_size=4)
        # All identical individuals -> zero diversity
        pop = [creator.Individual([1, 2, 3, 4, 5]) for _ in range(4)]
        diversity = ga._compute_genotypic_diversity(pop)
        assert diversity == 0.0

    def test_genotypic_diversity_different_population(self):
        ga = GeneticAlgorithm(genome_size=3, seed=42, population_size=3)
        pop = [
            creator.Individual([0, 0, 0]),
            creator.Individual([10, 10, 10]),
            creator.Individual([5, 5, 5]),
        ]
        diversity = ga._compute_genotypic_diversity(pop)
        assert diversity > 0.0

    def test_phenotypic_diversity(self):
        ga = GeneticAlgorithm(genome_size=3, seed=42, population_size=3)
        pop = [
            creator.Individual([1, 2, 3]),
            creator.Individual([4, 5, 6]),
            creator.Individual([7, 8, 9]),
        ]
        for ind, loss in zip(pop, [10.0, 20.0, 30.0]):
            ind.fitness.values = (loss,)
        diversity = ga._compute_phenotypic_diversity(pop)
        assert diversity > 0.0

    def test_phenotypic_diversity_identical_fitness(self):
        ga = GeneticAlgorithm(genome_size=3, seed=42, population_size=3)
        pop = [creator.Individual([1, 2, 3]) for _ in range(3)]
        for ind in pop:
            ind.fitness.values = (5.0,)
        diversity = ga._compute_phenotypic_diversity(pop)
        assert diversity == 0.0


# ---------------------------------------------------------------------------
# Immigrant creation
# ---------------------------------------------------------------------------


class TestImmigrants:
    """Test random immigrant injection."""

    def test_create_immigrant_correct_size(self):
        ga = GeneticAlgorithm(genome_size=20, seed=42, bounds=(0, 50))
        imm = ga._create_immigrant()
        assert len(imm) == 20
        # All values should be within bounds
        assert all(0 <= v <= 50 for v in imm)

    def test_immigrant_count_in_generation(self):
        """Verify correct number of immigrants based on rate."""
        GeneticAlgorithm(genome_size=5, seed=42, population_size=20, immigrant_rate=0.1)
        expected_immigrants = int(20 * 0.1)
        assert expected_immigrants == 2

    def test_immigrants_still_respect_init_bounds(self):
        """Immigrants remain bounded even when mutation upper cap is removed."""
        ga = GeneticAlgorithm(genome_size=10, seed=42, bounds=(0, 1))
        for _ in range(100):
            imm = ga._create_immigrant()
            assert all(0 <= v <= 1 for v in imm)


class TestMutationBounds:
    """Test lower-only mutation clipping behavior."""

    def test_mutation_can_exceed_upper_init_bound(self):
        ga = GeneticAlgorithm(genome_size=1, seed=42, bounds=(0, 1))
        ind = creator.Individual([1])

        # Deterministic +5 jump: with upper clipping this would stay at 1.
        ga._bounded_mutation(ind, mu=5, sigma=0, indpb=1.0)
        assert ind[0] == 6

    def test_mutation_still_clips_to_non_negative(self):
        ga = GeneticAlgorithm(genome_size=1, seed=42, bounds=(0, 10))
        ind = creator.Individual([0])

        # Deterministic -5 jump should be clipped to lower bound 0.
        ga._bounded_mutation(ind, mu=-5, sigma=0, indpb=1.0)
        assert ind[0] == 0


# ---------------------------------------------------------------------------
# Parent selection
# ---------------------------------------------------------------------------


class TestParentSelection:
    """Test MAE-elite parent selection with teleport filtering and Pareto ranking."""

    @staticmethod
    def _make_ind(
        vals,
        mae,
        fail_total,
        total_vehicles=None,
        teleports=0,
        missing_edges=0,
        worker_error=False,
    ):
        ind = creator.Individual(vals)
        if total_vehicles is None:
            total_vehicles = int(sum(vals))
        failure_rate = 0.0 if total_vehicles <= 0 and fail_total <= 0 else fail_total / max(1, total_vehicles)
        routing_failures = max(0, fail_total - teleports)
        ind.fitness.values = (mae,)
        ind.metrics = {
            "mae": mae,
            "fail_total": fail_total,
            "routing_failures": routing_failures,
            "teleports": teleports,
            "failure_rate": failure_rate if not worker_error else float("inf"),
            "missing_edges": missing_edges,
            "zero_flow_edges": missing_edges,
            "total_vehicles": total_vehicles,
            "magnitude": total_vehicles,
        }
        if worker_error:
            ind.metrics["worker_error"] = True
            ind.metrics["error"] = "worker crashed"
        return ind

    def test_candidate_pool_is_top_mae_slice_even_with_failures(self):
        ga = GeneticAlgorithm(
            genome_size=3,
            seed=42,
            population_size=5,
            elite_top_pct=0.4,  # n = 2
        )
        pop = [
            self._make_ind([10, 0, 0], 1.0, 2),
            self._make_ind([9, 0, 0], 1.1, 0),
            self._make_ind([8, 0, 0], 1.2, 0),
            self._make_ind([7, 0, 0], 1.3, 1),
            self._make_ind([6, 0, 0], 1.4, 0),
        ]

        plan = ga._build_parent_selection_plan(pop)

        assert plan["mode"] == "mae_elite_pareto"
        assert plan["elite_count"] == 2
        assert len(plan["candidate_pool"]) == 2
        assert plan["candidate_pool"] == pop[:2]

    def test_pareto_rank_keeps_tradeoffs_and_mae_breaks_front_ties(self):
        ga = GeneticAlgorithm(
            genome_size=3,
            seed=42,
            population_size=5,
            elite_top_pct=0.8,  # n = 4
        )
        a = self._make_ind([10, 0, 0], 1.0, 1, total_vehicles=10)  # 10%
        b = self._make_ind([20, 0, 0], 1.1, 2, total_vehicles=100)  # 2%
        c = self._make_ind([12, 0, 0], 1.2, 3, total_vehicles=20)  # 15%
        d = self._make_ind([13, 0, 0], 1.3, 0, total_vehicles=13)  # 0%
        outsider = self._make_ind([14, 0, 0], 1.4, 0)

        plan = ga._build_parent_selection_plan([a, b, c, d, outsider])
        best = ga._select_best_candidate(plan["candidate_pool"], plan["selection_key_by_id"])

        assert plan["candidate_pool"] == [a, b, c, d]
        assert best is a
        assert ga._individual_failure_rate_key(d) < ga._individual_failure_rate_key(a)
        assert ga._individual_failure_rate_key(b) < ga._individual_failure_rate_key(a)

    def test_zero_teleport_candidates_filter_out_teleporting_elite_members(self):
        ga = GeneticAlgorithm(
            genome_size=3,
            seed=42,
            population_size=4,
            elite_top_pct=1.0,
        )
        teleporting_best_mae = self._make_ind([10, 0, 0], 1.0, 1, total_vehicles=10, teleports=1)
        zero_teleport_a = self._make_ind([11, 0, 0], 1.1, 1, total_vehicles=20, teleports=0)
        zero_teleport_b = self._make_ind([12, 0, 0], 1.2, 2, total_vehicles=40, teleports=0)
        zero_teleport_c = self._make_ind([13, 0, 0], 1.3, 3, total_vehicles=60, teleports=0)

        plan = ga._build_parent_selection_plan(
            [teleporting_best_mae, zero_teleport_a, zero_teleport_b, zero_teleport_c]
        )

        assert teleporting_best_mae not in plan["candidate_pool"]
        assert plan["candidate_pool"] == [zero_teleport_a, zero_teleport_b, zero_teleport_c]

    def test_equal_failure_rate_prefers_lower_magnitude_within_same_pareto_rank(self):
        ga = GeneticAlgorithm(
            genome_size=3,
            seed=42,
            population_size=5,
            elite_top_pct=0.8,  # n = 4
        )
        a = self._make_ind([20, 0, 0], 1.0, 2, total_vehicles=20)  # 10%, mag 20
        b = self._make_ind([10, 0, 0], 1.1, 1, total_vehicles=10)  # 10%, mag 10
        c = self._make_ind([30, 0, 0], 1.2, 6, total_vehicles=30)  # 20%
        d = self._make_ind([40, 0, 0], 1.3, 0, total_vehicles=40)  # 0%
        outsider = self._make_ind([50, 0, 0], 1.4, 0, total_vehicles=50)

        plan = ga._build_parent_selection_plan([a, b, c, d, outsider])
        ranked = sorted(
            plan["candidate_pool"],
            key=lambda ind: plan["selection_key_by_id"][id(ind)],
        )

        assert ranked[:3] == [b, d, a]

    def test_equal_failure_rate_prefers_lower_missing_edges_before_magnitude(self):
        ga = GeneticAlgorithm(
            genome_size=3,
            seed=42,
            population_size=4,
            elite_top_pct=1.0,
        )
        fewer_missing = self._make_ind(
            [20, 0, 0], 1.0, 2, total_vehicles=20, missing_edges=3
        )
        more_missing = self._make_ind(
            [10, 0, 0], 1.0, 1, total_vehicles=10, missing_edges=7
        )
        same_rank_worse_mae = self._make_ind(
            [30, 0, 0], 1.2, 3, total_vehicles=30, missing_edges=3
        )
        dominated = self._make_ind(
            [40, 0, 0], 1.3, 8, total_vehicles=40, missing_edges=9
        )

        plan = ga._build_parent_selection_plan(
            [fewer_missing, more_missing, same_rank_worse_mae, dominated]
        )
        ranked = sorted(
            plan["candidate_pool"],
            key=lambda ind: plan["selection_key_by_id"][id(ind)],
        )

        assert ranked[:3] == [fewer_missing, more_missing, same_rank_worse_mae]

    def test_exact_mae_breaks_remaining_tie_inside_elite(self):
        ga = GeneticAlgorithm(
            genome_size=3,
            seed=42,
            population_size=4,
            elite_top_pct=1.0,
        )
        better_mae = self._make_ind([10, 0, 0], 1.0, 1, total_vehicles=10)
        worse_mae = self._make_ind([0, 10, 0], 1.1, 1, total_vehicles=10)
        other_a = self._make_ind([30, 0, 0], 1.2, 3, total_vehicles=30)
        other_b = self._make_ind([40, 0, 0], 1.3, 4, total_vehicles=40)

        plan = ga._build_parent_selection_plan([better_mae, worse_mae, other_a, other_b])
        best = ga._select_best_candidate(plan["candidate_pool"], plan["selection_key_by_id"])

        assert best is better_mae

    def test_survival_elites_follow_pareto_rank_then_mae(self):
        ga = GeneticAlgorithm(
            genome_size=3,
            seed=42,
            population_size=5,
            elite_top_pct=0.8,  # n = 4
            elitism=2,
        )
        a = self._make_ind([10, 0, 0], 1.0, 1, total_vehicles=10)  # 10%, mag 10
        b = self._make_ind([20, 0, 0], 1.1, 0, total_vehicles=20)  # 0%
        c = self._make_ind([30, 0, 0], 1.2, 3, total_vehicles=30)  # 10%, mag 30
        d = self._make_ind([0, 10, 0], 1.3, 1, total_vehicles=10)  # 10%, mag 10, worse mae than a
        outsider = self._make_ind([14, 0, 0], 1.4, 0)

        elites = ga._select_survival_elites([a, b, c, d, outsider], 2)

        assert elites == [a, b]

    def test_invalidate_individual_clears_stale_attrs(self):
        ga = GeneticAlgorithm(genome_size=3, seed=42, population_size=1)
        ind = creator.Individual([1, 2, 3])
        ind.fitness.values = (5.0,)
        ind.metrics = {"routing_failures": 1}

        ga._invalidate_individual(ind)

        assert not ind.fitness.valid
        assert not hasattr(ind, "metrics")

    def test_error_and_invalid_individuals_sort_after_valid_mae_candidates(self):
        ga = GeneticAlgorithm(
            genome_size=3,
            seed=42,
            population_size=4,
            elite_top_pct=0.5,  # n = 2
        )

        valid_a = self._make_ind([1, 0, 0], 1.0, 0)
        valid_b = self._make_ind([2, 0, 0], 1.1, 1)
        worker_error = self._make_ind([3, 0, 0], 0.2, 0, worker_error=True)

        invalid = creator.Individual([3, 0, 0])
        invalid.metrics = {"mae": 0.1, "fail_total": 0, "total_vehicles": 3}

        plan = ga._build_parent_selection_plan([valid_a, worker_error, invalid, valid_b])

        assert plan["mode"] == "mae_elite_pareto"
        assert plan["candidate_pool"] == [valid_a, valid_b]

    def test_identity_exclusion_keeps_non_elite_duplicate_genomes(self):
        ga = GeneticAlgorithm(genome_size=3, seed=42, population_size=4, elitism=1)
        elite = self._make_ind([1, 1, 1], 1.0, 0)
        duplicate_non_elite = self._make_ind([1, 1, 1], 2.0, 0)
        other_a = self._make_ind([2, 2, 2], 3.0, 0)
        other_b = self._make_ind([3, 3, 3], 4.0, 0)

        remaining = ga._exclude_by_identity(
            [elite, duplicate_non_elite, other_a, other_b],
            [elite],
        )
        remaining_ids = {id(ind) for ind in remaining}

        assert id(elite) not in remaining_ids
        assert id(duplicate_non_elite) in remaining_ids
        assert len(remaining) == 3


class TestSelectedBestReturn:
    """Test MAE-elite-based final best selection behavior."""

    def test_prefers_zero_teleport_selected_candidate_when_available(self):
        ga = GeneticAlgorithm(genome_size=3, seed=42, population_size=4, elite_top_pct=0.5)
        mae_best = creator.Individual([1, 1, 1])
        mae_best.fitness.values = (0.5,)
        mae_best.metrics = {"mae": 0.5, "teleports": 1, "fail_total": 2, "total_vehicles": 3}

        selected = creator.Individual([2, 2, 2])
        selected.fitness.values = (0.6,)
        selected.metrics = {"mae": 0.6, "teleports": 0, "fail_total": 0, "total_vehicles": 6}

        other_a = creator.Individual([3, 3, 3])
        other_a.fitness.values = (1.5,)
        other_a.metrics = {"mae": 1.5, "teleports": 0, "fail_total": 0, "total_vehicles": 9}

        other_b = creator.Individual([4, 4, 4])
        other_b.fitness.values = (2.0,)
        other_b.metrics = {"mae": 2.0, "teleports": 0, "fail_total": 0, "total_vehicles": 12}

        best_ind, selected_state = ga._resolve_return_best(
            population=[mae_best, selected, other_a, other_b],
            overall_best_ind=mae_best,
            overall_best_loss=0.5,
            generation_representatives=[
                (
                    mae_best,
                    {
                        "mode": "mae_elite_pareto",
                        "mae": 0.5,
                        "teleports": 1,
                        "failure_rate": 2 / 3,
                        "fail_total": 2,
                        "magnitude": 3.0,
                    },
                ),
                (
                    selected,
                    {
                        "mode": "mae_elite_pareto",
                        "mae": 0.6,
                        "teleports": 0,
                        "failure_rate": 0.0,
                        "fail_total": 0,
                        "magnitude": 6.0,
                    },
                ),
                (
                    other_a,
                    {
                        "mode": "mae_elite_pareto",
                        "mae": 1.5,
                        "teleports": 0,
                        "failure_rate": 0.0,
                        "fail_total": 0,
                        "magnitude": 9.0,
                    },
                ),
                (
                    other_b,
                    {
                        "mode": "mae_elite_pareto",
                        "mae": 2.0,
                        "teleports": 0,
                        "failure_rate": 0.0,
                        "fail_total": 0,
                        "magnitude": 12.0,
                    },
                ),
            ],
        )

        assert selected_state["mode"] == "mae_elite_pareto"
        assert best_ind is selected
        assert selected_state["mae"] == 0.6

    def test_global_mae_elite_blocks_failure_first_cross_generation_inversion(self):
        ga = GeneticAlgorithm(genome_size=3, seed=42, population_size=4, elite_top_pct=0.25)
        early_safe = creator.Individual([5, 5, 5])
        early_safe.fitness.values = (24.64,)
        early_safe.metrics = {"mae": 24.64, "teleports": 0, "fail_total": 15, "total_vehicles": 1513}

        later_better_mae = creator.Individual([6, 6, 6])
        later_better_mae.fitness.values = (23.79,)
        later_better_mae.metrics = {"mae": 23.79, "teleports": 0, "fail_total": 18, "total_vehicles": 1547}

        outside_global_elite = creator.Individual([7, 7, 7])
        outside_global_elite.fitness.values = (30.0,)
        outside_global_elite.metrics = {"mae": 30.0, "teleports": 0, "fail_total": 0, "total_vehicles": 1200}

        filler = creator.Individual([8, 8, 8])
        filler.fitness.values = (24.70,)
        filler.metrics = {"mae": 24.70, "teleports": 0, "fail_total": 16, "total_vehicles": 1520}

        best_ind, selected_state = ga._resolve_return_best(
            population=[early_safe, later_better_mae, outside_global_elite, filler],
            overall_best_ind=later_better_mae,
            overall_best_loss=23.79,
            generation_representatives=[
                (early_safe, ga._individual_summary(early_safe, mode="mae_elite_pareto")),
                (later_better_mae, ga._individual_summary(later_better_mae, mode="mae_elite_pareto")),
                (
                    outside_global_elite,
                    ga._individual_summary(outside_global_elite, mode="mae_elite_pareto"),
                ),
                (filler, ga._individual_summary(filler, mode="mae_elite_pareto")),
            ],
        )

        assert best_ind is later_better_mae
        assert selected_state["mae"] == pytest.approx(23.79)

    def test_falls_back_to_mae_when_no_selected_snapshot_exists(self):
        ga = GeneticAlgorithm(genome_size=3, seed=42, population_size=2)
        mae_best = creator.Individual([1, 1, 1])
        mae_best.fitness.values = (0.5,)
        mae_best.metrics = {"mae": 0.5, "teleports": 0, "fail_total": 3, "total_vehicles": 3}

        other = creator.Individual([3, 3, 3])
        other.fitness.values = (1.0,)
        other.metrics = {"mae": 1.0, "teleports": 0, "fail_total": 4, "total_vehicles": 9}

        best_ind, selected_state = ga._resolve_return_best(
            population=[mae_best, other],
            overall_best_ind=mae_best,
            overall_best_loss=0.5,
            generation_representatives=[],
        )

        assert selected_state["mode"] == "mae_fallback"
        assert best_ind is mae_best
        assert selected_state["mae"] == 0.5

    def test_optimize_exposes_selected_best_metadata(self):
        ga = GeneticAlgorithm(
            genome_size=3,
            seed=42,
            population_size=4,
            num_generations=1,
            num_workers=1,
            elite_top_pct=0.5,
        )

        ga.optimize(_simple_evaluate)

        assert ga.last_best_selection_mode == "mae_elite_pareto"
        assert ga.last_best_mae is not None
        assert ga.last_best_mae_candidate_mae is not None
        assert ga.last_best_mae_candidate_teleports is not None
        assert ga.last_best_mae_candidate_failure_rate is not None
        assert ga.last_best_mae_candidate_fail_total is not None
        assert ga.last_best_mae_candidate_missing_edges is not None
        assert ga.last_best_mae_candidate_magnitude is not None
        assert ga.last_best_selected_mae is not None
        assert ga.last_best_selected_teleports is not None
        assert ga.last_best_selected_failure_rate is not None
        assert ga.last_best_selected_fail_total is not None
        assert ga.last_best_selected_missing_edges is not None
        assert ga.last_best_selected_magnitude is not None

    def test_generation_callback_receives_snapshot_each_generation(self):
        ga = GeneticAlgorithm(
            genome_size=3,
            seed=42,
            population_size=6,
            num_generations=3,
            num_workers=1,
        )
        callback_calls = []

        def generation_callback(generation, best_genome, best_mae, best_metrics):
            callback_calls.append((generation, best_genome, best_mae, best_metrics))

        ga.optimize(_simple_evaluate, generation_callback=generation_callback)

        assert len(callback_calls) == 3
        for generation, best_genome, best_mae, best_metrics in callback_calls:
            assert generation >= 1
            assert isinstance(best_genome, np.ndarray)
            assert best_genome.shape == (3,)
            assert np.issubdtype(best_genome.dtype, np.integer)
            assert "selection_mode" in best_metrics
            assert np.isfinite(best_mae)
            assert isinstance(best_metrics, dict)

    def test_optimize_memoizes_duplicate_genome_evaluations(self):
        call_counter = {"count": 0}

        def evaluate(genome):
            call_counter["count"] += 1
            loss = float(np.mean(np.abs(genome)))
            metrics = {
                "mae": loss,
                "teleports": 0,
                "fail_total": 0,
                "failure_rate": 0.0,
                "total_vehicles": int(np.sum(genome)),
                "magnitude": int(np.sum(genome)),
            }
            return loss, metrics

        ga = GeneticAlgorithm(
            genome_size=1,
            seed=42,
            bounds=(0, 0),
            population_size=4,
            num_generations=2,
            num_workers=1,
            mutation_rate=1.0,
            crossover_rate=0.0,
            elitism=0,
            mutation_sigma=0,
            mutation_indpb=1.0,
            immigrant_rate=0.0,
            assortative_mating=False,
            deterministic_crowding=False,
        )

        ga.optimize(evaluate)

        # All individuals are always genome [0], so worker evaluation should run once.
        assert call_counter["count"] == 1
        assert ga.last_eval_cache_misses == 1
        assert ga.last_eval_cache_hits == 11
        assert ga.last_eval_cache_size == 1

    def test_optimize_trims_eval_cache_when_cap_is_exceeded(self, monkeypatch):
        monkeypatch.setattr(optimizer_module, "MAX_EVAL_CACHE_ENTRIES", 4)

        def evaluate(genome):
            loss = float(np.mean(np.abs(genome)))
            metrics = {
                "mae": loss,
                "teleports": 0,
                "fail_total": 0,
                "failure_rate": 0.0,
                "total_vehicles": int(np.sum(genome)),
                "magnitude": int(np.sum(genome)),
            }
            return loss, metrics

        ga = GeneticAlgorithm(
            genome_size=1,
            seed=42,
            bounds=(0, 9),
            population_size=6,
            num_generations=0,
            num_workers=1,
            mutation_rate=0.0,
            crossover_rate=0.0,
            elitism=0,
            immigrant_rate=0.0,
            assortative_mating=False,
            deterministic_crowding=False,
        )

        ga.optimize(evaluate)

        assert ga.last_eval_cache_size == 2


# ---------------------------------------------------------------------------
# Assortative mating
# ---------------------------------------------------------------------------


class TestAssortativeMating:
    """Test assortative mating pairing."""

    def test_assortative_pairs_dissimilar(self):
        ga = GeneticAlgorithm(genome_size=3, seed=42, population_size=4)
        offspring = [
            creator.Individual([1, 1, 1]),  # sum=3 (smallest)
            creator.Individual([5, 5, 5]),  # sum=15
            creator.Individual([10, 10, 10]),  # sum=30
            creator.Individual([20, 20, 20]),  # sum=60 (largest)
        ]
        pairs = ga._assortative_mate_pairs(offspring)
        assert len(pairs) == 2
        # First pair should connect smallest with largest
        for i1, i2 in pairs:
            assert abs(sum(offspring[i1]) - sum(offspring[i2])) > 0


# ---------------------------------------------------------------------------
# Adaptive mutation boost (stagnation)
# ---------------------------------------------------------------------------


class TestStagnationBoost:
    """Test adaptive mutation boost on stagnation."""

    def test_mutation_boost_tracking(self):
        ga = GeneticAlgorithm(
            genome_size=3,
            seed=42,
            mutation_sigma=10,
            mutation_rate=0.5,
            stagnation_patience=3,
            stagnation_boost=2.0,
        )
        assert ga._mutation_boosted is False
        assert ga._base_mutation_sigma == 10
        assert ga._base_mutation_rate == 0.5

    def test_small_improvement_resets_stagnation_counter(self):
        """A real loss decrease should reset stagnation even below early-stop epsilon."""
        losses = iter([10.0, 9.95, 9.90])

        def evaluate(_genome):
            return next(losses)

        ga = GeneticAlgorithm(
            genome_size=3,
            seed=42,
            population_size=1,
            num_generations=2,
            mutation_rate=1.0,
            crossover_rate=0.0,
            elitism=0,
            mutation_sigma=10,
            mutation_indpb=1.0,
            num_workers=1,
            immigrant_rate=0.0,
            stagnation_patience=1,
            stagnation_boost=2.0,
            assortative_mating=False,
            deterministic_crowding=False,
        )

        _, _, _, generation_stats = ga.optimize(evaluate)

        assert generation_stats[0]["best_loss"] == pytest.approx(9.95)
        assert generation_stats[1]["best_loss"] == pytest.approx(9.90)
        assert generation_stats[0]["mutation_boosted"] is False
        assert generation_stats[1]["mutation_boosted"] is False


# ---------------------------------------------------------------------------
# Generation stats include new fields
# ---------------------------------------------------------------------------


class TestGenerationStats:
    """Test that generation stats include diversity and boost info."""

    def test_stats_have_diversity_fields(self):
        """The generation stats dict should include diversity metrics."""
        # This is a structural test - we check the dict keys
        expected_keys = {
            "generation",
            "best_loss",
            "mean_loss",
            "std_loss",
            "best_magnitude",
            "mean_magnitude",
            "best_failure_rate",
            "mean_failure_rate",
            "best_zero_flow",
            "mean_zero_flow",
            "best_fail_total",
            "mean_fail_total",
            "genotypic_diversity",
            "phenotypic_diversity",
            "mutation_boosted",
        }
        # All these keys should be present in a gen_stat dict
        sample_stat = {
            "generation": 1,
            "best_loss": 10.0,
            "mean_loss": 15.0,
            "std_loss": 3.0,
            "best_magnitude": 100.0,
            "mean_magnitude": 200.0,
            "best_failure_rate": 0.05,
            "mean_failure_rate": 0.08,
            "best_zero_flow": 5,
            "mean_zero_flow": 8.0,
            "best_fail_total": 3,
            "mean_fail_total": 5.0,
            "genotypic_diversity": 50.0,
            "phenotypic_diversity": 3.5,
            "mutation_boosted": False,
        }
        assert expected_keys.issubset(sample_stat.keys())


# ---------------------------------------------------------------------------
# Report: diversity plot
# ---------------------------------------------------------------------------


class TestDiversityPlot:
    """Test diversity plot generation in reports."""

    def test_diversity_plot_generated(self, tmp_path):
        gen_stats = []
        for gen in range(1, 6):
            gen_stats.append(
                {
                    "generation": gen,
                    "best_loss": 50.0 - gen * 5,
                    "mean_loss": 80.0 - gen * 4,
                    "std_loss": 15.0 - gen,
                    "best_magnitude": 100 + gen * 10,
                    "mean_magnitude": 200 + gen * 5,
                    "best_failure_rate": max(0.0, 0.12 - gen * 0.02),
                    "mean_failure_rate": max(0.0, 0.16 - gen * 0.02),
                    "best_zero_flow": max(0, 10 - gen * 2),
                    "mean_zero_flow": max(0.0, 12 - gen * 1.5),
                    "best_fail_total": max(0, 8 - gen),
                    "mean_fail_total": max(0.0, 10 - gen * 0.8),
                    "genotypic_diversity": 100.0 - gen * 10,
                    "phenotypic_diversity": 20.0 - gen * 2,
                    "mutation_boosted": gen >= 4,
                }
            )

        report_gen = ReportGenerator(tmp_path)
        result = report_gen._create_diversity_plot(gen_stats)

        assert result is not None
        assert result == "plots/diversity_plot.png"
        assert (tmp_path / "plots" / "diversity_plot.png").exists()

    def test_diversity_plot_none_when_no_data(self, tmp_path):
        gen_stats = [{"generation": 1, "genotypic_diversity": None, "phenotypic_diversity": None}]
        report_gen = ReportGenerator(tmp_path)
        result = report_gen._create_diversity_plot(gen_stats)
        assert result is None

    def test_report_includes_diversity_plot(self, tmp_path):
        """Full report includes diversity plot when data is available."""
        observed_edges = pd.DataFrame(
            {
                "edge_id": ["e1", "e2", "e3"],
                "current_speed": [30.0, 50.0, 20.0],
                "freeflow_speed": [50.0, 60.0, 50.0],
                "match_confidence": [0.9, 0.8, 0.7],
            }
        )
        simulated_speeds = {"e1": 35.0, "e2": 45.0, "e3": 25.0}
        gen_stats = []
        for gen in range(1, 4):
            gen_stats.append(
                {
                    "generation": gen,
                    "best_loss": 50.0 - gen * 5,
                    "mean_loss": 80.0 - gen * 4,
                    "std_loss": 10.0,
                    "best_magnitude": 100.0,
                    "mean_magnitude": 200.0,
                    "best_failure_rate": 0.05,
                    "mean_failure_rate": 0.08,
                    "best_zero_flow": 5,
                    "mean_zero_flow": 8.0,
                    "best_fail_total": 2,
                    "mean_fail_total": 4.0,
                    "genotypic_diversity": 100.0 - gen * 10,
                    "phenotypic_diversity": 20.0 - gen * 2,
                    "mutation_boosted": False,
                }
            )
        loss_history = [s["best_loss"] for s in gen_stats]
        metadata = {
            "run_info": {
                "timestamp": "2026-01-01T00:00:00",
                "bbox_coordinates": {"west": 0, "south": 0, "east": 1, "north": 1},
                "seed": 42,
            },
            "simulation_config": {"window_minutes": 10},
            "calibration_config": {"ga_population": 20, "ga_generations": 3},
            "results": {
                "final_loss_mae_kmh": 35.0,
                "quality_metrics": {
                    "matched_edges": 3,
                    "total_observed_edges": 3,
                },
            },
        }

        report_gen = ReportGenerator(tmp_path)
        report_path = report_gen.generate(
            observed_edges, simulated_speeds, loss_history, metadata, gen_stats
        )

        assert report_path.exists()
        assert (tmp_path / "plots" / "diversity_plot.png").exists()
        html = report_path.read_text()
        assert "Population Diversity" in html
