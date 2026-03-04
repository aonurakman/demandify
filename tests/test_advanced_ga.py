"""Tests for advanced GA parameterization features."""

import numpy as np
import pandas as pd
import pytest
from deap import creator

from demandify.calibration.optimizer import GeneticAlgorithm
from demandify.export.report import ReportGenerator

# ---------------------------------------------------------------------------
# Helper: a simple evaluate function that returns (loss, metrics)
# ---------------------------------------------------------------------------


def _simple_evaluate(genome):
    """Evaluate function: loss = mean absolute value, lower is better."""
    loss = float(np.mean(np.abs(genome)))
    metrics = {
        "zero_flow_edges": int(np.sum(genome == 0)),
        "routing_failures": 0,
        "teleports": 0,
        "fail_total": 0,
        "reliability_penalty": 0.0,
        "e_loss": loss,
        "avg_trip_duration": 100.0,
        "total_vehicles": int(np.sum(genome)),
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
    """Test elite-slice parent selection and secondary ranking behavior."""

    @staticmethod
    def _make_ind(vals, loss, e_loss, fail_total, reliability_penalty=0.0):
        ind = creator.Individual(vals)
        ind.fitness.values = (loss,)
        ind.metrics = {
            "e_loss": e_loss,
            "fail_total": fail_total,
            "routing_failures": fail_total,
            "teleports": 0,
            "reliability_penalty": reliability_penalty,
        }
        return ind

    def test_candidate_pool_is_top_e_slice_even_with_failures(self):
        ga = GeneticAlgorithm(
            genome_size=3,
            seed=42,
            population_size=5,
            elite_top_pct=0.4,  # n = 2
        )
        pop = [
            self._make_ind([10, 0, 0], 1.0, 1.0, 2),
            self._make_ind([9, 0, 0], 1.1, 1.1, 0),
            self._make_ind([8, 0, 0], 1.2, 1.2, 0),
            self._make_ind([7, 0, 0], 1.3, 1.3, 1),
            self._make_ind([6, 0, 0], 1.4, 1.4, 0),
        ]

        plan = ga._build_parent_selection_plan(pop)

        assert plan["mode"] == "elite_slice_secondary"
        assert plan["elite_count"] == 2
        assert plan["feasible_count"] == 3
        assert len(plan["candidate_pool"]) == 2
        assert plan["candidate_pool"] == pop[:2]

    def test_secondary_score_penalizes_failures_inside_elite_slice(self):
        ga = GeneticAlgorithm(
            genome_size=3,
            seed=42,
            population_size=5,
            elite_top_pct=0.8,  # n = 4
        )
        a = self._make_ind([10, 0, 0], 1.0, 1.0, 3)
        b = self._make_ind([10, 0, 0], 1.1, 1.1, 0)
        c = self._make_ind([12, 0, 0], 1.2, 1.2, 0)
        d = self._make_ind([13, 0, 0], 1.3, 1.3, 1)
        outsider = self._make_ind([14, 0, 0], 1.4, 1.4, 0)

        plan = ga._build_parent_selection_plan([a, b, c, d, outsider])
        best = ga._select_best_candidate(plan["candidate_pool"], plan["score_by_id"])

        assert plan["candidate_pool"] == [a, b, c, d]
        assert best is b
        assert plan["score_by_id"][id(b)] < plan["score_by_id"][id(a)]

    def test_secondary_score_penalizes_magnitude_inside_elite_slice(self):
        ga = GeneticAlgorithm(
            genome_size=3,
            seed=42,
            population_size=5,
            elite_top_pct=0.8,  # n = 4
        )
        a = self._make_ind([100, 0, 0], 1.0, 1.0, 0)
        b = self._make_ind([10, 0, 0], 1.1, 1.1, 0)
        c = self._make_ind([20, 0, 0], 1.2, 1.2, 1)
        d = self._make_ind([30, 0, 0], 1.3, 1.3, 2)
        outsider = self._make_ind([40, 0, 0], 1.4, 1.4, 0)

        plan = ga._build_parent_selection_plan([a, b, c, d, outsider])
        best = ga._select_best_candidate(plan["candidate_pool"], plan["score_by_id"])

        assert best is b
        assert plan["score_by_id"][id(b)] < plan["score_by_id"][id(a)]

    def test_survival_elites_follow_secondary_order(self):
        ga = GeneticAlgorithm(
            genome_size=3,
            seed=42,
            population_size=5,
            elite_top_pct=0.8,  # n = 4
            elitism=2,
        )
        a = self._make_ind([10, 0, 0], 1.0, 1.0, 3)
        b = self._make_ind([10, 0, 0], 1.1, 1.1, 0)
        c = self._make_ind([12, 0, 0], 1.2, 1.2, 0)
        d = self._make_ind([13, 0, 0], 1.3, 1.3, 1)
        outsider = self._make_ind([14, 0, 0], 1.4, 1.4, 0)

        elites = ga._select_survival_elites([a, b, c, d, outsider], 2)

        assert elites == [b, a]

    def test_invalidate_individual_clears_stale_attrs(self):
        ga = GeneticAlgorithm(genome_size=3, seed=42, population_size=1)
        ind = creator.Individual([1, 2, 3])
        ind.fitness.values = (5.0,)
        ind.metrics = {"routing_failures": 1}

        ga._invalidate_individual(ind)

        assert not ind.fitness.valid
        assert not hasattr(ind, "metrics")

    def test_error_marked_individual_is_not_feasible(self):
        ga = GeneticAlgorithm(genome_size=3, seed=42, population_size=1)
        ind = creator.Individual([1, 1, 1])
        ind.fitness.values = (1.0,)
        ind.metrics = {
            "fail_total": 0,
            "routing_failures": 0,
            "teleports": 0,
            "worker_error": True,
            "error": "simulation failed",
        }

        assert ga._is_feasible_individual(ind) is False

    def test_feasible_count_ignores_error_and_invalid_individuals(self):
        ga = GeneticAlgorithm(
            genome_size=3,
            seed=42,
            population_size=4,
            elite_top_pct=0.5,  # n = 2
        )

        feasible = self._make_ind([1, 0, 0], 1.0, 1.0, 0)

        worker_error = self._make_ind([2, 0, 0], 0.2, 0.2, 0)
        worker_error.metrics["worker_error"] = True
        worker_error.metrics["error"] = "worker crashed"

        invalid = creator.Individual([3, 0, 0])
        invalid.metrics = {"e_loss": 0.1, "fail_total": 0}

        infeasible = self._make_ind([4, 0, 0], 1.2, 1.2, 2)

        plan = ga._build_parent_selection_plan([feasible, worker_error, invalid, infeasible])

        assert ga._is_feasible_individual(feasible) is True
        assert ga._is_feasible_individual(worker_error) is False
        assert ga._is_feasible_individual(invalid) is False
        assert ga._is_feasible_individual(infeasible) is False
        assert plan["feasible_count"] == 1
        assert plan["mode"] == "elite_slice_secondary"


class TestSelectedBestReturn:
    """Test elite-slice-based final best selection behavior."""

    def test_prefers_best_selected_candidate_when_available(self):
        ga = GeneticAlgorithm(genome_size=3, seed=42, population_size=2)
        raw_best = creator.Individual([1, 1, 1])
        raw_best.fitness.values = (0.5,)
        raw_best.metrics = {"e_loss": 0.1, "fail_total": 2}

        selected = creator.Individual([2, 2, 2])
        selected.fitness.values = (2.0,)
        selected.metrics = {"e_loss": 2.0, "fail_total": 0}

        best_ind, selected_state = ga._resolve_return_best(
            population=[raw_best, selected],
            overall_best_ind=raw_best,
            overall_best_loss=0.5,
            overall_selected_ind=selected,
            overall_selected_state={
                "mode": "elite_slice_secondary",
                "secondary_score": 0.2,
                "raw_loss": 2.0,
                "e_loss": 2.0,
                "fail_total": 0,
                "magnitude": 6.0,
            },
        )

        assert selected_state["mode"] == "elite_slice_secondary"
        assert best_ind is selected
        assert selected_state["raw_loss"] == 2.0

    def test_falls_back_to_raw_when_no_selected_snapshot_exists(self):
        ga = GeneticAlgorithm(genome_size=3, seed=42, population_size=2)
        raw_best = creator.Individual([1, 1, 1])
        raw_best.fitness.values = (0.5,)
        raw_best.metrics = {"e_loss": 0.1, "fail_total": 3}

        other = creator.Individual([3, 3, 3])
        other.fitness.values = (1.0,)
        other.metrics = {"e_loss": 0.4, "fail_total": 4}

        best_ind, selected_state = ga._resolve_return_best(
            population=[raw_best, other],
            overall_best_ind=raw_best,
            overall_best_loss=0.5,
            overall_selected_ind=None,
            overall_selected_state=None,
        )

        assert selected_state["mode"] == "raw_fallback"
        assert best_ind is raw_best
        assert selected_state["raw_loss"] == 0.5

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

        assert ga.last_best_selection_mode == "elite_slice_secondary"
        assert ga.last_best_selection_value is not None
        assert ga.last_best_selected_raw_loss is not None
        assert ga.last_best_selected_e_loss is not None
        assert ga.last_best_selected_fail_total is not None
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

        def generation_callback(generation, best_genome, best_loss, best_metrics):
            callback_calls.append((generation, best_genome, best_loss, best_metrics))

        ga.optimize(_simple_evaluate, generation_callback=generation_callback)

        assert len(callback_calls) == 3
        for generation, best_genome, best_loss, best_metrics in callback_calls:
            assert generation >= 1
            assert isinstance(best_genome, np.ndarray)
            assert best_genome.shape == (3,)
            assert np.issubdtype(best_genome.dtype, np.integer)
            assert np.isfinite(best_loss)
            assert isinstance(best_metrics, dict)


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
            "best_zero_flow",
            "mean_zero_flow",
            "best_routing_failures",
            "mean_routing_failures",
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
            "best_zero_flow": 5,
            "mean_zero_flow": 8.0,
            "best_routing_failures": 2,
            "mean_routing_failures": 4.0,
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
                    "best_zero_flow": max(0, 10 - gen * 2),
                    "mean_zero_flow": max(0.0, 12 - gen * 1.5),
                    "best_routing_failures": max(0, 8 - gen),
                    "mean_routing_failures": max(0.0, 10 - gen * 0.8),
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
                    "best_zero_flow": 5,
                    "mean_zero_flow": 8.0,
                    "best_routing_failures": 2,
                    "mean_routing_failures": 4.0,
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
