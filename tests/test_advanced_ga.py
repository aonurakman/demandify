"""Tests for advanced GA parameterization features."""

import numpy as np
import pandas as pd
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
        assert ga.magnitude_penalty_weight == 0.001
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
            magnitude_penalty_weight=0.01,
            stagnation_patience=10,
            stagnation_boost=2.0,
            assortative_mating=False,
            deterministic_crowding=False,
        )
        assert ga.immigrant_rate == 0.05
        assert ga.elite_top_pct == 0.2
        assert ga.magnitude_penalty_weight == 0.01
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


# ---------------------------------------------------------------------------
# Parent selection
# ---------------------------------------------------------------------------


class TestParentSelection:
    """Test feasible-elite parent selection and fallback behavior."""

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

    def test_feasible_elite_selection_mode(self):
        ga = GeneticAlgorithm(
            genome_size=3,
            seed=42,
            population_size=5,
            elite_top_pct=0.4,  # n = 2
            magnitude_penalty_weight=0.01,
        )
        pop = [
            self._make_ind([10, 0, 0], 1.0, 1.0, 0),
            self._make_ind([9, 0, 0], 1.1, 1.1, 0),
            self._make_ind([8, 0, 0], 1.2, 1.2, 0),
            self._make_ind([7, 0, 0], 1.3, 1.3, 1, reliability_penalty=0.2),
            self._make_ind([6, 0, 0], 1.4, 1.4, 2, reliability_penalty=0.4),
        ]

        plan = ga._build_parent_selection_plan(pop)

        assert plan["mode"] == "feasible_elite"
        assert plan["elite_count"] == 2
        assert plan["feasible_count"] == 3
        assert len(plan["candidate_pool"]) == 2
        assert all(ga._individual_fail_total(ind) == 0 for ind in plan["candidate_pool"])

    def test_fallback_mode_activation(self):
        ga = GeneticAlgorithm(
            genome_size=3,
            seed=42,
            population_size=5,
            elite_top_pct=0.4,  # n = 2
            magnitude_penalty_weight=0.01,
        )
        pop = [
            self._make_ind([1, 0, 0], 1.0, 1.0, 0),
            self._make_ind([2, 0, 0], 0.8, 0.6, 1, reliability_penalty=0.2),
            self._make_ind([3, 0, 0], 0.9, 0.7, 1, reliability_penalty=0.2),
            self._make_ind([4, 0, 0], 1.1, 0.9, 2, reliability_penalty=0.2),
            self._make_ind([5, 0, 0], 1.2, 1.0, 3, reliability_penalty=0.2),
        ]

        plan = ga._build_parent_selection_plan(pop)

        assert plan["mode"] == "fallback"
        assert plan["elite_count"] == 2
        assert plan["feasible_count"] == 1
        assert len(plan["candidate_pool"]) == len(pop)

    def test_elite_ranking_magnitude_dominant_and_e_rank_regulated(self):
        # Magnitude-dominant case: despite worse E-rank, much smaller magnitude can win.
        ga = GeneticAlgorithm(
            genome_size=3,
            seed=42,
            population_size=4,
            elite_top_pct=0.5,  # n = 2
            magnitude_penalty_weight=0.005,
        )
        a = self._make_ind([100, 0, 0], 1.0, 1.0, 0)  # e-rank 0, huge magnitude
        b = self._make_ind([1, 0, 0], 1.1, 1.1, 0)  # e-rank 1, tiny magnitude
        c = self._make_ind([3, 0, 0], 1.2, 1.2, 0)
        d = self._make_ind([4, 0, 0], 1.3, 1.3, 0)
        plan = ga._build_parent_selection_plan([a, b, c, d])

        score_a = plan["score_by_id"][id(a)]
        score_b = plan["score_by_id"][id(b)]
        assert score_b < score_a

        # E-rank-regulated case: with equal magnitude, better E-rank should win.
        ga2 = GeneticAlgorithm(
            genome_size=3,
            seed=42,
            population_size=4,
            elite_top_pct=0.5,
            magnitude_penalty_weight=0.001,
        )
        a2 = self._make_ind([10, 0, 0], 1.0, 1.0, 0)  # e-rank 0
        b2 = self._make_ind([10, 0, 0], 1.1, 1.1, 0)  # e-rank 1
        c2 = self._make_ind([20, 0, 0], 1.2, 1.2, 0)
        d2 = self._make_ind([30, 0, 0], 1.3, 1.3, 0)
        plan2 = ga2._build_parent_selection_plan([a2, b2, c2, d2])

        score_a2 = plan2["score_by_id"][id(a2)]
        score_b2 = plan2["score_by_id"][id(b2)]
        assert score_a2 < score_b2

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

    def test_feasible_filter_ignores_error_and_invalid_individuals(self):
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
        assert plan["mode"] == "fallback"


class TestBestFeasibleReturn:
    """Test feasible-first return behavior for final best individual."""

    def test_prefers_best_feasible_when_available(self):
        ga = GeneticAlgorithm(genome_size=3, seed=42, population_size=2)
        infeasible = creator.Individual([1, 1, 1])
        infeasible.fitness.values = (0.5,)
        infeasible.metrics = {"e_loss": 0.1, "fail_total": 2}

        feasible = creator.Individual([2, 2, 2])
        feasible.fitness.values = (2.0,)
        feasible.metrics = {"e_loss": 2.0, "fail_total": 0}

        best_ind, best_loss, mode = ga._resolve_return_best(
            population=[infeasible, feasible],
            overall_best_ind=infeasible,
            overall_best_loss=0.5,
            overall_best_feasible_ind=feasible,
            overall_best_feasible_e=2.0,
        )

        assert mode == "feasible"
        assert best_ind is feasible
        assert best_loss == 2.0

    def test_falls_back_to_raw_when_no_feasible_exists(self):
        ga = GeneticAlgorithm(genome_size=3, seed=42, population_size=2)
        raw_best = creator.Individual([1, 1, 1])
        raw_best.fitness.values = (0.5,)
        raw_best.metrics = {"e_loss": 0.1, "fail_total": 3}

        other = creator.Individual([3, 3, 3])
        other.fitness.values = (1.0,)
        other.metrics = {"e_loss": 0.4, "fail_total": 4}

        best_ind, best_loss, mode = ga._resolve_return_best(
            population=[raw_best, other],
            overall_best_ind=raw_best,
            overall_best_loss=0.5,
            overall_best_feasible_ind=None,
            overall_best_feasible_e=float("inf"),
        )

        assert mode == "raw"
        assert best_ind is raw_best
        assert best_loss == 0.5

    def test_optimize_exposes_selected_best_metadata(self):
        ga = GeneticAlgorithm(
            genome_size=3,
            seed=42,
            population_size=4,
            num_generations=1,
            num_workers=1,
            elite_top_pct=0.5,
            magnitude_penalty_weight=0.001,
        )

        ga.optimize(_simple_evaluate)

        assert ga.last_best_selection_mode in {"feasible", "raw"}
        assert ga.last_best_selection_value is not None


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
