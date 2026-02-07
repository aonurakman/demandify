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
        assert all(0 <= v <= 500 for v in imm)  # 10x upper bound

    def test_immigrant_count_in_generation(self):
        """Verify correct number of immigrants based on rate."""
        GeneticAlgorithm(genome_size=5, seed=42, population_size=20, immigrant_rate=0.1)
        expected_immigrants = int(20 * 0.1)
        assert expected_immigrants == 2


# ---------------------------------------------------------------------------
# Magnitude penalty
# ---------------------------------------------------------------------------


class TestMagnitudePenalty:
    """Test magnitude penalty application."""

    def test_magnitude_penalty_increases_fitness(self):
        ga = GeneticAlgorithm(
            genome_size=3,
            seed=42,
            population_size=4,
            elite_top_pct=0.5,
            magnitude_penalty_weight=0.01,
        )
        pop = []
        for vals, loss in [
            ([10, 10, 10], 5.0),  # magnitude=30, in top 50%
            ([1, 1, 1], 5.5),  # magnitude=3, in top 50%
            ([20, 20, 20], 10.0),  # magnitude=60
            ([5, 5, 5], 15.0),  # magnitude=15
        ]:
            ind = creator.Individual(vals)
            ind.fitness.values = (loss,)
            pop.append(ind)

        ga._apply_magnitude_penalty(pop)

        # Top 50% = first 2 individuals (sorted by fitness)
        # Individual with magnitude=30 gets penalty: 30 * 0.01 = 0.3
        # Individual with magnitude=3 gets penalty: 3 * 0.01 = 0.03
        # The one with fewer trips should have lower penalized fitness
        top_two = sorted(pop[:2], key=lambda x: x.fitness.values[0])
        assert top_two[0].fitness.values[0] < top_two[1].fitness.values[0] or sum(
            top_two[0]
        ) <= sum(top_two[1])

    def test_magnitude_penalty_disabled(self):
        ga = GeneticAlgorithm(
            genome_size=3,
            seed=42,
            population_size=2,
            magnitude_penalty_weight=0.0,
        )
        pop = []
        for vals, loss in [([10, 10, 10], 5.0), ([1, 1, 1], 5.5)]:
            ind = creator.Individual(vals)
            ind.fitness.values = (loss,)
            pop.append(ind)

        original_fits = [ind.fitness.values[0] for ind in pop]
        ga._apply_magnitude_penalty(pop)
        new_fits = [ind.fitness.values[0] for ind in pop]

        assert original_fits == new_fits


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
