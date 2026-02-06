import numpy as np

from demandify.calibration import worker


def test_stable_seed_reproducible():
    genome = np.array([1, 2, 3])
    seed_a = worker._stable_seed(genome, base_seed=42)
    seed_b = worker._stable_seed(genome.copy(), base_seed=42)
    assert seed_a == seed_b


def test_stable_seed_changes_with_genome_or_base():
    genome = np.array([1, 2, 3])
    genome2 = np.array([1, 2, 4])

    seed_a = worker._stable_seed(genome, base_seed=42)
    seed_b = worker._stable_seed(genome2, base_seed=42)
    seed_c = worker._stable_seed(genome, base_seed=7)

    assert seed_a != seed_b
    assert seed_a != seed_c
