import numpy as np

from qga import BinaryQGA


def test_binary_qga_returns_bits():
    def fitness(bits, _):
        return float(np.sum(bits))

    optimizer = BinaryQGA(
        genome_length=12,
        population_size=8,
        generations=20,
        fitness_func=fitness,
        seed=1,
    )

    result = optimizer.run()

    assert result.best_solution.shape == (12,)
    assert set(result.best_solution.tolist()).issubset({0, 1})
    assert len(result.history_best) == 20
