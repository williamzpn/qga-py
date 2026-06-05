import numpy as np

from qga import PermutationQGA


def test_permutation_qga_returns_valid_permutation():
    def fitness(route, _):
        return -float(np.sum(np.abs(np.diff(route))))

    optimizer = PermutationQGA(
        n_items=10,
        population_size=8,
        generations=20,
        fitness_func=fitness,
        seed=2,
    )

    result = optimizer.run()

    assert result.best_solution.shape == (10,)
    assert sorted(result.best_solution.tolist()) == list(range(10))
    assert len(result.history_best) == 20
