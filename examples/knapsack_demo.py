import numpy as np

from qga import BinaryQGA


def knapsack_fitness(bits, inputs):
    weights, max_weight = inputs
    total_items = int(np.sum(bits))
    total_weight = float(np.sum(weights[bits == 1]))

    if total_weight > max_weight:
        return total_items - (total_weight - max_weight)

    return total_items


if __name__ == "__main__":
    rng = np.random.default_rng(1234)
    weights = rng.normal(loc=200, scale=80, size=500)
    max_weight = np.sum(weights) / 5

    optimizer = BinaryQGA(
        genome_length=len(weights),
        population_size=20,
        generations=500,
        fitness_func=knapsack_fitness,
        fitness_inputs=(weights, max_weight),
        seed=1234,
        verbose=True,
    )

    result = optimizer.run()

    print("Best fitness:", result.best_fitness)
    print("Number selected:", np.sum(result.best_solution))
    print("Total weight:", np.sum(weights[result.best_solution == 1]))
    print("Max weight:", max_weight)
