import numpy as np

from qga import PermutationQGA


def tsp_fitness(route, distance):
    length = 0.0
    for i in range(1, len(route)):
        length += distance[route[i - 1], route[i]]
    length += distance[route[-1], route[0]]
    return -length


if __name__ == "__main__":
    coords = np.array(
        [
            [7.68, 45.06],
            [9.19, 45.46],
            [8.93, 44.40],
            [12.33, 45.43],
            [11.34, 44.49],
            [11.25, 43.76],
            [13.95, 42.31],
            [12.48, 41.89],
            [14.24, 40.83],
        ]
    )

    diff = coords[:, None, :] - coords[None, :, :]
    distance = np.sqrt(np.sum(diff**2, axis=-1))

    optimizer = PermutationQGA(
        n_items=distance.shape[0],
        population_size=40,
        generations=500,
        fitness_func=tsp_fitness,
        fitness_inputs=distance,
        seed=4321,
        verbose=True,
    )

    result = optimizer.run()

    print("Best route:", result.best_solution)
    print("Best route length:", -result.best_fitness)
