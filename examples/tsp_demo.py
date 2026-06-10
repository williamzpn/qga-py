import numpy as np
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qga import PermutationQGA


def canonical_tsp_route(route):
    route = list(route)
    zero_index = route.index(0)
    route = route[zero_index:] + route[:zero_index]

    reversed_route = [route[0]] + list(reversed(route[1:]))
    if tuple(reversed_route) < tuple(route):
        route = reversed_route

    return tuple(route)


def tsp_route_length(route, distance):
    length = 0.0
    for i in range(1, len(route)):
        length += distance[route[i - 1], route[i]]
    length += distance[route[-1], route[0]]
    return length


def tsp_fitness(route, inputs):
    canonical_route = canonical_tsp_route(route)
    cache = inputs["cache"]

    if canonical_route not in cache:
        cache[canonical_route] = tsp_route_length(canonical_route, inputs["distance"])

    length = cache[canonical_route]
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
            [10.99, 45.44],
            [9.67, 45.70],
            [8.62, 45.82],
        ]
    )

    diff = coords[:, None, :] - coords[None, :, :]
    distance = np.sqrt(np.sum(diff**2, axis=-1))
    fitness_inputs = {
        "distance": distance,
        "cache": {},
    }

    population_size = 40
    generations = 500

    optimizer = PermutationQGA(
        n_items=distance.shape[0],
        population_size=population_size,
        generations=generations,
        fitness_func=tsp_fitness,
        fitness_inputs=fitness_inputs,
        seed=4321,
        mutation_rate_start=0.15,
        mutation_rate_end=0.01,
        verbose=True,
    )

    start = time.perf_counter()
    result = optimizer.run()
    elapsed = time.perf_counter() - start
    route_tries = population_size * generations
    best_route = np.array(canonical_tsp_route(result.best_solution))

    print("Number of cities:", distance.shape[0])
    print("Best route:", best_route)
    print("Best route length:", -result.best_fitness)
    print("Number of route tries:", route_tries)
    print("Number of unique route evaluations:", len(fitness_inputs["cache"]))
    print("Time used (seconds):", elapsed)
