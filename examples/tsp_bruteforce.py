import itertools
import time

import numpy as np


def tsp_route_length(route, distance):
    length = 0.0
    for i in range(1, len(route)):
        length += distance[route[i - 1], route[i]]
    length += distance[route[-1], route[0]]
    return length


def brute_force_tsp(distance):
    n_items = distance.shape[0]
    best_route = None
    best_length = float("inf")
    route_tries = 0

    # Fix city 0 as the start to avoid counting the same cycle from every city.
    # Also keep only one direction: route and reversed route have the same length
    # for this symmetric Euclidean TSP.
    for remaining_route in itertools.permutations(range(1, n_items)):
        if remaining_route[0] > remaining_route[-1]:
            continue

        route = (0,) + remaining_route
        route_tries += 1

        route_length = tsp_route_length(route, distance)
        if route_length < best_length:
            best_route = route
            best_length = route_length

    if best_route is None:
        raise RuntimeError("Brute force search finished without evaluating any route")

    return np.array(best_route), best_length, route_tries


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

    start = time.perf_counter()
    best_route, best_length, route_tries = brute_force_tsp(distance)
    elapsed = time.perf_counter() - start

    print("Number of cities:", distance.shape[0])
    print("Best route:", best_route)
    print("Best route length:", best_length)
    print("Number of route tries:", route_tries)
    print("Time used (seconds):", elapsed)
