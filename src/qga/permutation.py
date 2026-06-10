from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .result import QGAResult


class PermutationQGA:
    """
    Quantum-inspired optimizer for ordering and routing problems.

    Solutions are 0-based permutations of ``0..n_items-1``. Unlike a generic
    multi-value encoding, this observer samples without replacement, so each
    item appears exactly once.
    """

    def __init__(
        self,
        n_items: int, #scan lines
        population_size: int, # of candidate solutions per generation
        generations: int, # of iterations to run
        fitness_func: Callable[[np.ndarray, Any], float],# function to evaluate solution quality, takes a permutation and optional inputs
        fitness_inputs: Any = None,# additional data passed to fitness_func, e.g. distance matrix for TSP
        theta_start: float = np.pi * 0.05,# initial value for the angle parameter
        theta_end: float = np.pi * 0.025,
        mutation_rate_start: float | None = None,
        mutation_rate_end: float | None = None,
        seed: int | None = None,
        maximize: bool = True,# maximize the fitness function
        verbose: bool = False,# print progress every 50 generations
    ) -> None:
        if n_items <= 1:
            raise ValueError("n_items must be greater than 1")
        if population_size <= 0:
            raise ValueError("population_size must be positive")
        if generations <= 0:
            raise ValueError("generations must be positive")

        self.n_items = n_items
        self.population_size = population_size
        self.generations = generations
        self.fitness_func = fitness_func
        self.fitness_inputs = fitness_inputs
        self.theta_start = theta_start
        self.theta_end = theta_end
        self.mutation_rate_start = (
            mutation_rate_start if mutation_rate_start is not None else 1 / (n_items + 1)
        )
        self.mutation_rate_end = (
            mutation_rate_end if mutation_rate_end is not None else 2 / (n_items + 1)
        )
        self.maximize = maximize
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

        self.probs = np.full(
            (self.population_size, self.n_items, self.n_items),
            1.0 / self.n_items,
            dtype=float,
        )# shape: (population_size, n_items, n_items) - for one candidate path, the first scanning position has equal probability for all items, the second position also has equal probability for all remaining items, etc.
         #self.probs[0, 2, 5] = 0.4 第 0 个概率个体认为，第 3 个扫描位置有 40% 概率选择 item 5。
        self.best_solution: np.ndarray | None = None
        self.best_fitness = -np.inf if maximize else np.inf

    def _schedule(self, start: float, end: float, generation: int) -> float:
        if self.generations <= 1:
            return end
        ratio = generation / (self.generations - 1)
        return start + ratio * (end - start)

    def _is_better(self, candidate: float, incumbent: float) -> bool:
        return candidate > incumbent if self.maximize else candidate < incumbent

    def observe(self) -> np.ndarray:
        population = np.empty((self.population_size, self.n_items), dtype=int)

        for p in range(self.population_size):
            remaining = list(range(self.n_items))
            for position in range(self.n_items):
                weights = self.probs[p, position, remaining]
                total = weights.sum()
                if total <= 0:
                    weights = np.full(len(remaining), 1.0 / len(remaining))
                else:
                    weights = weights / total

                chosen_index = int(self.rng.choice(len(remaining), p=weights))
                population[p, position] = remaining.pop(chosen_index)

        return population

    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        return np.array(
            [self.fitness_func(solution.copy(), self.fitness_inputs) for solution in population],
            dtype=float,
        )

    def update_probabilities(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        theta: float,
    ) -> None:
        if self.best_solution is None:
            best_idx = np.argmax(fitness) if self.maximize else np.argmin(fitness)
            target = population[best_idx]
        else:
            target = self.best_solution

        eta = min(max(theta / np.pi, 0.0), 1.0)

        for p in range(self.population_size):
            for position, item in enumerate(target):
                self.probs[p, position, :] *= 1.0 - eta
                self.probs[p, position, item] += eta
                self.probs[p, position, :] /= self.probs[p, position, :].sum()

    def mutate_probabilities(self, mutation_rate: float) -> None:
        uniform = np.full(self.n_items, 1.0 / self.n_items)

        for p in range(self.population_size):
            for position in range(self.n_items):
                if self.rng.random() <= mutation_rate:
                    self.probs[p, position, :] = 0.5 * self.probs[p, position, :] + 0.5 * uniform
                    self.probs[p, position, :] /= self.probs[p, position, :].sum()

    def run(self) -> QGAResult:
        history_best: list[float] = []
        history_mean: list[float] = []

        for gen in range(self.generations):
            theta = self._schedule(self.theta_start, self.theta_end, gen)
            mutation_rate = self._schedule(self.mutation_rate_start, self.mutation_rate_end, gen)

            population = self.observe()
            fitness = self.evaluate_population(population)

            best_idx = np.argmax(fitness) if self.maximize else np.argmin(fitness)
            gen_best_solution = population[best_idx].copy()
            gen_best_fitness = float(fitness[best_idx])

            if self.best_solution is None or self._is_better(gen_best_fitness, self.best_fitness):
                self.best_solution = gen_best_solution
                self.best_fitness = gen_best_fitness

            self.update_probabilities(population, fitness, theta)
            self.mutate_probabilities(mutation_rate)

            history_best.append(float(self.best_fitness))
            history_mean.append(float(np.mean(fitness)))

            if self.verbose and (gen % 50 == 0 or gen == self.generations - 1):
                print(
                    f"Generation {gen:5d} | "
                    f"Best = {self.best_fitness:.6f} | "
                    f"Mean = {np.mean(fitness):.6f}"
                )

        if self.best_solution is None:
            raise RuntimeError("PermutationQGA finished without evaluating any solution")

        return QGAResult(
            best_solution=self.best_solution.copy(),
            best_fitness=float(self.best_fitness),
            history_best=history_best,
            history_mean=history_mean,
        )
