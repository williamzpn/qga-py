from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .result import QGAResult


class QuantumGeneticAlgorithm:
    """
    Quantum-inspired genetic algorithm for binary and discrete problems.

    Candidate values follow the original R-style convention:
    each gene is represented as an integer in ``1..nvalues_sol``.
    For binary problems, ``1`` means 0/not selected and ``2`` means 1/selected.
    """

    def __init__(
        self,
        popsize: int,
        generation_max: int,
        nvalues_sol: int,
        genome: int,
        thetainit: float,
        thetaend: float,
        eval_fitness: Callable[[np.ndarray, Any], float],
        eval_func_inputs: Any = None,
        pop_mutation_rate_init: float | None = None,
        pop_mutation_rate_end: float | None = None,
        mutation_rate_init: float | None = None,
        mutation_rate_end: float | None = None,
        mutation_flag: bool = True,
        seed: int | None = None,
        maximize: bool = True,
        verbose: bool = False,
    ) -> None:
        if popsize <= 0:
            raise ValueError("popsize must be positive")
        if generation_max <= 0:
            raise ValueError("generation_max must be positive")
        if nvalues_sol < 2:
            raise ValueError("nvalues_sol must be at least 2")
        if genome <= 0:
            raise ValueError("genome must be positive")

        self.popsize = popsize
        self.generation_max = generation_max
        self.nvalues_sol = nvalues_sol
        self.genome = genome
        self.thetainit = thetainit
        self.thetaend = thetaend
        self.eval_fitness = eval_fitness
        self.eval_func_inputs = eval_func_inputs
        self.mutation_flag = mutation_flag
        self.maximize = maximize
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

        self.pop_mutation_rate_init = (
            pop_mutation_rate_init if pop_mutation_rate_init is not None else 1 / (popsize + 1)
        )
        self.pop_mutation_rate_end = (
            pop_mutation_rate_end if pop_mutation_rate_end is not None else 1 / (popsize + 1)
        )
        self.mutation_rate_init = (
            mutation_rate_init if mutation_rate_init is not None else 1 / (genome + 1)
        )
        self.mutation_rate_end = (
            mutation_rate_end if mutation_rate_end is not None else 1 / (genome + 1)
        )

        self.probs = np.full(
            (self.popsize, self.genome, self.nvalues_sol),
            1.0 / self.nvalues_sol,
            dtype=float,
        )

        self.best_solution: np.ndarray | None = None
        self.best_fitness = -np.inf if maximize else np.inf

    def _schedule(self, start: float, end: float, generation: int) -> float:
        if self.generation_max <= 1:
            return end
        ratio = generation / (self.generation_max - 1)
        return start + ratio * (end - start)

    def _is_better(self, candidate: float, incumbent: float) -> bool:
        return candidate > incumbent if self.maximize else candidate < incumbent

    def observe(self) -> np.ndarray:
        solutions = np.empty((self.popsize, self.genome), dtype=int)
        values = np.arange(1, self.nvalues_sol + 1)

        for p in range(self.popsize):
            for g in range(self.genome):
                solutions[p, g] = self.rng.choice(values, p=self.probs[p, g])

        return solutions

    def evaluate_population(self, solutions: np.ndarray) -> np.ndarray:
        return np.array(
            [self.eval_fitness(solution.copy(), self.eval_func_inputs) for solution in solutions],
            dtype=float,
        )

    def update_probabilities(
        self,
        solutions: np.ndarray,
        fitness: np.ndarray,
        theta: float,
    ) -> None:
        if self.best_solution is None:
            best_idx = np.argmax(fitness) if self.maximize else np.argmin(fitness)
            target = solutions[best_idx]
        else:
            target = self.best_solution

        eta = min(max(theta / np.pi, 0.0), 1.0)

        for p in range(self.popsize):
            for g in range(self.genome):
                target_value = target[g] - 1
                self.probs[p, g, :] *= 1.0 - eta
                self.probs[p, g, target_value] += eta
                self.probs[p, g, :] /= self.probs[p, g, :].sum()

    def mutate_probabilities(self, pop_mut_rate: float, mut_rate: float) -> None:
        if not self.mutation_flag:
            return

        for p in range(self.popsize):
            if self.rng.random() > pop_mut_rate:
                continue

            for g in range(self.genome):
                if self.rng.random() <= mut_rate:
                    if self.nvalues_sol == 2:
                        self.probs[p, g, :] = self.probs[p, g, ::-1]
                    else:
                        uniform = np.full(self.nvalues_sol, 1.0 / self.nvalues_sol)
                        self.probs[p, g, :] = 0.5 * self.probs[p, g, :] + 0.5 * uniform
                        self.probs[p, g, :] /= self.probs[p, g, :].sum()

    def run(self) -> QGAResult:
        history_best: list[float] = []
        history_mean: list[float] = []

        for gen in range(self.generation_max):
            theta = self._schedule(self.thetainit, self.thetaend, gen)
            pop_mut_rate = self._schedule(
                self.pop_mutation_rate_init,
                self.pop_mutation_rate_end,
                gen,
            )
            mut_rate = self._schedule(
                self.mutation_rate_init,
                self.mutation_rate_end,
                gen,
            )

            solutions = self.observe()
            fitness = self.evaluate_population(solutions)

            best_idx = np.argmax(fitness) if self.maximize else np.argmin(fitness)
            gen_best_solution = solutions[best_idx].copy()
            gen_best_fitness = float(fitness[best_idx])

            if self.best_solution is None or self._is_better(gen_best_fitness, self.best_fitness):
                self.best_solution = gen_best_solution
                self.best_fitness = gen_best_fitness

            self.update_probabilities(solutions, fitness, theta)
            self.mutate_probabilities(pop_mut_rate, mut_rate)

            history_best.append(float(self.best_fitness))
            history_mean.append(float(np.mean(fitness)))

            if self.verbose and (gen % 50 == 0 or gen == self.generation_max - 1):
                print(
                    f"Generation {gen:5d} | "
                    f"Best = {self.best_fitness:.6f} | "
                    f"Mean = {np.mean(fitness):.6f}"
                )

        if self.best_solution is None:
            raise RuntimeError("QGA finished without evaluating any solution")

        return QGAResult(
            best_solution=self.best_solution.copy(),
            best_fitness=float(self.best_fitness),
            history_best=history_best,
            history_mean=history_mean,
        )


class BinaryQGA(QuantumGeneticAlgorithm):
    """Convenience wrapper for binary problems returning 0/1 solutions."""

    def __init__(
        self,
        genome_length: int,
        population_size: int,
        generations: int,
        fitness_func: Callable[[np.ndarray, Any], float],
        fitness_inputs: Any = None,
        theta_start: float = np.pi * 0.05,
        theta_end: float = np.pi * 0.025,
        seed: int | None = None,
        maximize: bool = True,
        verbose: bool = False,
        mutation: bool = True,
    ) -> None:
        def wrapped_fitness(solution: np.ndarray, inputs: Any) -> float:
            bits = solution - 1
            return fitness_func(bits, inputs)

        super().__init__(
            popsize=population_size,
            generation_max=generations,
            nvalues_sol=2,
            genome=genome_length,
            thetainit=theta_start,
            thetaend=theta_end,
            eval_fitness=wrapped_fitness,
            eval_func_inputs=fitness_inputs,
            mutation_flag=mutation,
            seed=seed,
            maximize=maximize,
            verbose=verbose,
        )

    def run(self) -> QGAResult:
        result = super().run()
        return QGAResult(
            best_solution=result.best_solution - 1,
            best_fitness=result.best_fitness,
            history_best=result.history_best,
            history_mean=result.history_mean,
        )
