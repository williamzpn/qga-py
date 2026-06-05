from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QGAResult:
    best_solution: np.ndarray
    best_fitness: float
    history_best: list[float]
    history_mean: list[float]
