# qga-py

Quantum-inspired genetic and evolutionary algorithms for combinatorial
optimization on classical computers.

This project does not require quantum hardware. It implements practical
quantum-inspired optimizers that maintain probability amplitudes or
probability-like distributions, sample candidate solutions, evaluate them, and
update the distributions toward better solutions.

## Installation

```bash
pip install qga-py
```

## Usage

Binary optimization:

```python
from qga import BinaryQGA

def fitness(bits, _):
    return bits.sum()

optimizer = BinaryQGA(
    genome_length=100,
    population_size=20,
    generations=200,
    fitness_func=fitness,
)

result = optimizer.run()
print(result.best_solution, result.best_fitness)
```

Permutation optimization:

```python
from qga import PermutationQGA

def fitness(route, distance):
    length = sum(distance[route[i - 1], route[i]] for i in range(len(route)))
    return -length

optimizer = PermutationQGA(
    n_items=20,
    population_size=40,
    generations=300,
    fitness_func=fitness,
    fitness_inputs=distance,
)

result = optimizer.run()
print(result.best_solution, result.best_fitness)
```

## Included Examples

- `examples/knapsack_demo.py`
- `examples/tsp_demo.py`

## Publishing Checklist

1. Update the project URLs in `pyproject.toml`.
2. Run tests with `pytest`.
3. Build with `python -m build`.
4. Upload to TestPyPI first with `python -m twine upload --repository testpypi dist/*`.
5. Install from TestPyPI and verify imports.
6. Upload to PyPI with `python -m twine upload dist/*`.

## License

MIT
