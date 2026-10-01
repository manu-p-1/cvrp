# Contributing

Hey! Thanks for your interest; I am trying to restart as much of this as I can based on some old files I had started on. Most of the code was written for a term project and we need help to add new strategies and techniques that can help university students and researchers. TYIA!

## Getting Started

1. Fork the repository and clone your fork
2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install "matplotlib>=3.8,<4"
   ```
3. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Code Style

- Follow PEP 8 conventions
- Use type hints for function signatures
- Keep docstrings concise and describe *what*, not *why it exists*...but humor yourself if you feel you need to strongly explain why

## Project Structure

| File | Purpose |
|---|---|
| `driver.py` | CLI entry point and argument parsing |
| `ocvrp/cvrp.py` | `CVRP` class: evolutionary loop, fitness, and plotting |
| `ocvrp/algorithms.py` | Crossover and mutation operators, plus the operator registries |
| `ocvrp/local_search.py` | Inter-route operators (relocate, exchange, 2-opt*) |
| `ocvrp/constructive.py` | Nearest neighbor, Clarke-Wright savings, and sweep heuristics |
| `ocvrp/distance.py` | Precomputed distance matrix |
| `ocvrp/parallel.py` | Parallel multi-start runs (`IslandModel`) |
| `ocvrp/util.py` | Data classes (`Building`, `Individual`) and `.ocvrp` parser |
| `data/` | Problem set files in `.ocvrp` format |
| `testing/` | Python and PowerShell test scripts |

## Adding a New Algorithm

Keep operators as plain module-level functions (no lambdas or closures). `--islands` sends them to worker processes, which only works for functions that can be pickled.

### Crossover
1. Add the function to `ocvrp/algorithms.py` with signature:
   ```python
   def my_xo(ind1: Individual, ind2: Individual, cvrp=None) -> Tuple[Individual, Individual]:
   ```
2. Return two new `Individual` objects with `fitness=None`
3. Add it to `CROSSOVER_REGISTRY` at the bottom of `ocvrp/algorithms.py`
4. Add a CLI flag in `driver.py` under the `cx_types` mutually exclusive group
5. Add the `elif` branch in `driver.py` to map the flag to the function
6. Add the flag to the crossover lists in both PowerShell test scripts

### Mutation
1. Add the function to `ocvrp/algorithms.py` (or `ocvrp/local_search.py` for inter-route operators) with signature:
   ```python
   def my_mut(child: Individual, cvrp=None) -> Individual:
   ```
2. **Always return a new `Individual`** - never mutate the input in-place
3. Add it to `MUTATION_REGISTRY` at the bottom of `ocvrp/algorithms.py`
4. Add a CLI flag in `driver.py` under the `mt_types` mutually exclusive group
5. Add the `elif` branch in `driver.py` to map the flag to the function
6. Add the flag to the mutation lists in both PowerShell test scripts

## Adding a Problem Set

Create a `.ocvrp` file in `data/` following the format documented in the README. The first node must be the depot. Its demand is ignored, so 0 is the convention.

## Testing

Run the tests from the repository root:

- **Parser** - `.ocvrp` edge cases and a parse of every data file:
  ```bash
  python testing/test_parser.py
  ```
- **Features** - distance matrix, heuristics, seeding, operators, and a few CLI checks:
  ```bash
  python testing/test_v2.py
  ```
- **Sequential** - runs every crossover, every mutation, each data set except Leuven2, and the main CLI flags one at a time:
  ```powershell
  pwsh testing/CVRP_Test.ps1
  ```
- **Parallel** - runs every crossover/mutation pair on the test set, plus two combinations on each of three data sets, via `Start-ThreadJob`:
  ```powershell
  pwsh testing/CVRP_TestParallel.ps1
  ```

The PowerShell scripts check exit codes and look for `best_individual_fitness` in the JSON output. Pass `-Python` to use a specific interpreter. Every script exits with the number of failed tests.

For a quick one-off smoke test:
```bash
python driver.py data/test_set.ocvrp -g 500 -p 50 -A
```

## Pull Requests

- One feature or fix per PR
- Test the code - include screenshots of passing tests if possible
- Include a brief description of what changed and why
- Verify your changes run without errors on at least one problem set
- Update the CHANGELOG.md with a new entry at the top
