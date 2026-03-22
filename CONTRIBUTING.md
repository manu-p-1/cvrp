# Contributing

Hey! Thanks for your interest; I am trying to restart as much of this as I can based on some old files I had start on. Most of the code was written for a term project and we need help to add new strategies and techniques that can help universtiy students and researchrs. TYIA!

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
| `ocvrp/cvrp.py` | CVRP evolutionary engine and visualization |
| `ocvrp/algorithms.py` | Crossover and mutation operators |
| `ocvrp/util.py` | Data classes (`Building`, `Individual`) and `.ocvrp` parser |
| `data/` | Problem set files in `.ocvrp` format |

## Adding a New Algorithm

### Crossover
1. Add the function to `ocvrp/algorithms.py` with signature:
   ```python
   def my_xo(ind1: Individual, ind2: Individual, cvrp=None) -> Tuple[Individual, Individual]:
   ```
2. Return two new `Individual` objects with `fitness=None`
3. Add a CLI flag in `driver.py` under the `cx_types` mutually exclusive group
4. Add the `elif` branch in `driver.py` to map the flag to the function

### Mutation
1. Add the function to `ocvrp/algorithms.py` with signature:
   ```python
   def my_mut(child: Individual, cvrp=None) -> Individual:
   ```
2. **Always return a new `Individual`** — never mutate the input in-place
3. Add a CLI flag in `driver.py` under the `mt_types` mutually exclusive group
4. Add the `elif` branch in `driver.py` to map the flag to the function

## Adding a Problem Set

Create a `.ocvrp` file in `data/` following the format documented in the README. The first node must be the depot (demand = 0).

## Testing

PowerShell 7 test suites live in the `testing/` directory:

- **Sequential** — runs all crossovers, mutations, datasets, and CLI flags one at a time:
  ```powershell
  pwsh testing/CVRP_Test.ps1
  ```
- **Parallel** — runs a crossover × mutation × dataset matrix via `Start-ThreadJob`:
  ```powershell
  pwsh testing/CVRP_TestParallel.ps1
  ```

Both scripts validate exit codes and check for `best_individual_fitness` in the JSON output.

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
