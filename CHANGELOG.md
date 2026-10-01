# Changelog

## 9/30/2026

### Breaking Changes
- `selection_size` must be at least 2; a selection size of 1 used to crash on the first generation
- `-p`, `-s`, and `-g` reject anything `CVRP` would reject instead of quietly falling back to the defaults on 0
- `-M` with `--islands` is a usage error instead of silently skipping the plot
- `IslandModel` passes every extra keyword argument on to `CVRP` and sends operator functions to the islands directly; operator names given as strings are no longer accepted
- `CVRPEncoder` raises `TypeError` for unsupported types instead of writing `null`
- `nearest_neighbor` raises `ValueError` when a demand exceeds the vehicle capacity (it used to loop forever), and population seeding no longer swallows exceptions from the heuristics

### Bug Fixes
- `CVRP_Test.ps1` used `$Args` as a parameter name, which collides with PowerShell's automatic `$args`; every test ran `driver.py` with no arguments and failed
- `CVRP_TestParallel.ps1` always exited 0; it now exits with the number of failed tests
- `-m 0` and `-c 0` were replaced with the defaults
- `--islands` ignored `-R`
- Custom operators failed in `run_parallel` because the islands looked operators up by name
- `two_opt_mut` never reversed a segment of two customers, so it could stop short of a 2-opt local optimum
- `OCVRPParser` failed on whitespace-only lines between headers
- `Individual.__delitem__` always raised `TypeError`
- An invalid `-o` path printed a traceback instead of a usage error
- Pipfile `start` script couldn't find `driver.py`

### Citations
- Best route crossover listed its authors in the wrong order with an incomplete venue (it is Tavares, Pereira, Machado, Costa 2003, pp. 27-33). The operator is the geographic GVR crossover from Pereira et al. (2002), which is now cited as well
- Nazif & Lee is 2012, 36(5), 2110-2117, not 2011
- The edge recombination paper's title says "traveling salesmen"
- GVR scramble no longer cites the GVR paper, which doesn't define a scramble mutation
- Displacement mutation cites Michalewicz's 1992 first edition, and inter-route exchange cites Savelsbergh (1992) instead of Kindervater & Savelsbergh (1997)
- Filled in missing venues, issues, and pages, and listed the surveys the operator attributions follow

### Other
- `MUTATION_REGISTRY` includes the inter-route operators
- PowerShell suites cover every mutation operator, plus `--seed-pct` and `--islands`
- README: current CLI help, corrected operator types and problem statement, parallel mode described as a multi-start GA, `__main__` guard in the parallel examples, and an Evolutionary Loop section in place of the Performance Optimizations table
- Version bumped to 3.0.0

## 7/22/2026

### New Modules
- `ocvrp/distance.py` - Precomputed all-pairs Euclidean distance matrix with O(1) lookup by node ID; replaces repeated `math.sqrt` calls in fitness evaluation and local search (Toth & Vigo 2002)
- `ocvrp/constructive.py` - Classical constructive heuristics for population seeding:
  - **Nearest Neighbor** greedy construction (Laporte 1992)
  - **Clarke-Wright Savings** parallel savings algorithm (Clarke & Wright 1964)
  - **Sweep Algorithm** angle-based clustering (Gillett & Miller 1974)
- `ocvrp/local_search.py` - Inter-route improvement operators (mutation-compatible):
  - **Relocate** - move one customer to the best position in another route (Savelsbergh 1992), `-L`
  - **Exchange** - swap customers between two routes (Kindervater & Savelsbergh 1997), `-Z`
  - **2-opt\*** - exchange tails between two routes (Potvin & Rousseau 1995), `-Y`
- `ocvrp/parallel.py` - Multi-start island-model parallel GA using `concurrent.futures.ProcessPoolExecutor` (Alba & Tomassini 2002; Cantú-Paz 1998)

### Performance
- `calc_fitness` looks up edge costs in the precomputed distance matrix instead of calling `math.sqrt` per edge
- `two_opt_mut` uses the distance matrix when available and falls back to `Building.distance` otherwise
- Algorithm registries (`CROSSOVER_REGISTRY`, `MUTATION_REGISTRY`) mapping operator names to functions

### Population Initialization
- New `seed_pct` parameter (0.0-1.0): seeds a fraction of the initial population with constructive heuristic solutions and their mutated variants; remaining slots filled randomly
- `--seed-pct` CLI flag

### Parallelization
- `CVRP.run_parallel(n_islands=N)` convenience method
- `IslandModel` class for running the multi-start parallel GA directly
- `--islands N` CLI flag to enable parallel mode from the command line
- Each island runs in a separate OS process with its own random seed

### Documentation
- `README.md` Algorithms section with references for the crossover, mutation, constructive, and parallel methods
- Package docstring in `__init__.py` with module overview
- Version bumped to 2.0.0

## 6/1/2025 - 3/21/2026

Local uncommitted files transferred from my Precision to my Mac, for better or for worse.

### Bug Fixes
- `edge_recomb_xo` used wrong parent for `ind2_adjacent`
- Mutations mutated in-place, corrupting population; all return new objects now
- Inconsistent crossover return types; unified to `Tuple[Individual, Individual]`
- Loop variable `i` shadowed inside diversity maintenance; renamed to `gen`
- Cloned parents when crossover skipped to avoid mutating live population members
- Double-wrapped `Individual(Individual(...))` in diversity maintenance
- Replaced string-based algorithm dispatch with function references
- Moved matplotlib import to lazy load (only when plotting)
- `_is_int_ge` error said "float" instead of "int"
- `OCVRPParser` file handle leak; refactored to context manager
- Proper exception chaining in parser
- `ValidOutputFile` never raised the error it created
- `mkdir` race condition; added `exist_ok=True`
- Default mutation mismatch between code (`swap_mut`) and docs (`inversion_mut`)
- `OCVRPParser.parse()` used `split(":")` - values containing colons (e.g. in COMMENTS) were truncated; now `split(":", 1)`
- `OCVRPParser._grab_buildings` crashed on blank lines within the node section
- `OCVRPParser` numeric headers parsed via `int()` directly; now `int(float())` to handle edge cases
- `OCVRPParser` `SyntaxError` from `_grab_buildings` was swallowed and re-wrapped by generic handler
- `OCVRPParser` added required-header validation and DIM-vs-node-count check after parsing
- `OCVRPParser` added per-row column-count validation in node parsing
- Removed unused `_headers` tuple from `OCVRPParser`

### Convergence Fixes
- Replaced global-worst replacement with worst-of-7 tournament replacement
- Diversity measurement via node-ID tuples instead of fitness-based `set()`
- Forced mutation when crossover doesn't fire (no more pure clones)
- Diversity maintenance: every 2k gens / 10% threshold / 30% replacement (was 10k / 1% / 75%)
- Adaptive mutation checks every 250 gens (was 1k)
- ILS-style diversity restoration: perturb with displacement/scramble/inversion, then re-optimize

### New Algorithms
- `pmx_xo` - Partially Mapped Crossover (Goldberg & Lingle 1985), `-X`
- `scramble_mut` - random subsequence shuffle, `-K`
- `two_opt_mut` - 2-opt local search per route (Croes 1958), `-T`
- `or_opt_mut` - relocates a segment of 1-3 genes (Or 1976), `-F`
- `displacement_mut` - unrestricted segment relocation (Michalewicz 1996), `-D`

### Visualization
- 4-panel figure: convergence curve, diversity plot, route map, fitness histogram
- Figures closed after save to avoid memory leaks

### Operator Complexity
- `inversion_mut` index bias fixed
- PMX and order crossover, and the child construction in cycle crossover, reduced from O(n^2) to O(n)

### Evolutionary Loop
- Elitism: best-ever individual tracked and recovered if lost
- Adaptive mutation rate based on population diversity
- Cached `top_pool` in diversity maintenance
- Progress reporting for large populations (10k or more)

### Testing
- Rewrote PowerShell test scripts with actual algorithm coverage
- `CVRP_Test.ps1`: sequential smoke tests across crossovers, mutations, and data sets
- `CVRP_TestParallel.ps1`: parallel test matrix using `Start-ThreadJob` (PowerShell 7)
- Tests validate exit codes and that a fitness value appears in the output

### Other
- Standardized crossover/mutation signatures
- Python requirement: 3.7 to 3.12
- Matplotlib pinned to `>=3.8,<4`
- Default population size: 600 to 800
- Plot label corrections (genotypes vs fitness values, 10% vs 1% threshold)
