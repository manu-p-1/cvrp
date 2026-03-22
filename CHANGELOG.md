# Changelog

## 6/1/2025 – 3/21/2026

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
- `OCVRPParser.parse()` used `split(":")` — values containing colons (e.g. in COMMENTS) were truncated; now `split(":", 1)`
- `OCVRPParser._grab_buildings` crashed on blank lines within the node section
- `OCVRPParser` numeric headers parsed via `int()` directly; now `int(float())` to handle edge cases
- `OCVRPParser` `SyntaxError` from `_grab_buildings` was swallowed and re-wrapped by generic handler
- `OCVRPParser` added required-header validation and DIM-vs-node-count check after parsing
- `OCVRPParser` added per-row column-count validation in node parsing
- Removed unused `_headers` tuple from `OCVRPParser`

### Convergence Fixes
- Global-worst replacement → restricted tournament replacement (k=7)
- Diversity measurement via node-ID tuples instead of fitness-based `set()`
- Forced mutation when crossover doesn't fire (no more pure clones)
- Diversity maintenance: every 2k gens / 10% threshold / 30% replacement (was 10k / 1% / 75%)
- Adaptive mutation checks every 250 gens (was 1k)
- ILS-style diversity restoration: perturb with displacement/scramble/inversion, then re-optimize

### New Algorithms
- `pmx_xo` — Partially Mapped Crossover (Goldberg & Lingle 1985), `-X`
- `scramble_mut` — random subsequence shuffle, `-K`
- `two_opt_mut` — 2-opt local search per route (Croes 1958), `-T`
- `or_opt_mut` — 1–3 segment relocation (Or 1976), `-F`
- `displacement_mut` — unrestricted segment relocation (Michalewicz 1996), `-D`

### Visualization
- 4-panel figure: convergence curve, diversity plot, route map, fitness histogram
- Figures closed after save to avoid memory leaks

### Operator Complexity
- `inversion_mut` index bias fixed
- PMX, cycle, and order crossover all reduced from O(n²) to O(n)

### Evolutionary Loop
- Elitism: best-ever individual tracked and recovered if lost
- Adaptive mutation rate based on population diversity
- Cached `top_pool` in diversity maintenance
- Progress reporting for large populations (≥ 10k)

### Testing
- Rewrote PowerShell test scripts with actual algorithm coverage
- `CVRP_Test.ps1`: sequential smoke tests across crossovers, mutations, and data sets
- `CVRP_TestParallel.ps1`: parallel test matrix using `Start-ThreadJob` (PowerShell 7)
- Tests validate exit codes, fitness output, and known-optimal convergence

### Other
- Standardized crossover/mutation signatures
- Python requirement: 3.7 → 3.12
- Matplotlib pinned to `>=3.8,<4`
- Default population size: 600 → 800
- Plot label corrections (genotypes vs fitness values, 10% vs 1% threshold)
