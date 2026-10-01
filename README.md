# Optimizing the Capacitated Vehicle Routing Problem using Genetic Algorithms

## Info

### Authors
- Manu Puduvalli
- Samuel Yuen
- Lok Kwong
- Vrund Parikh
- Glen George

### Objective
This project was created to fulfill the term-project requirement for Dr. Khaled Rasheed's CSCI 4560 Evolutionary
Computing Applications class at the University of Georgia. It has been extended since then with more operators,
constructive heuristics, and a parallel mode; see the [changelog](CHANGELOG.md) for details.

## Introduction

### Problem
The Capacitated Vehicle Routing Problem (CVRP) is an NP-Hard combinatorial optimization problem. Similar to the
travelling salesperson problem, several nodes are placed on a 2D Euclidean space which represent customers. Each node
has a numerical value associated with it representing a pickup demand of packages or goods. A depot sends out
vehicles, and all vehicles have the same capacity. Each customer must be visited by exactly one vehicle, and no vehicle
can pick up more than its capacity. The goal is to find a set of routes, each starting and ending at the depot, with
the smallest total distance.

### Research
We solve this problem using evolutionary approaches, specifically genetic algorithms, and compare the results against
the best known solution for each problem set.

### Abstract
This project was submitted in conjunction with a conference-like research paper.

*This paper studies the effect of optimized heuristic crossover operations by utilizing a Genetic Algorithm to optimize 
the Capacitated Vehicle Routing Problem (CVRP) and understand the effect on optimized crossovers on diversity 
maintenance. Best-Route Crossover showed promising results within 5% of the optimal solution on various well-known 
CVRP data sets. An optimized type of Cycle Crossover exhibited a solution to maintain population diversity and prevent 
premature convergence.*

### Index Terms
*Best-Route Crossover, Capacitated Vehicle Routing Problem, Cycle Crossover, Genetic Algorithm, Genetic Vehicle 
Representation, Optimization, Travelling Salesperson Problem, Vehicle Routing Problem*

## Algorithms

Where an operator or technique comes from a specific paper, the paper is listed so you can compare the implementation
against the original.

### Constructive Heuristics

These build a complete, capacity-feasible solution from scratch. With `--seed-pct`, part of the starting population
comes from them, and they can also be called on their own (see below).

| Heuristic | Description | Reference |
|---|---|---|
| **Nearest Neighbor** | Starts each route at the depot and keeps moving to the closest unvisited customer that still fits | Laporte, G. (1992). "The vehicle routing problem: An overview of exact and approximate algorithms." *European Journal of Operational Research*, 59(3), 345-358. Section 3 notes that the TSP nearest neighbour method carries over to the CVRP. |
| **Clarke-Wright Savings** | Computes pairwise savings *s(i,j) = d(0,i) + d(0,j) - d(i,j)* and merges routes in descending savings order | Clarke, G. and Wright, J.W. (1964). "Scheduling of vehicles from a central depot to a number of delivery points." *Operations Research*, 12(4), 568-581. |
| **Sweep** | Sorts customers by polar angle around the depot, starting from a random angle, and fills routes in that order | Gillett, B.E. and Miller, L.R. (1974). "A heuristic algorithm for the vehicle-dispatch problem." *Operations Research*, 22(2), 340-349. |

### Crossover Operators

| Operator | Flag | Reference |
|---|---|---|
| **Best Route Crossover** | `-B` | The geographic GVR crossover: a sub-route from one parent is inserted after the closest customer in the other. Pereira, F.B., Tavares, J., Machado, P., Costa, E. (2002). "GVR: A new genetic representation for the vehicle routing problem." In *Artificial Intelligence and Cognitive Science (AICS 2002)*, LNCS 2464, Springer, pp. 95-102. Compared against a generic variant in Tavares, J., Pereira, F.B., Machado, P., Costa, E. (2003). "Crossover and diversity: A study about GVR." In *GECCO 2003: Proceedings of the Bird of a Feather Workshops* (ADoRo workshop), pp. 27-33. |
| **Cycle Crossover** (optimized) | `-C` | Nazif, H. and Lee, L.S. (2012). "Optimised crossover genetic algorithm for capacitated vehicle routing problem." *Applied Mathematical Modelling*, 36(5), 2110-2117. |
| **Edge Recombination** | `-E` | Whitley, D., Starkweather, T., Fuquay, D. (1989). "Scheduling problems and traveling salesmen: The genetic edge recombination operator." *Proc. 3rd International Conference on Genetic Algorithms (ICGA)*, pp. 133-140. |
| **Order Crossover (OX)** | `-O` | Davis, L. (1985). "Applying adaptive algorithms to epistatic domains." *Proc. 9th International Joint Conference on Artificial Intelligence (IJCAI)*, Vol. 1, pp. 162-164. |
| **Partially Mapped (PMX)** | `-X` | Goldberg, D.E. and Lingle, R., Jr. (1985). "Alleles, loci, and the traveling salesman problem." *Proc. 1st International Conference on Genetic Algorithms (ICGA)*, pp. 154-159. |

### Mutation / Local Search Operators

An individual is a single permutation of customers. It is split into routes by walking the permutation and starting
a new route whenever the next customer would go over capacity. *Whole chromosome* operators work on that
permutation, so they can move customers across route boundaries. *Intra-route* operators work on one route at a time,
and *inter-route* operators move customers from one route to another.

| Operator | Flag | Type | Reference |
|---|---|---|---|
| **Inversion** | `-I` | Whole chromosome | Holland, J.H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press. |
| **Swap** | `-W` | Whole chromosome | Banzhaf, W. (1990). "The 'molecular' traveling salesman." *Biological Cybernetics*, 64(1), 7-14. |
| **GVR Scramble** | `-G` | Intra-route | Scramble (Syswerda 1991, below) applied to the longest route only. This variant is the project's own. |
| **Scramble** | `-K` | Whole chromosome | Syswerda, G. (1991). "Schedule optimization using genetic algorithms." In L. Davis (Ed.), *Handbook of Genetic Algorithms*, Van Nostrand Reinhold, pp. 332-349. |
| **2-opt** | `-T` | Intra-route | Croes, G.A. (1958). "A method for solving traveling-salesman problems." *Operations Research*, 6(6), 791-812. |
| **Or-opt** | `-F` | Whole chromosome | Or, I. (1976). *Traveling salesman-type combinatorial problems and their relation to the logistics of regional blood banking.* PhD thesis, Northwestern University. |
| **Displacement** | `-D` | Whole chromosome | Michalewicz, Z. (1992). *Genetic Algorithms + Data Structures = Evolution Programs*. Springer-Verlag. |
| **Inter-route Relocate** | `-L` | Inter-route | Savelsbergh, M.W.P. (1992). "The vehicle routing problem with time windows: Minimizing route duration." *ORSA Journal on Computing*, 4(2), 146-154. |
| **Inter-route Exchange** | `-Z` | Inter-route | Savelsbergh (1992), above. |
| **2-opt\*** | `-Y` | Inter-route | Potvin, J.-Y. and Rousseau, J.-M. (1995). "An exchange heuristic for routeing problems with time windows." *Journal of the Operational Research Society*, 46(12), 1433-1446. |

The attributions for inversion, swap, scramble, and displacement follow Larrañaga, P., Kuijpers, C.M.H., Murga, R.H.,
Inza, I., Dizdarevic, S. (1999). "Genetic algorithms for the travelling salesman problem: A review of representations
and operators." *Artificial Intelligence Review*, 13(2), 129-170. The inter-route ones follow Bräysy, O. and
Gendreau, M. (2005). "Vehicle routing problem with time windows, Part I: Route construction and local search
algorithms." *Transportation Science*, 39(1), 104-118.

### Parallel Runs

`--islands N` (or `CVRP.run_parallel`) runs N copies of the GA in separate processes. Each copy gets its own random
seed and an equal share of the population (at least 50 individuals), and the best result is returned. The islands
never exchange individuals, so this is a multi-start GA spread across CPU cores rather than an island model with
migration. Background reading:

| Topic | Reference |
|---|---|
| Island models | Alba, E. and Tomassini, M. (2002). "Parallelism and evolutionary algorithms." *IEEE Trans. on Evolutionary Computation*, 6(5), 443-462. |
| Parallel GAs survey | Cantú-Paz, E. (1998). "A survey of parallel genetic algorithms." *Calculateurs Parallèles, Réseaux et Systèmes Répartis*, 10(2), 141-171. |

### Evolutionary Loop

The GA is steady-state: each generation picks two parents, produces two children, and puts them back into the
population. The parents are the two fittest individuals in a random sample of `--sel` individuals. Each child then
replaces the worst of 7 randomly sampled individuals, as long as the child is at least as fit, which keeps more
diversity than always replacing the global worst. The sampling idea comes from restricted tournament selection
(Harik, G.R. (1995). "Finding multimodal solutions using restricted tournament selection." *Proc. 6th International
Conference on Genetic Algorithms (ICGA)*, pp. 24-31), but this version compares fitness rather than genotype
similarity. If crossover doesn't fire, the copied parents still go through the mutation operator.

Two things push back against premature convergence. Every 250 generations, the mutation probability is doubled (up
to 0.5) if fewer than 15% of the genotypes are unique, or quadrupled (up to 1.0) below 5%, and it goes back to the
configured value once diversity recovers. Every 2,000 generations, if 10% or fewer of the genotypes are unique, the
worst 30% of the population is replaced with perturbed copies of the top 10%. Each copy gets 2-3 random displacement,
scramble, or inversion moves, then one pass of the configured mutation.

The best individual ever seen is kept on the side and returned if replacement lost it. All pairwise distances are
computed once when the problem set loads and rounded to the nearest integer, which is the convention the benchmark
optimal values use.

## Problem Sets
Problem sets are organized in a custom format called `.ocvrp` for easier data processing. It's a simple format that is 
easy to use. It contains 5 required headers and 1 optional *COMMENTS* header:

1. Name of the problem set - (str)
2. Comments for the problem set - (str) (optional)
3. The dimension of the problem set (including the depot) - (int)
4. The maximum capacity each vehicle is able to hold (int) 
5. The optimal value for the data set (int)
6. Nodes for data set

Node data is organized in tabular format with 4 elements per row. These elements are:

1. Node number - (int)
2. Node x-coordinate - (int or float)
3. Node y-coordinate - (int or float)
4. Node service demand - (int)

Header values follow the format:  

`HEADER: value`   

And for Node headers:
```
HEADER:
value
value
...
```

Ordering of headers, spacing in between a header and value, or spacing in between sets of headers are irrelevant.
The numeric values of nodes must be positioned below the NODES header value. Headers are case-insensitive although convention
is to use all-CAPS.
The first node **must be the depot location** and comments or any other unrecognizable characters are forbidden. 
An example of the format is shown below.

```
NAME: A-n54-k7
COMMENTS: Augerat 1995 Set A
DIM: 54
CAPACITY: 100
OPTIMAL: 1167
NODES:
1 61 5 0
2 85 53 24
3 17 57 9
4 49 93 15
5 69 11 17

...
```

## Execution

### Requirements

- Python 3.12 or newer
- `pip` or `pipenv`

### Setup

There are two options to run the algorithm:

1. With pip (using a virtual environment)
    ```
    python3 -m venv .venv
    source .venv/bin/activate
    pip install "matplotlib>=3.8,<4"
    python driver.py data/A-n54-k7.ocvrp -g 1000 -A
    ```
2.  With pipenv
    ```
    pip install pipenv
    pipenv install
    pipenv run python driver.py data/A-n54-k7.ocvrp -g 1000 -A
    ```

matplotlib is the only third-party dependency, and it is only imported when plotting (`-M`).

The problem set path is the only required argument. Any option you leave out uses its default:
- Population Size: `800`
- Selection Size: `5`
- Number of Generations: `100000`
- Mutation Probability: `0.15`
- Crossover Probability: `0.85`
- Crossover: `best_route_xo`
- Mutation: `inversion_mut`

### Command Line

To run the program on your command line, view the arguments by running `python driver.py -h` or
`python driver.py --help`

```
usage: driver.py [-h] [-o ] [-p ] [-s ] [-g ] [-m ] [-c ] [-r ] [-B | -C | -E | -O | -X] [-I |
                 -W | -G | -K | -T | -F | -D | -L | -Z | -Y] [-i []] [-P] [-A] [-S] [-R] [-M]
                 [--islands ] [--seed-pct ]
                 file

Runs the CVRP with any of the optional arguments

positional arguments:
  file             the path to the problem set

options:
  -h, --help       show this help message and exit
  -o, --output     the path to output the results (creates the path if it does not exist)
  -p, --pop        the population size
  -s, --sel        the selection size
  -g, --ngen       the number of generations
  -m, --mutpb      the mutation probability
  -c, --cxpb       the crossover probability
  -r, --run        the number of times to run the problem
  -B, --brxo       use best route crossover
  -C, --cxo        use cycle crossover
  -E, --erxo       use edge recombination crossover
  -O, --oxo        use order crossover
  -X, --pmxo       use partially mapped crossover (PMX)
  -I, --vmt        use inversion mutation
  -W, --swmt       use swap mutation
  -G, --gvmt       use GVR based scramble mutation
  -K, --scmt       use scramble mutation
  -T, --topt       use 2-opt local search mutation
  -F, --oropt      use or-opt mutation
  -D, --disp       use displacement mutation
  -L, --reloc      use inter-route relocate mutation (local search)
  -Z, --exch       use inter-route exchange mutation (local search)
  -Y, --toptstar   use 2-opt* inter-route tail exchange mutation
  -i, --indent []  the indentation amount of the result string
  -P, --pgen       prints the current generation
  -A, --agen       prints population statistics every 250 generations
  -S, --save       saves the results to a file
  -R, --routes     adds every route (verbose) of the best individual to the result
  -M, --plot       saves a summary plot of each run (needs matplotlib)
  --islands        number of independent GA runs to execute in parallel (the population is split
                   between them)
  --seed-pct       fraction of population seeded with constructive heuristics (0.0-1.0)
```

The population size must be at least 5 and the selection size at least 2. With `--islands`, the `-P` and `-A`
progress output is turned off inside each island, and `-M` can't be used.

#### Saving Results

Saved files go to a `results` directory by default, or to the path given with `-o`. The directory is created if it
does not exist.

With `-S`, the results of all runs are saved to one JSON file named:

CROSSOVER ALGORITHM\_GENERATIONS\_CROSSOVER PROBABILITY\_DATA SET\_\_YYYYMMDD\_\_HH\_MM\_SSAM/PM

For example:

`best_route_xo_100000_0.85_F-n45-k4__20201213__06_03_29PM.json`

With `-M`, a JPEG plot of each run is saved to the same directory, named:

CROSSOVER ALGORITHM\_GENERATIONS\_CROSSOVER PROBABILITY\_DATA SET\_\_YYYYMMDD\_\_HH\_MM\_SSAM/PM\_\_RUN NUMBER\_\_FITNESS VALUE

For example:

`best_route_xo_100000_0.85_F-n45-k4__20201213__06_03_29PM__RUN1__FIT732.jpg`

The plot shows the convergence curve, population diversity, the routes of the best solution, and the fitness
distribution of the final population:

<img width="1988" height="1919" alt="Example plot with convergence, diversity, route map, and fitness histogram panels" src="https://github.com/user-attachments/assets/6f5dfa81-2f9a-4475-8362-1a96c0ea9100" />

### Using Package

For non-terminal based runs and integration, a CVRP object can be created and run by calling the `run()` function.
```python
import json
from ocvrp import algorithms as algo
from ocvrp.cvrp import CVRP
from ocvrp.util import CVRPEncoder

# The path to the .ocvrp file is the problem set for this instance
cvrp = CVRP("./data/A-n54-k7.ocvrp", cxpb=0.75, ngen=50_000, pgen=True,
            plot_save_path="A-n54-k7-Run1.png", cx_algo=algo.edge_recomb_xo)

# Result contains a dict of information about the run which includes the best individual found
result = cvrp.run()

js_res = json.dumps(obj=result, cls=CVRPEncoder, indent=2)
print(js_res)

cvrp.reset()
```

`CVRP.run()` returns a dictionary summarizing the run. Here is one from a default run on F-n45-k4:

```
{
'name': 'CVRP',
'problem_set_name': 'F-n45-k4',
'problem_set_optimal': 724,
'time': '522.453125 seconds',
'vehicles': 4,
'vehicle_capacity': 2010,
'dimension': 44,
'population_size': 800,
'selection_size': 5,
'generations': 100000,
'cxpb': 0.85,
'mutpb': 0.15,
'cx_algorithm': 'best_route_xo',
'mut_algorithm': 'inversion_mut',
'best_individual_fitness': 729
}
```
If `plot_save_path` is set, the dictionary also has a `plot_save_path` entry. If `verbose_routes` is set to `True`,
the routes of the best individual are added under `best_individual`. To convert the dictionary to JSON, pass the
`CVRPEncoder` class to `json.dumps`.

#### Parallel Runs

Run several independent copies of the GA across CPU cores and keep the best result. Worker processes re-import the
main module on macOS and Windows, so scripts need an `if __name__ == "__main__":` guard:
```python
from ocvrp.cvrp import CVRP

if __name__ == "__main__":
    cvrp = CVRP("./data/A-n54-k7.ocvrp", ngen=25_000, seed_pct=0.1)
    result = cvrp.run_parallel(n_islands=4)
    print(f"Best fitness: {result['best_individual_fitness']}")
```

The result has the same keys as `run()`, plus `n_islands` and `island_pop`. You can also use `IslandModel` directly,
which passes any extra keyword arguments on to each `CVRP`:
```python
from ocvrp.parallel import IslandModel
from ocvrp import algorithms as algo

if __name__ == "__main__":
    model = IslandModel(
        "./data/A-n54-k7.ocvrp",
        n_islands=8,
        total_pop=1600,
        ngen=50_000,
        cx_algo=algo.cycle_xo,
        mutpb=0.20,
        seed_pct=0.1,
    )
    result = model.run()
```

#### Constructive Heuristics (Standalone)

The constructive heuristics can also be used independently:
```python
from ocvrp.util import OCVRPParser
from ocvrp.distance import DistanceMatrix
from ocvrp.constructive import clarke_wright_savings, nearest_neighbor, sweep_heuristic

ps = OCVRPParser("./data/A-n54-k7.ocvrp").parse()
depot = ps.get_ps_depot()
buildings = ps.get_ps_buildings()
capacity = ps.get_ps_capacity()
dm = DistanceMatrix(depot, buildings)

savings_routes = clarke_wright_savings(depot, buildings, capacity, dm)
nn_routes = nearest_neighbor(depot, buildings, capacity, dm)
sweep_routes = sweep_heuristic(depot, buildings, capacity, dm)

for i, route in enumerate(savings_routes, 1):
    print(f"Route {i}: {[b.node for b in route]}")
```

### Testing

The tests live in `testing/` and expect to be run from the repository root.

| Script | What it covers |
|---|---|
| `test_parser.py` | `.ocvrp` parser edge cases, plus a parse of every file in `data/` |
| `test_v2.py` | Distance matrix, constructive heuristics, seeding, the inter-route operators, 2-opt, and a few CLI checks |
| `CVRP_Test.ps1` | Runs the driver once per crossover, mutation, and data set (except Leuven2), plus some combinations and CLI flags |
| `CVRP_TestParallel.ps1` | Every crossover/mutation pair on the test set, plus two combinations on each of three data sets, run concurrently with `Start-ThreadJob` |

```bash
python testing/test_parser.py
python testing/test_v2.py
pwsh testing/CVRP_Test.ps1
pwsh testing/CVRP_TestParallel.ps1 -MaxParallel 8
```

The PowerShell scripts need PowerShell 7. They check each run's exit code and look for `best_individual_fitness` in
the JSON output. Both accept `-Python` to pick the interpreter (for example `-Python .venv/bin/python`), and the
parallel script runs 4 jobs at a time unless you pass `-MaxParallel`. Every script exits with the number of failed
tests.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for the full list of changes.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.
