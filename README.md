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
Computing Applications class at the University of Georgia.

## Introduction

### Problem
The Capacitated Vehicle Routing Problem (CVRP) is an NP-Hard combinatorial optimization problem. Similar to the
travelling salesperson problem, several nodes are placed on a 2D Euclidean space which represent customers. Each node
has a numerical value associated with it representing a pickup demand of packages or goods. At the center of a grid
space exists a depot responsible for sending out vehicles and all vehicles have a homogenous capacity. Every vehicle
must visit each customer and pickup their packages without exceeding the vehicles capacity. The goal is to find the most
optimal route for each vehicle with respect to capacity and overall distance.

### Research
We solve this problem using evolutionary approaches, specifically genetic algorithms, to optimize the problem set
into its best known solution.

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
2. Node x-coordinate - (int) (float)
3. Node y-coordinate - (int) (float)
4. Node service demand - (int) (float)

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

- Python 3 - version must be `>= 3.12`
- Python `pip` or `pipenv`

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
Running without command line arguments runs the program with the default arguments:
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
usage: driver.py [-h] [-o] [-p] [-s] [-g] [-m] [-c] [-r]
                 [-B | -C | -E | -O | -X] [-I | -W | -G | -K | -T | -F | -D]
                 [-i ] [-P] [-A] [-S] [-R] [-M]
                 file

Runs the CVRP with any of the optional arguments

positional arguments:
  file                the path to the problem set

optional arguments:
  -h, --help          show this help message and exit
  -o , --output       the path to output the results (creates the path if it does not exist)
  -p , --pop          the population size
  -s , --sel          the selection size
  -g , --ngen         the generation size
  -m , --mutpb        the mutation probability
  -c , --cxpb         the crossover probability
  -r , --run          the number of times to run the problem
  -B, --brxo          use best route crossover
  -C, --cxo           use cycle crossover
  -E, --erxo          use edge recombination crossover
  -O, --oxo           use order crossover
  -X, --pmxo          use partially mapped crossover (PMX)
  -I, --vmt           use inversion mutation
  -W, --swmt          use swap mutation
  -G, --gvmt          use GVR based scramble mutation
  -K, --scmt          use scramble mutation
  -T, --topt          use 2-opt local search mutation
  -F, --oropt         use or-opt mutation
  -D, --disp          use displacement mutation
  -i [], --indent []  the indentation amount of the result string
  -P, --pgen          prints the current generation
  -A, --agen          prints the average fitness every 1000 generations
  -S, --save          saves the results to a file
  -R, --routes        adds every route (verbose) of the best individual to the result
  -M, --plot          plot average fitness across generations with matplotlib
```
#### Saving Results

If the `-S` option is specified to save the results to a file, the output is stored in a `results` directory as a JSON 
file.
If the `results` directory does not exist, one will be created. If the `-o` option is specified with `-S`, 
the results are saved to a path. The path is created if it does not
exist.

The file naming convention for saving results are as follows:  

CROSSOVER ALGORITHM\_GENERATION SIZE\_CROSSOVER PROBABILITY\_DATA SET\_\_YYYYMMDD\_\_HH\_MM\_SSAM/PM

An example is:  

`best_route_xo_100000_0.85_F-n45-k4__20201213__06_03_29PM`  

If the `-M` option is specified to save matplotlib results ta file, the output is stored in a `results` directory 
similar
to saving the results to a file. If the `results` directory does not exist, one will be created for you. 
The file naming convention for matplotlib results are as follows:  

CROSSOVER ALGORITHM\_GENERATION SIZE\_CROSSOVER PROBABILITY\_DATA SET\_\_YYYYMMDD\_\_HH\_MM\_SSAM/PM\_\_RUN NUMBER\_\_FITNESS VALUE  

An example is:  

`best_route_xo_100000_0.85_F-n45-k4__20201213__06_03_29PM__RUN1__FIT732`

### Using Package

For non-terminal based runs and integration, a CVRP object can be created and run by calling the `run()` function. 
```python
import json
from ocvrp import algorithms as algo
from ocvrp.cvrp import CVRP
from ocvrp.util import CVRPEncoder

# The path to the .ocvrp file is the problem set for this instance
cvrp = CVRP("./data/A-n54-k7.ocvrp", cxpb=0.75, ngen=50_000, pgen=True, plot=True, cx_algo=algo.edge_recomb_xo)

# Result contains a dict of information about the run which includes the best individual found 
result = cvrp.run()

# Save the matplotlib figure to a file (only if plot=True)
if '_fig' in result:
    result['_fig'].savefig("A-n54-k7-Run1.png", bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(result['_fig'])

js_res = json.dumps(obj=result, cls=CVRPEncoder, indent=2)
print(js_res)

cvrp.reset()
```

`CVRP.run()` will return a dictionary object of the run summary. An example of the object is provided:

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
'mat_plot': <module 'matplotlib.pyplot'>,
'_fig': <Figure>,
'best_individual_fitness': 729
}
```
If `verbose_routes` is set to `True` for the CVRP instance, the exact route of the best individual will be
added to the dictionary object. To convert the dictionary to a JSON object, a `CVRPEncoder` class is
provided to specify to the `json.dumps` function.

### Testing

PowerShell 7 test suites live in the `testing/` directory:

| Script | Description |
|---|---|
| `CVRP_Test.ps1` | Sequential suite — tests every crossover, mutation, dataset, combo, multi-run, and CLI flag |
| `CVRP_TestParallel.ps1` | Parallel matrix — runs crossover × mutation × dataset combos via `Start-ThreadJob` |

Run them with:
```powershell
pwsh testing/CVRP_Test.ps1
pwsh testing/CVRP_TestParallel.ps1
```

Both scripts validate exit codes and check for `best_individual_fitness` in the JSON output. The parallel script defaults to 4 concurrent jobs; adjust `-MaxParallel` inside the script if needed.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for the full list of changes.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.
