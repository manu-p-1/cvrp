# Optimizing the Capacitated Vehicle Routing Problem using Genetic Algorithms

## Info

### Authors
- Vrund Parikh
- Manu Puduvalli
- Lok Kwong
- Glen George
- Samuel Yuen

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
*This paper studies the effect of optimized heuristic crossover operations by utilizing a Genetic Algorithm to optimize the Capacitated Vehicle Routing Problem (CVRP) and understand the effect on optimized crossovers on diversity maintenance. Best-Route Crossover showed promising results within 10% of the optimal solution on various well-known CVRP datasets. An optimized type of Cycle Crossover exhibited a solution to maintain population diversity and prevent premature convergence.*

### Index Terms
*Best-Route Crossover, Capacitated Vehicle Routing Problem, Cycle Crossover, Genetic Algorithm, Genetic Vehicle Representation, Optimization, Travelling Salesperson Problem, Vehicle Routing Problem*

## Problem Sets
Problem sets are organized in a custom format called `.ocvrp` for easier data processing. It's a simple format that is easy to use. It contains 5 important headers:

1. Name of the problem set
2. Comments for the problem set
3. The dimension of the problem set (including the depot)
4. The maximum capacity each vehicle is able to hold (as an integer)
5. The optimal value for the data set (rounded to the nearest integer)

These five headers must be specified in the same order as specified above to ensure proper functionality. Data is organized in tabular format with 4 elements per row. These elements are:

1. Node number
2. Node x-coordinate
3. Node y-coordinate
4. Node service demand

The first node **must be the depot location** and comments or any other unrecognizable characters are forbidden. An example of the format is shown below.

```
NAME: A-n54-k7
COMMENTS: None
DIM: 54
CAPACITY: 100
OPTIMAL: 1167
1 61 5 0
2 85 53 24
3 17 57 9
4 49 93 15
5 69 11 17

...
```

## Execution

### Requirements

- Python 3 - version must be `>= 3.6.0`
- Python `pip` or `pipenv`

### Setup

There are two options to run the algorithm:

1. With pip
    ```
    python install pip
    pip install matplotlib
    python driver.py
    ```
2.  With pipenv
    ```
    python install pip
    pip install pipenv
    pipenv install
    pipenv run python driver.py
    ```
Running without command line arguments runs the program with the default arguments:
- Population Size: `600`
- Selection Size: `5`
- Number of Generations: `100000`
- Mutation Probability: `0.15`
- Crossover Probability: `0.85`
- Crossover: `best_route_xo`
- Mutation: `inversion_mut` 

### Command Line

To run the program on your command line, view the optional arguments by running `python driver.py -h` or
`python driver.py --help`

```
usage: driver.py [-h] [-f] [-p] [-s] [-g] [-m] [-c] [-r] [-B | -C | -E | -O]
                 [-I | -W | -G] [-i ] [-P] [-A] [-S] [-R] [-M]

Runs the CVRP with any of the optional arguments

optional arguments:
  -h, --help          show this help message and exit
  -f , --file         the path to the problem set
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
  -I, --vmt           use inversion mutation
  -W, --swmt          use swap mutation
  -G, --gvmt          use GVR based scramble mutation
  -i [], --indent []  the indentation amount of the result string
  -P, --pgen          prints the current generation
  -A, --agen          prints the average fitness every 1000 generations
  -S, --save          saves the results to a file
  -R, --routes        adds every route (verbose) of the best individual to the result
  -M, --plot          plot average fitness across generations with matplotlib
```
#### Saving Results

If the `-S` option is specified to save the results to a file, the output is stored in a `results` directory as a JSON file.
If the `results` directory does not exist, one will be created for you. The file naming convention for saving results are as follows:  

CROSSOVER ALGORITHM\_GENERATION SIZE\_CROSSOVER PROBABILITY\_DATA SET\_\_YYYYMMDD\_\_HH\_MM\_SSAM/PM

An example is:  

`best_route_xo_100000_0.85_F-n45-k4__20201213__06_03_29PM`  

If the `-M` option is specified to save matplotlib results ta file, the output is stored in a `results` directory similar
to saving the results to a file. If the `results` directory does not exist, one will be created for you. The file naming convention for matplotlib results are as follows:  

CROSSOVER ALGORITHM\_GENERATION SIZE\_CROSSOVER PROBABILITY\_DATA SET\_\_YYYYMMDD\_\_HH\_MM\_SSAM/PM\_\_RUN NUMBER\_\_FITNESS VALUE  

An example is:  

`best_route_xo_100000_0.85_F-n45-k4__20201213__06_03_29PM__RUN1__FIT732`

### Using Package

For non-terminal based runs and integration, a CVRP object can be created and run by calling the `run()` function. 
```python
from ocvrp import algorithms
from ocvrp.cvrp import CVRP
from ocvrp.util import BuildingEncoder

cvrp = CVRP(cxpb=0.75, ngen=50_000, pgen=True, plot=True)

# Result contains a dict of information about the run which includes the best individual found 
result = cvrp.run()

js_res = json.dumps(obj=result, cls=BuildingEncoder, indent=2)
print(js_res)

cvrp.reset()
```

`CVRP.run()` will return a dictionary object of the run summary. An example of the object is provided:

```python
{
	'name': 'CVRP', 
	'problem_set_name': 'F-n45-k4', 
	'problem_set_optimal': 724, 
	'time': '522.453125 seconds', 
	'vehicles':4, 
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
	'best_individual_fitness': 729
}
```
If `verbose_routes` is set to `True` for the CVRP instance, the exact route of the best individual will be
added to the dictionary object. To convert the dictionary to a JSON object, a `BuildingEncoder` class is
provided to specify to the `json.dumps` function.

### Testing

A PowerShell script template has been provided under the `testing` directory for batch processing algorithm runs. There are two versions: 

1. `CVRP_Test.ps1`
2. `CVRP_TestThreadedJob.ps1`

The first option runs a single-threaded job. The second option runs a multi-threaded job but requires PowerShell version 7. To check your PowerShell version, run the following command on your PowerShell terminal:

```powershell
Get-Host | Select-Object Version
```

For more information on PowerShell Jobs visit:  
<https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.core/start-job?view=powershell-7.1>
<https://docs.microsoft.com/en-us/powershell/module/threadjob/start-threadjob?view=powershell-7.1>
<https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.core/get-job?view=powershell-7.1>

We encourage you to modify the script template to meet your needs.
