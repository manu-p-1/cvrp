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

## Execution

### Requirements

- Python 3 - version must be `>= 3.6.0`

### Execution
Running `python driver.py` runs the program with default arguments:
- Population Size: 600
- Selection Size: 5
- Number of Generations: 100000
- Mutation Probability: 0.15
- Crossover Probability: 0.75
- Crossover: Best Route CXO
- Mutation: Inversion MUT 

To specify arguments apart from default, view the optional arguments by running `python driver.py -h` or
`python driver.py --help`

```
usage: driver.py [-h] [-f] [-p] [-s] [-g] [-m] [-c] [-r] [-B | -C | -E | -O]
                 [-I | -W | -G] [-i ] [-P] [-A] [-S] [-M]

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
  -M, --plot          plot average fitness across generations with matplotlib
```
