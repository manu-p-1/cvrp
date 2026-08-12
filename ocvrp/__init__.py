"""
ocvrp - Capacitated Vehicle Routing Problem Optimizer

A genetic algorithm framework for solving the Capacitated Vehicle Routing
Problem (CVRP) with constructive heuristics, inter-route local search
operators, precomputed distance matrices, and parallel island-model support.

Modules
-------
algorithms      Crossover and mutation operators for the GA
constructive    Classical construction heuristics (nearest-neighbor,
                Clarke-Wright savings, sweep)
cvrp            Core CVRP solver class with evolutionary loop
distance        Precomputed all-pairs distance matrix
local_search    Inter-route improvement operators (relocate, exchange, 2-opt*)
parallel        Multi-start island-model parallelization
util            Data structures (Building, Individual) and .ocvrp parser
"""

__version__ = "2.0.0"
