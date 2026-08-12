"""
https://github.com/manu-p-1/cvrp
constructive.py

Classical constructive heuristics for generating feasible initial CVRP
solutions. These produce complete route plans from scratch and are used
to seed the genetic algorithm population with high-quality starting points,
significantly accelerating convergence.
"""

import math
import random

from ocvrp.util import Building, Individual


def nearest_neighbor(depot, buildings, capacity, dist_matrix):
    """Greedy nearest-neighbor construction heuristic.

    Starting from the depot, repeatedly visits the closest unvisited customer
    whose demand fits the remaining vehicle capacity. When no feasible
    customer exists, the vehicle returns to the depot and a new route begins.

    Reference:
        Laporte, G. (1992). "The Vehicle Routing Problem: An overview of
        exact and approximate algorithms." European Journal of Operational
        Research, 59(3), 345-358.

    :param depot: The depot Building object
    :param buildings: List of customer Building objects
    :param capacity: Maximum vehicle capacity
    :param dist_matrix: Precomputed DistanceMatrix instance
    :return: List of routes (each route is a list of Building objects)
    """
    unvisited = set(buildings)
    routes = []

    while unvisited:
        route = []
        load = 0
        current = depot

        while True:
            best = None
            best_dist = float('inf')
            for b in unvisited:
                if load + b.quant <= capacity:
                    d = dist_matrix.dist(current, b)
                    if d < best_dist:
                        best_dist = d
                        best = b

            if best is None:
                break

            route.append(best)
            unvisited.remove(best)
            load += best.quant
            current = best

        if route:
            routes.append(route)

    return routes


def clarke_wright_savings(depot, buildings, capacity, dist_matrix):
    """Parallel savings algorithm of Clarke and Wright.

    Computes savings s(i,j) = d(depot,i) + d(depot,j) - d(i,j) for every
    customer pair. Iterates through savings in descending order, merging
    the routes of i and j when (a) they belong to different routes,
    (b) both are at the boundary (first or last) of their respective routes,
    and (c) the merged route does not exceed vehicle capacity.

    Reference:
        Clarke, G. and Wright, J.W. (1964). "Scheduling of vehicles from
        a central depot to a number of delivery points." Operations
        Research, 12(4), 568-581.

    :param depot: The depot Building object
    :param buildings: List of customer Building objects
    :param capacity: Maximum vehicle capacity
    :param dist_matrix: Precomputed DistanceMatrix instance
    :return: List of routes (each route is a list of Building objects)
    """
    # Each customer starts in its own singleton route
    route_of = {}          # customer node -> route id
    routes = {}            # route id -> list of Buildings
    route_load = {}        # route id -> total demand

    for b in buildings:
        route_of[b.node] = b.node
        routes[b.node] = [b]
        route_load[b.node] = b.quant

    # Compute all pairwise savings
    blist = list(buildings)
    savings = []
    for i in range(len(blist)):
        for j in range(i + 1, len(blist)):
            bi, bj = blist[i], blist[j]
            s = (dist_matrix.dist(depot, bi)
                 + dist_matrix.dist(depot, bj)
                 - dist_matrix.dist(bi, bj))
            if s > 0:
                savings.append((s, bi, bj))

    savings.sort(key=lambda x: x[0], reverse=True)

    for s, bi, bj in savings:
        ri = route_of.get(bi.node)
        rj = route_of.get(bj.node)

        if ri is None or rj is None or ri == rj:
            continue
        if ri not in routes or rj not in routes:
            continue

        # Capacity check
        if route_load[ri] + route_load[rj] > capacity:
            continue

        route_i = routes[ri]
        route_j = routes[rj]

        # Merge only if both customers are at route endpoints
        merged = None
        if route_i[-1] == bi and route_j[0] == bj:
            merged = route_i + route_j
        elif route_i[-1] == bi and route_j[-1] == bj:
            merged = route_i + route_j[::-1]
        elif route_i[0] == bi and route_j[0] == bj:
            merged = route_i[::-1] + route_j
        elif route_i[0] == bi and route_j[-1] == bj:
            merged = route_j + route_i

        if merged is None:
            continue

        # Perform the merge under ri's key
        new_load = route_load[ri] + route_load[rj]
        routes[ri] = merged
        route_load[ri] = new_load

        for b in merged:
            route_of[b.node] = ri

        del routes[rj]
        del route_load[rj]

    return list(routes.values())


def sweep_heuristic(depot, buildings, capacity, dist_matrix):
    """Sweep algorithm of Gillett and Miller.

    Sorts customers by polar angle from the depot, then sweeps through
    them, assigning customers to the current route until the vehicle
    capacity is reached, at which point a new route is opened. A random
    starting angle is used so repeated calls produce diverse solutions.

    Reference:
        Gillett, B.E. and Miller, L.R. (1974). "A heuristic algorithm
        for the vehicle-dispatch problem." Operations Research, 22(2),
        340-349.

    :param depot: The depot Building object
    :param buildings: List of customer Building objects
    :param capacity: Maximum vehicle capacity
    :param dist_matrix: Precomputed DistanceMatrix instance (unused but
        accepted for interface consistency)
    :return: List of routes (each route is a list of Building objects)
    """
    def angle_from_depot(b):
        return math.atan2(b.y - depot.y, b.x - depot.x)

    sorted_buildings = sorted(buildings, key=angle_from_depot)

    # Randomize starting position for diversity across multiple calls
    n = len(sorted_buildings)
    start = random.randint(0, max(0, n - 1))
    ordered = sorted_buildings[start:] + sorted_buildings[:start]

    routes = []
    current_route = []
    current_load = 0

    for b in ordered:
        if current_load + b.quant > capacity:
            if current_route:
                routes.append(current_route)
            current_route = [b]
            current_load = b.quant
        else:
            current_route.append(b)
            current_load += b.quant

    if current_route:
        routes.append(current_route)

    return routes


def routes_to_individual(routes):
    """Flatten a list of routes into an Individual gene sequence.

    :param routes: List of routes (each a list of Building objects)
    :return: An Individual with concatenated genes and fitness=None
    """
    genes = []
    for route in routes:
        genes.extend(route)
    return Individual(genes, None)
