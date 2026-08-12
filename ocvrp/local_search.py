"""
https://github.com/manu-p-1/cvrp
local_search.py

Inter-route and composite local search operators for CVRP. These operate
on the partitioned (GVR) route representation and can move customers
between routes, providing improvements that intra-route operators alone
cannot achieve.

All public functions conform to the mutation signature
    fn(child: Individual, cvrp) -> Individual
so they are directly usable as mutation operators in the GA.
"""

from ocvrp.util import Building, Individual


def _route_load(route):
    """Total demand of a route."""
    return sum(b.quant for b in route)


def relocate_mut(child, cvrp=None):
    """Inter-route relocate: move one customer to the best feasible position
    in another route.

    Evaluates all (customer, target-route, insertion-position) triples and
    applies the single move yielding the greatest distance reduction. If no
    improving move exists the individual is returned unchanged.

    Reference:
        Savelsbergh, M.W.P. (1992). "The vehicle routing problem with time
        windows: Minimizing route duration." ORSA Journal on Computing,
        4(2), 146-154.

    :param child: The child Individual object
    :param cvrp: An instance of the CVRP class
    :return: A new (possibly improved) Individual
    """
    partitioned = cvrp.partition_routes(child)
    if len(partitioned) <= 1:
        return Individual(child[:], None)

    dm = getattr(cvrp, 'dist_matrix', None)
    d = dm.dist if dm else Building.distance
    depot = cvrp.depot
    cap = cvrp.vehicle_cap

    route_keys = list(partitioned.keys())
    loads = {rk: _route_load(partitioned[rk]) for rk in route_keys}

    best_delta = 0
    best_move = None

    for rk1 in route_keys:
        route1 = partitioned[rk1]
        for idx in range(len(route1)):
            customer = route1[idx]
            prev1 = depot if idx == 0 else route1[idx - 1]
            next1 = depot if idx == len(route1) - 1 else route1[idx + 1]
            removal_saving = (d(prev1, customer) + d(customer, next1)
                              - d(prev1, next1))

            for rk2 in route_keys:
                if rk2 == rk1:
                    continue
                if loads[rk2] + customer.quant > cap:
                    continue

                route2 = partitioned[rk2]
                for pos in range(len(route2) + 1):
                    prev2 = depot if pos == 0 else route2[pos - 1]
                    next2 = depot if pos == len(route2) else route2[pos]
                    insertion_cost = (d(prev2, customer) + d(customer, next2)
                                     - d(prev2, next2))

                    delta = insertion_cost - removal_saving
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (rk1, idx, rk2, pos)

    if best_move is None:
        return Individual(child[:], None)

    rk1, idx, rk2, pos = best_move
    customer = partitioned[rk1].pop(idx)
    partitioned[rk2].insert(pos, customer)

    # Remove empty routes
    partitioned = {k: v for k, v in partitioned.items() if v}

    return Individual(cvrp.de_partition_routes(partitioned), None)


def exchange_mut(child, cvrp=None):
    """Inter-route exchange: swap one customer from each of two routes.

    Evaluates all (customer_i, customer_j) pairs across different routes
    and performs the swap yielding the greatest distance reduction.

    Reference:
        Kindervater, G.A.P. and Savelsbergh, M.W.P. (1997). "Vehicle
        routing: Handling edge exchanges." In Local Search in Combinatorial
        Optimization, Aarts, E. and Lenstra, J.K. (eds.), Wiley, pp. 337-360.

    :param child: The child Individual object
    :param cvrp: An instance of the CVRP class
    :return: A new (possibly improved) Individual
    """
    partitioned = cvrp.partition_routes(child)
    if len(partitioned) <= 1:
        return Individual(child[:], None)

    dm = getattr(cvrp, 'dist_matrix', None)
    d = dm.dist if dm else Building.distance
    depot = cvrp.depot
    cap = cvrp.vehicle_cap

    route_keys = list(partitioned.keys())
    loads = {rk: _route_load(partitioned[rk]) for rk in route_keys}

    best_delta = 0
    best_swap = None

    for ki_idx in range(len(route_keys)):
        rk1 = route_keys[ki_idx]
        route1 = partitioned[rk1]
        for i in range(len(route1)):
            ci = route1[i]
            prev_i = depot if i == 0 else route1[i - 1]
            next_i = depot if i == len(route1) - 1 else route1[i + 1]
            cost_i = d(prev_i, ci) + d(ci, next_i)

            for kj_idx in range(ki_idx + 1, len(route_keys)):
                rk2 = route_keys[kj_idx]
                route2 = partitioned[rk2]
                for j in range(len(route2)):
                    cj = route2[j]

                    # Capacity feasibility after swap
                    new_load1 = loads[rk1] - ci.quant + cj.quant
                    new_load2 = loads[rk2] - cj.quant + ci.quant
                    if new_load1 > cap or new_load2 > cap:
                        continue

                    prev_j = depot if j == 0 else route2[j - 1]
                    next_j = depot if j == len(route2) - 1 else route2[j + 1]
                    cost_j = d(prev_j, cj) + d(cj, next_j)

                    new_cost_i = d(prev_i, cj) + d(cj, next_i)
                    new_cost_j = d(prev_j, ci) + d(ci, next_j)

                    delta = (new_cost_i + new_cost_j) - (cost_i + cost_j)
                    if delta < best_delta:
                        best_delta = delta
                        best_swap = (rk1, i, rk2, j)

    if best_swap is None:
        return Individual(child[:], None)

    rk1, i, rk2, j = best_swap
    partitioned[rk1][i], partitioned[rk2][j] = partitioned[rk2][j], partitioned[rk1][i]

    return Individual(cvrp.de_partition_routes(partitioned), None)


def two_opt_star_mut(child, cvrp=None):
    """2-opt* operator: exchange tails between two routes.

    For every pair of routes and every pair of edge positions, evaluates
    whether swapping the route tails at those edges improves total distance
    without violating capacity. Applies the single best improving move.

    Reference:
        Potvin, J.-Y. and Rousseau, J.-M. (1995). "An exchange heuristic
        for routeing problems with time windows." Journal of the
        Operational Research Society, 46(12), 1433-1446.

    :param child: The child Individual object
    :param cvrp: An instance of the CVRP class
    :return: A new (possibly improved) Individual
    """
    partitioned = cvrp.partition_routes(child)
    if len(partitioned) <= 1:
        return Individual(child[:], None)

    dm = getattr(cvrp, 'dist_matrix', None)
    d = dm.dist if dm else Building.distance
    depot = cvrp.depot
    cap = cvrp.vehicle_cap

    route_keys = list(partitioned.keys())

    best_delta = 0
    best_move = None

    for ki_idx in range(len(route_keys)):
        rk1 = route_keys[ki_idx]
        route1 = partitioned[rk1]
        n1 = len(route1)

        for kj_idx in range(ki_idx + 1, len(route_keys)):
            rk2 = route_keys[kj_idx]
            route2 = partitioned[rk2]
            n2 = len(route2)

            for i in range(n1):
                node_i = route1[i]
                next_i = depot if i == n1 - 1 else route1[i + 1]

                # Demand of tail of route1 after position i
                tail1_demand = sum(b.quant for b in route1[i + 1:])
                head1_demand = sum(b.quant for b in route1[:i + 1])

                for j in range(n2):
                    node_j = route2[j]
                    next_j = depot if j == n2 - 1 else route2[j + 1]

                    tail2_demand = sum(b.quant for b in route2[j + 1:])
                    head2_demand = sum(b.quant for b in route2[:j + 1])

                    # New route1 = route1[:i+1] + route2[j+1:]
                    # New route2 = route2[:j+1] + route1[i+1:]
                    if head1_demand + tail2_demand > cap:
                        continue
                    if head2_demand + tail1_demand > cap:
                        continue

                    old_cost = d(node_i, next_i) + d(node_j, next_j)
                    new_cost = d(node_i, next_j) + d(node_j, next_i)
                    delta = new_cost - old_cost

                    if delta < best_delta:
                        best_delta = delta
                        best_move = (rk1, i, rk2, j)

    if best_move is None:
        return Individual(child[:], None)

    rk1, i, rk2, j = best_move
    route1 = partitioned[rk1]
    route2 = partitioned[rk2]

    new_route1 = route1[:i + 1] + route2[j + 1:]
    new_route2 = route2[:j + 1] + route1[i + 1:]

    partitioned[rk1] = new_route1
    partitioned[rk2] = new_route2

    # Remove empty routes
    partitioned = {k: v for k, v in partitioned.items() if v}

    return Individual(cvrp.de_partition_routes(partitioned), None)
