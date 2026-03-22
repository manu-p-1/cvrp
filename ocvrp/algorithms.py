"""
https://github.com/manu-p-1/cvrp
algorithms.py

This module contains a list of functions to perform crossover and mutation operations.
"""
import math
import random as r
from typing import Dict, Tuple, List

from ocvrp.util import Building, Individual


def _best_route_xo_single(ind1: Individual, ind2: Individual, cvrp) -> Individual:
    """
    Produces a single child from best route crossover between two individuals.
    """
    ind1_partitioned = cvrp.partition_routes(ind1)
    ind2_partitioned = cvrp.partition_routes(ind2)

    # choose a random route from chromosome 1
    route_number = r.choice(list(ind1_partitioned.keys()))
    route_from_ind1 = ind1_partitioned[route_number]

    # Random route chosen to be replaced in chromosome 2 --> list
    section_from_ind1 = route_from_ind1[r.randint(0, len(route_from_ind1) - 1):]

    for route_num in ind2_partitioned.keys():
        # Removing duplicates before we insert the genes
        for building in ind2_partitioned[route_num][:]:
            if building in section_from_ind1:
                ind2_partitioned[route_num].remove(building)

    child = ind2_partitioned

    closest_child_route = 0
    closest_child_bldg_idx = 0
    closest_distance = -1

    for route_num in child.keys():
        for idx, building in enumerate(child[route_num]):
            building_ind1 = section_from_ind1[0]
            distance = Building.distance(building_ind1, building)

            if closest_distance == -1:
                closest_distance = distance
                closest_child_bldg_idx = idx
                closest_child_route = route_num
            elif distance < closest_distance:
                closest_distance = distance
                closest_child_bldg_idx = idx
                closest_child_route = route_num

    child[closest_child_route][closest_child_bldg_idx + 1:closest_child_bldg_idx + 1] = section_from_ind1

    return Individual(cvrp.de_partition_routes(child), None)


def best_route_xo(ind1: Individual, ind2: Individual, cvrp) -> Tuple[Individual, Individual]:
    """
    Given two individuals, does best route crossover proposed by Costa et al.

    Reference:
    Costa, E., Machado, P., Pereira, B., Tavares, J.,
    "Crossover and Diversity: A Study about GVR", In Proceedings of the Analysis and Design of Representations and
    Operators (ADorO'2003) a bird-of-a-feather workshop at (GECCO-2003), Chicago, Illinois, USA, 12-16 July 2003

    :param ind1: The first Individual
    :param ind2: The second Individual
    :param cvrp: An instance of the CVRP class
    :return: A tuple of two Individuals with recombinant genes (fitness not yet evaluated).
    """
    child1 = _best_route_xo_single(ind1, ind2, cvrp)
    child2 = _best_route_xo_single(ind2, ind1, cvrp)
    return child1, child2


class CycleInfo:
    """
    A helper class to find cycles given two individuals for cycle crossover
    """

    def __init__(self, father: Individual, mother: Individual):
        """
        Creates a new instance of the Cycle info. The cycles are created from father
        to mother.

        :param father: The first Individual
        :param mother: The second Individual
        """
        self._mother = mother
        self._father = father

    @staticmethod
    def _find_cycle(start, correspondence_map):
        """
        This FUNCTION was taken and modified from the attributed author
        ----------------------------------------------------------
        AUTHOR: EVAN CONRAD - https://github.com/Flaque
        TITLE: Python-GA
        YEAR: 2017
        AVAILABILITY:
        https://github.com/Flaque/Python-GA/blob/master/cx.py
        ----------------------------------------------------------
        :param start: The starting number of the cycle
        :param correspondence_map: A map corresponding the fathers allele index to the mothers
        :return: the cycle formation of the chromosome
        """
        cycle = [start]
        current = correspondence_map[start]
        while current not in cycle:
            cycle.append(current)
            current = correspondence_map[current]
        return cycle

    def get_cycle_info(self):
        """
        Helper function which hides the implementation of _get_cycle_info
        :return: The cycles occurring in both parents as a list
        """
        return self._get_cycle_info()

    def _get_cycle_info(self):
        """
        Creates a relationship correspondence map between the mother and father Individuals.
        We forgo any duplicate or repeating numbers in the cycles
        :return: The cycles occurring in both parents as a list
        """
        f_cpy = self._father[:]
        m_cpy = self._mother[:]
        correspondence_map = dict(zip(f_cpy, m_cpy))

        cycles_list = []
        for i in range(len(f_cpy)):
            cycle = self._find_cycle(f_cpy[i], correspondence_map)

            if len(cycles_list) == 0:
                cycles_list.append(cycle)
            else:
                flag = False
                for j in cycles_list:
                    for k in cycle:
                        if k in j:
                            flag = True
                            break
                if not flag:
                    cycles_list.append(cycle)
        return cycles_list


def cycle_xo(ind1: Individual, ind2: Individual, cvrp) -> Tuple[Individual, Individual]:
    """
    Given two individuals, does an optimized version of cycle crossover proposed by Nazif and Lee.

    Reference:
    Habibeh Nazif and Lai Soon Lee, "Optimised crossover genetic algorithm for capacitated vehicle routing problem",
    Applied Mathematical Modeling 36, 2011.

    We decide to modify the proposed crossover by finding as many combinations of cycles within a specified runtime,
    instead of 2^c (proposed by authors) or 2^5 (recommendation by authors) combinations. The runtime is
    gradually multiplied by a factor of 2 until a cycle length of 4. For lengths greater than or equal to 5, the
    runtime is calculated by math.ceil(5 * math.log(cycle_len) + 24).

    :param ind1: The first Individual
    :param ind2: The second Individual
    :param cvrp: An instance of the CVRP class
    :return: A tuple of (o_child, e_child) Individuals with fitness evaluated.
    """
    cl = CycleInfo(ind1, ind2).get_cycle_info()
    p_children = []
    cycle_len = len(cl)

    # O(1) position lookup for ind1 genes
    ind1_pos = {gene: idx for idx, gene in enumerate(ind1)}

    # Calculates number of iterations to generate combos before quitting.
    if cycle_len == 1:
        max_iters = 2
    elif cycle_len == 2:
        max_iters = 4
    elif cycle_len == 3:
        max_iters = 8
    elif cycle_len == 4:
        max_iters = 16
    else:
        max_iters = math.ceil(5 * math.log(cycle_len) + 24)

    for _ in range(max_iters):
        o_child = Individual([None] * len(ind1), None)
        e_child = Individual([None] * len(ind1), None)

        # The binaries represent combination of binaries
        binaries = [bool(r.getrandbits(1)) for _ in cl]
        all_ = len(set(binaries))

        # With binaries of all identical values, we ensure at least one change in the cycle combination
        if all_ == 1:
            ri = r.randint(0, len(binaries) - 1)
            binaries[ri] = not binaries[ri]

        # We forgo this binary if it already exists
        if any(b['binaries'] == binaries for b in p_children):
            continue

        bin_counter = 0
        for c in cl:
            if not binaries[bin_counter]:
                # if 0, get from ind2
                for allele in c:
                    ind1_idx = ind1_pos[allele]
                    o_child[ind1_idx] = ind2[ind1_idx]
                    e_child[ind1_idx] = ind1[ind1_idx]
            else:
                # else 1, get from ind1
                for allele in c:
                    ind1_idx = ind1_pos[allele]
                    o_child[ind1_idx] = allele
                    e_child[ind1_idx] = ind2[ind1_idx]
            bin_counter += 1

        o_child.fitness = cvrp.calc_fitness(o_child)
        p_children.append({
            "o-child": o_child,
            "e-child": e_child,
            "cycles": cl,
            "binaries": binaries
        })

    # Sort based on O-Child fitness, then grab the corresponding e-child for that o-child as well
    best = min(p_children, key=lambda pc: pc['o-child'].fitness)
    best["e-child"].fitness = cvrp.calc_fitness(best['e-child'])

    return best["o-child"], best["e-child"]


def _edge_recomb_xo_single(ind1: Individual, ind2: Individual) -> Individual:
    """
    Produces a single child from edge recombination crossover between two individuals.
    """
    ind1_adjacent = {}
    ind2_adjacent = {}
    ind_len = len(ind1)

    for i in range(ind_len):
        bldg1 = ind1[i]
        bldg2 = ind2[i]

        if i == 0:
            ind1_adjacent[bldg1] = [ind1[i + 1], ind1[ind_len - 1]]
            ind2_adjacent[bldg2] = [ind2[i + 1], ind2[ind_len - 1]]
        elif i == ind_len - 1:
            ind1_adjacent[bldg1] = [ind1[i - 1], ind1[0]]
            ind2_adjacent[bldg2] = [ind2[i - 1], ind2[0]]  # Fixed: was ind1[0]
        else:
            ind1_adjacent[bldg1] = [ind1[i - 1], ind1[i + 1]]
            ind2_adjacent[bldg2] = [ind2[i - 1], ind2[i + 1]]

    # Create a union of the neighbors from each individuals' adjacency matrix
    ind_union = {}
    for bldg, neighbors in ind1_adjacent.items():
        ind_union[bldg] = list(set(neighbors) | set(ind2_adjacent[bldg]))

    child = []
    n = r.choice(list(ind_union.keys()))

    while len(child) < ind_len:
        child.append(n)

        for neighbors in list(ind_union.values()):
            if n in neighbors:
                neighbors.remove(n)

        if len(ind_union[n]) > 0:
            n_star = min(ind_union[n], key=lambda d: len(ind_union[d]))
        else:
            diff = list(set(child) ^ set(list(ind_union.keys())))
            n_star = r.choice(diff) if len(diff) > 0 else None
        n = n_star

    return Individual(child, None)


def edge_recomb_xo(ind1: Individual, ind2: Individual, cvrp=None) -> Tuple[Individual, Individual]:
    """
    Given two individuals, does an edge recombination proposed by Whitley et al.

    Reference:
    Whitley, Darrell; Timothy Starkweather; D'Ann Fuquay (1989). "Scheduling problems and traveling salesman:
    The genetic edge recombination operator". International Conference on Genetic Algorithms. pp. 133-140

    :param ind1: The first Individual
    :param ind2: The second Individual
    :param cvrp: An instance of the CVRP class (unused)
    :return: A tuple of two Individuals with recombinant genes (fitness not yet evaluated).
    """
    child1 = _edge_recomb_xo_single(ind1, ind2)
    child2 = _edge_recomb_xo_single(ind2, ind1)
    return child1, child2


def _order_xo_single(ind1: Individual, ind2: Individual) -> Individual:
    """
    Produces a single child from order crossover between two individuals.
    """
    bound = len(ind1)

    cxp1 = r.randint(0, (bound - 1) // 2)
    cxp2 = r.randint(((bound - 1) // 2) + 1, bound - 1)
    child = [None] * bound
    child_set = set()
    for i in range(cxp1, cxp2 + 1):
        child[i] = ind1[i]
        child_set.add(ind1[i])

    parent_idx = cxp2 + 1
    child_idx = cxp2 + 1

    parent_bound = child_bound = bound

    while child_idx != cxp1:
        if parent_idx == parent_bound:
            parent_idx = 0
            parent_bound = cxp2 + 1

        if child_idx == child_bound:
            child_idx = 0
            child_bound = cxp1

        if ind2[parent_idx] not in child_set:
            child[child_idx] = ind2[parent_idx]
            child_set.add(ind2[parent_idx])
            child_idx += 1
        parent_idx += 1

    return Individual(child, None)


def order_xo(ind1: Individual, ind2: Individual, cvrp=None) -> Tuple[Individual, Individual]:
    """
    Given two individuals, does an order crossover. Order crossover operates by choosing two indices
    within the first parent and copying everything within those indices to the child. The rest of the
    elements go from the second parent to the child. Duplicate values are ignored.

    :param ind1: The first Individual
    :param ind2: The second Individual
    :param cvrp: An instance of the CVRP class (unused)
    :return: A tuple of two Individuals with recombinant genes (fitness not yet evaluated).
    """
    child1 = _order_xo_single(ind1, ind2)
    child2 = _order_xo_single(ind2, ind1)
    return child1, child2


def inversion_mut(child: Individual, cvrp=None) -> Individual:
    """
    Selects two random positions and reverses the segment between them.

    :param child: The child Individual object
    :param cvrp: An instance of the CVRP class (unused)
    :return: A new mutated Individual
    """
    new_child = Individual(child[:], None)

    idx1, idx2 = sorted(r.sample(range(len(new_child)), 2))

    # Swap the values until all values are mirrored
    while idx1 <= idx2:
        _swap(new_child, idx1, idx2)
        idx1 += 1
        idx2 -= 1

    return new_child


def swap_mut(child: Individual, cvrp=None) -> Individual:
    """
    Swaps two randomly chosen genes.

    :param child: The child Individual object
    :param cvrp: An instance of the CVRP class (unused)
    :return: A new mutated Individual
    """
    new_child = Individual(child[:], None)
    idx1 = r.randint(0, len(new_child) - 1)
    idx2 = r.randint(0, len(new_child) - 1)
    _swap(new_child, idx1, idx2)

    return new_child


def gvr_scramble_mut(child: Individual, cvrp=None) -> Individual:
    """
    Scramble mutation using GVR representation. Finds the longest route and
    shuffles it. Requires at least 4 elements in the route to be effective.

    :param child: The child Individual object
    :param cvrp: An instance of the CVRP class
    :return: The mutated child as a new Individual
    """
    ind1_partitioned = cvrp.partition_routes(child)

    max_list = 1
    cur_max = len(ind1_partitioned[max_list])
    for route_num in ind1_partitioned.keys():
        if len(ind1_partitioned[route_num]) > cur_max:
            max_list = route_num
            cur_max = len(ind1_partitioned[route_num])

    r.shuffle(ind1_partitioned[max_list])
    new_genes = cvrp.de_partition_routes(ind1_partitioned)

    return Individual(new_genes, cvrp.calc_fitness(new_genes))


def pmx_xo(ind1: Individual, ind2: Individual, cvrp=None) -> Tuple[Individual, Individual]:
    """
    Partially Mapped Crossover (PMX) by Goldberg and Lingle.

    Reference:
    Goldberg, D.E. and Lingle, R. (1985). Proceedings of the International
    Conference on Genetic Algorithms, pp. 154-159.

    Two crossover points are selected. The segment between them is copied from
    the respective parent. Remaining positions are filled via a mapping from the
    exchanged segments.

    :param ind1: The first Individual
    :param ind2: The second Individual
    :param cvrp: An instance of the CVRP class (unused)
    :return: A tuple of two Individuals with recombinant genes (fitness not yet evaluated).
    """
    size = len(ind1)
    cxp1 = r.randint(0, size - 2)
    cxp2 = r.randint(cxp1 + 1, size - 1)

    child1_genes = [None] * size
    child2_genes = [None] * size

    # Copy segment from respective parents
    for i in range(cxp1, cxp2 + 1):
        child1_genes[i] = ind1[i]
        child2_genes[i] = ind2[i]

    # Build mapping from the exchanged segments
    def _fill_remaining(child_genes, source_parent, other_parent):
        # O(1) lookup: gene -> index in source_parent's segment
        seg_pos = {source_parent[j]: j for j in range(cxp1, cxp2 + 1)}
        child_set = set(g for g in child_genes if g is not None)
        for i in range(size):
            if child_genes[i] is not None:
                continue
            gene = other_parent[i]
            while gene in child_set:
                gene = other_parent[seg_pos[gene]]
            child_genes[i] = gene
            child_set.add(gene)

    _fill_remaining(child1_genes, ind1, ind2)
    _fill_remaining(child2_genes, ind2, ind1)

    return Individual(child1_genes, None), Individual(child2_genes, None)


def scramble_mut(child: Individual, cvrp=None) -> Individual:
    """
    Selects a random subsequence and shuffles it.

    :param child: The child Individual object
    :param cvrp: An instance of the CVRP class (unused)
    :return: A new mutated Individual
    """
    new_genes = child[:]
    size = len(new_genes)
    idx1 = r.randint(0, size - 2)
    idx2 = r.randint(idx1 + 1, size - 1)

    segment = new_genes[idx1:idx2 + 1]
    r.shuffle(segment)
    new_genes[idx1:idx2 + 1] = segment

    return Individual(new_genes, None)


def two_opt_mut(child: Individual, cvrp=None) -> Individual:
    """
    2-opt local search within each GVR route. Iteratively reverses segments
    that reduce total distance until no improvement is found.

    Reference: Croes, G. A. (1958). Operations Research, 6(6), 791-812.

    :param child: The child Individual object
    :param cvrp: An instance of the CVRP class
    :return: A new improved Individual
    """
    partitioned = cvrp.partition_routes(child)
    depot = cvrp.depot

    for route_num in partitioned:
        route = partitioned[route_num]
        n = len(route)
        if n < 4:
            continue

        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    # O(1) delta evaluation: compare removed vs added edges
                    before_i = depot if i == 0 else route[i - 1]
                    after_j = depot if j == n - 1 else route[j + 1]

                    d_old = Building.distance(before_i, route[i]) + Building.distance(route[j], after_j)
                    d_new = Building.distance(before_i, route[j]) + Building.distance(route[i], after_j)

                    if d_new < d_old:
                        route[i:j + 1] = route[i:j + 1][::-1]
                        improved = True
                        break
                if improved:
                    break

        partitioned[route_num] = route

    return Individual(cvrp.de_partition_routes(partitioned), None)


def or_opt_mut(child: Individual, cvrp=None) -> Individual:
    """
    Selects a short segment (1-3 genes) and relocates it to a different position.

    Reference: Or, I. (1976). Ph.D. Dissertation, Northwestern University.

    :param child: The child Individual object
    :param cvrp: An instance of the CVRP class (unused)
    :return: A new mutated Individual
    """
    new_genes = child[:]
    size = len(new_genes)
    if size < 4:
        return Individual(new_genes, None)

    seg_len = min(r.choice([1, 2, 3]), size - 2)
    start = r.randint(0, size - seg_len)
    segment = new_genes[start:start + seg_len]

    remaining = new_genes[:start] + new_genes[start + seg_len:]
    insert_pos = r.randint(0, len(remaining))
    new_genes = remaining[:insert_pos] + segment + remaining[insert_pos:]

    return Individual(new_genes, None)


def displacement_mut(child: Individual, cvrp=None) -> Individual:
    """
    Removes a random subsequence and reinserts it at a random position.

    Reference: Michalewicz, Z. (1996). Springer-Verlag, 3rd edition.

    :param child: The child Individual object
    :param cvrp: An instance of the CVRP class (unused)
    :return: A new mutated Individual
    """
    new_genes = child[:]
    size = len(new_genes)
    if size < 3:
        return Individual(new_genes, None)

    idx1 = r.randint(0, size - 2)
    idx2 = r.randint(idx1 + 1, size - 1)

    segment = new_genes[idx1:idx2 + 1]
    remaining = new_genes[:idx1] + new_genes[idx2 + 1:]

    insert_pos = r.randint(0, len(remaining))
    new_genes = remaining[:insert_pos] + segment + remaining[insert_pos:]

    return Individual(new_genes, None)


def _swap(ll, idx1: int, idx2: int) -> None:
    """
    Swaps two positions in pythonic fashion given a list-like object.
    :param ll: The list-like object to swap values from
    :param idx1: The first index
    :param idx2: The second index
    :return: None
    """
    ll[idx1], ll[idx2] = ll[idx2], ll[idx1]
