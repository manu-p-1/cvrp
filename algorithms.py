"""
https://github.com/manu-p-1/cvrp
algorithms.py

This module contains a list of functions to perform crossover and mutation operations
"""
import math
import random as r
from typing import Dict, Union, List

from util import Building, Individual


def best_route_xo(ind1: Individual, ind2: Individual, cvrp) -> Individual:
    """
    Given two individuals, does best route crossover proposed by Costa et al.

    Reference:
    Costa, E., Machado, P., Pereira, B., Tavares, J.,
    “Crossover and Diversity: A Study about GVR”, In Proceedings of the Analysis and Design of Representations and
    Operators (ADorO’2003) a bird-of-a-feather workshop at (GECCO-2003), Chicago, Illinois, USA, 12-16 July 2003

    The first and second individuals are partitioned into their GVR form. A random genetic fragment is chosen
    from a random route from the donor fragment. It is inserted in the recipient individual next to the node that
    is closest in distance to the first node in the fragment sub-route. Before it is inserted, all duplicates are
    removed to retain the integrity of the permuation.

    :param ind1: The first Individual
    :param ind2: The second Individual
    :param cvrp: An instance of the CVRP class
    :return: An Individual with the recombinant genes with no fitness evaluated yet.
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
        for i, building in enumerate(child[route_num]):
            """
            checks the closest distance from the first number in route to be inserted to route being inserted 
            and places it at that index print('hi') print(section_from_ind1) 
            """
            building_ind1 = section_from_ind1[0]
            distance = Building.distance(building_ind1, building)

            # initial min
            if closest_distance == -1:
                closest_distance = distance
                closest_child_bldg_idx = i
                closest_child_route = route_num
            elif distance < closest_distance:
                closest_distance = distance
                closest_child_bldg_idx = i
                closest_child_route = route_num

    child[closest_child_route][closest_child_bldg_idx + 1:closest_child_bldg_idx + 1] = section_from_ind1

    # Python allows instances to call @staticmethod decorators with no issues
    return Individual(cvrp.de_partition_routes(child), None)


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


def cycle_xo(ind1: Individual, ind2: Individual, cvrp) -> Dict[str, Union[List[None], List[list], List[bool]]]:
    """
    Given two individuals, does an optimized version of cycle crossover proposed by Nazif and Lee.

    Reference:
    Habibeh Nazif and Lai Soon Lee, “Optimised crossover genetic algorithm for capacitated vehicle routing problem”,
    Applied Mathematical Modeling 36, 2011.

    We decide to modify the proposed crossover by finding as many combinations of cycles within a specified runtime,
    instead of 2^c (proposed by authors) or 2^5 (recommendation by authors) combinations. The runtime is
    10 × ln(c + 0.1) + 22 × ln(c) - 19 × ln(c) where c is the cycle length.

    :param ind1: The first Individual
    :param ind2: The second Individual
    :param cvrp: An Individual with the recombinant genes with no fitness evaluated yet.
    :return: An List containing the O-child, E-child, cycles, and raw binaries. The O-child and E-child both have
    their fitness evaluated in the Individual instance, unlike the other crossover operators.
    """
    cl = CycleInfo(ind1, ind2).get_cycle_info()
    p_children = []
    cycle_len = len(cl)

    # Calculates number of iterations to generate combos before quitting. Unique formula to accomplish this
    runtime = math.ceil(10 * math.log(cycle_len + 0.1)
                        + 22 * math.log(cycle_len)
                        - 19 * math.log(cycle_len))

    for i in range(runtime):
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
                    ind1_idx = ind1.index(allele)
                    o_child[ind1_idx] = ind2[ind1_idx]
                    e_child[ind1_idx] = ind1[ind1_idx]
            else:
                # else 1, get from ind1
                for allele in c:
                    ind1_idx = ind1.index(allele)
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
    children = min(p_children, key=lambda pc: pc['o-child'].fitness)
    children["e-child"].fitness = cvrp.calc_fitness(children['e-child'])

    return children


def edge_recomb_xo(ind1: Individual, ind2: Individual) -> Individual:
    """
    Given two individuals, does an edge recombination proposed by Whitley et al.

    Reference:
    Whitley, Darrell; Timothy Starkweather; D'Ann Fuquay (1989). "Scheduling problems and traveling salesman:
    The genetic edge recombination operator". International Conference on Genetic Algorithms. pp. 133–140

    :param ind1: The first Individual
    :param ind2: The second Individual
    :return: An Individual with the recombinant genes with no fitness evaluated yet.
    """
    # May need to enforce size of len(2)
    ind1_adjacent = {}
    ind2_adjacent = {}
    ind_len = len(ind1)

    # Create an adjacency matrix for each node which has a list of its neighbors. This is done
    # for both individuals
    for i in range(ind_len):
        bldg1 = ind1[i]
        bldg2 = ind2[i]

        # Special cases for finding the correct neighbors for each node
        if i == 0:
            ind1_adjacent[bldg1] = [ind1[i + 1]]
            ind1_adjacent[bldg1].append(ind1[ind_len - 1])
            ind2_adjacent[bldg2] = [ind2[i + 1]]
            ind2_adjacent[bldg2].append(ind2[ind_len - 1])
        elif i == ind_len - 1:
            ind1_adjacent[bldg1] = [ind1[i - 1]]
            ind1_adjacent[bldg1].append(ind1[0])
            ind2_adjacent[bldg2] = [ind2[i - 1]]
            ind2_adjacent[bldg2].append(ind1[0])
        else:
            ind1_adjacent[bldg1] = [ind1[i - 1]]
            ind1_adjacent[bldg1].append(ind1[i + 1])
            ind2_adjacent[bldg2] = [ind2[i - 1]]
            ind2_adjacent[bldg2].append(ind2[i + 1])

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


def order_xo(ind1: Individual, ind2: Individual) -> Individual:
    """
    Given two individuals, does an order crossover. Order crossover operates by choosing two indices
    within the first parent and copying everything within those indices to the child. The rest of the
    elements go from the second parent to the child. Duplicate values are ignored.

    :param ind1: The first Individual
    :param ind2: The second Individual
    :return: An Individual with the recombinant genes with no fitness evaluated yet.
    """
    bound = len(ind1)

    cxp1 = r.randint(0, (bound - 1) // 2)
    cxp2 = r.randint(((bound - 1) // 2) + 1, bound - 1)
    child = [None] * bound
    for i in range(cxp1, cxp2 + 1):
        child[i] = ind1[i]

    parent_idx = cxp2 + 1
    child_idx = cxp2 + 1

    """
    The implementation is a bit tricky, but we mark indices and boundaries in order
    to pull the genes from the parent into the child which are not already in the child
    """

    parent_bound = child_bound = bound

    while child_idx != cxp1:
        if parent_idx == parent_bound:
            parent_idx = 0
            parent_bound = cxp2 + 1

        if child_idx == child_bound:
            child_idx = 0
            child_bound = cxp1

        if ind2[parent_idx] not in child:
            child[child_idx] = ind2[parent_idx]
            child_idx += 1
        parent_idx += 1

    return Individual(child, None)


def inversion_mut(child: Individual) -> Individual:
    """
    Mutates a child's genes by choosing two random indices between 0 and len(child) - 1.
    From a programming perspective, two indices are chosen: one between 0 and the list midpoint and one
    between the midpoint and the length of the list. Every value between the two chosen indices are mirrored.
    This way, the values mutate while preserving the permutation.

    :param child: The child Individual object
    :return: The mutated child
    """

    mid = (len(child) // 2) - 1
    idx1 = r.randint(0, mid)
    idx2 = r.randint(mid + 1, len(child) - 1)

    # Swap the values until all values are mirrored
    while idx1 <= idx2:
        _swap(child, idx1, idx2)
        idx1 += 1
        idx2 -= 1

    return child


def swap_mut(child: Individual) -> Individual:
    """
    Mutates a child's genes by choosing two random indices between 0 and len(child) - 1 and swapping
    their elements.
    :param child: The child Individual object
    :return: The mutated child
    """
    idx1 = r.randint(0, len(child) - 1)
    idx2 = r.randint(0, len(child) - 1)
    _swap(child, idx1, idx2)

    return child


def gvr_scramble_mut(child, cvrp):
    """
    A scramble mutation with added heuristics. Put's the child into the GVR representation,
    selects a random route then scrambles every value in that route. This type of mutation
    tends to be aggressive, however, can be useful for diversity maintenance.

    The precondition is that a route must contain at least 4 elements for its mutation to be
    effective. This is by nature of euclidean distance.

    :param child: The child Individual object
    :param cvrp: An instance of the CVRP class
    :return: The mutated child
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


def _swap(ll: Individual, idx1: int, idx2: int) -> None:
    """
    Swaps two positions in pythonic fashion given a list.
    :param ll: The list to swap values from
    :param idx1: The first index
    :param idx2: The second index
    :return: None
    """
    ll[idx1], ll[idx2] = ll[idx2], ll[idx1]
