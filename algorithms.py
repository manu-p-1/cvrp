import math
import random as r

from util import Building


def best_route_xo(ind1: list, ind2: list, cvrp) -> list:
    ind1_partitioned = cvrp.partition_routes(ind1)
    ind2_partitioned = cvrp.partition_routes(ind2)

    # choose a random route from individual 1
    route_number = r.choice(list(ind1_partitioned.keys()))
    route_from_ind1 = ind1_partitioned[route_number]

    # random route chosen to be replaced in individual 2 --> list
    section_from_ind1 = route_from_ind1[r.randint(0, len(route_from_ind1) - 1):]

    for route_num in ind2_partitioned.keys():
        # removing duplicates
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
    return cvrp.de_partition_routes(child)


class CycleInfo:
    """
    CODE ATTRIBUTION

    This class was taken and modified from the attributed author

    ----------------------------------------------------------

    AUTHOR: EVAN CONRAD - https://github.com/Flaque

    TITLE: Python-GA

    YEAR: 2017

    AVAILABILITY:

    https://github.com/Flaque/Python-GA
    https://github.com/Flaque/Python-GA/blob/master/cx.py
    ----------------------------------------------------------
    """

    def __init__(self, father, mother):
        self._mother = mother
        self._father = father

    @staticmethod
    def _map(father, mother):
        return dict(zip(father, mother))

    @staticmethod
    def _find_cycle(start, relation_map):
        cycle = [start]

        current = relation_map[start]
        while current not in cycle:
            cycle.append(current)
            current = relation_map[current]
        return cycle

    def get_cycle_info(self):
        return self._get_cycle_info()

    def _get_cycle_info(self):

        fathers_child = self._father[:]
        mothers_child = self._mother[:]
        relation_map = self._map(fathers_child, mothers_child)

        cycles_list = []
        for i in range(len(fathers_child)):
            cycle = self._find_cycle(fathers_child[i], relation_map)

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


def cycle_xo(ind1, ind2, cvrp):
    cl = CycleInfo(ind1, ind2).get_cycle_info()
    p_children = []
    cycle_len = len(cl)

    # Calculates number of iterations to generate combos before quitting. Unique formula to accomplish this
    runtime = math.ceil(10 * math.log(cycle_len + 0.1)
                        + 22 * math.log(cycle_len)
                        - 19 * math.log(cycle_len))

    for i in range(runtime):
        o_child, e_child = [None] * len(ind1), [None] * len(ind1)

        binaries = [bool(r.getrandbits(1)) for _ in cl]
        all_ = len(set(binaries))
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

        p_children.append({
            "o-child": o_child,
            "e-child": e_child,
            "cycles": cl,
            "binaries": binaries
        })

    # O-Child
    children = min(p_children, key=lambda t: cvrp.calc_fitness(t['o-child']))

    return children


def edge_recomb_xo(ind1, ind2):
    # May need to enforce size of len(2)
    ind1_adjacent = {}
    ind2_adjacent = {}
    ind_len = len(ind1)

    for i in range(ind_len):
        bldg1 = ind1[i]
        bldg2 = ind2[i]

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

    return child


def order_xo(ind1, ind2):
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

    return child


def inversion_mutation(child: list) -> list:
    """
    Mutates a child's genes by choosing two random indices between 0 and len(chromosome) - 1.
    From a programming perspective, two indices are chosen: one between 0 and the list midpoint and one
    between the midpoint and the length of the list. Every value between the two chosen indices are mirrored.
    This way, the values mutate while preserving the permutation.
    :param child: The child as a dictionary object (at this stage only containing a key to the chromosome as a list)
    :return: The mutated child
    """

    mid = (len(child) // 2) - 1
    idx1 = r.randint(0, mid)
    idx2 = r.randint(mid + 1, len(child) - 1)

    while idx1 <= idx2:
        _swap(child, idx1, idx2)
        idx1 += 1
        idx2 -= 1

    return child


def _swap(ll: list, idx1: int, idx2: int) -> None:
    """
    Swaps two positions in pythonic fashion given a list.
    :param ll: The list to swap values from
    :param idx1: The first index
    :param idx2: The second index
    :return: None
    """
    ll[idx1], ll[idx2] = ll[idx2], ll[idx1]
