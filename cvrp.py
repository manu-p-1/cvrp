import random as r
import time
from typing import Dict, Tuple

from util import Building


class CVRP:

    def __init__(self, problem_set: dict,
                 population_size: int,
                 selection_size: int,
                 ngen: int,
                 mutpb: float,
                 cxpb: float,
                 pgen: bool,
                 maximize_fitness: bool = False):

        var_len = len(problem_set["BUILDINGS"])
        self.pop = [r.sample(problem_set["BUILDINGS"], var_len) for _ in range(population_size)]
        self.depot = problem_set["DEPOT"]
        self.vehicle_cap = problem_set["CAPACITY"]
        self.optimal_fitness = problem_set["OPTIMAL"]
        self.problem_set_name = problem_set["NAME"]
        self.population_size = population_size
        self.selection_size = selection_size
        self.ngen = ngen
        self.mutpb = mutpb
        self.cxpb = cxpb
        self.pgen = pgen
        self.maximize_fitness = maximize_fitness

    def calc_fitness(self, individual):
        distance = 0
        partitioned_routes = self.partition_routes(individual)
        for _, route in partitioned_routes.items():
            for i in range(len(route) - 1):
                h1, h2 = route[i], route[i + 1]
                distance += Building.distance(h1, h2)
            distance += Building.distance(self.depot, route[0])
            distance += Building.distance(route[len(route) - 1], self.depot)

        return distance

    def partition_routes(self, individual: list) -> Dict:
        routes = {}
        current_weight = 0
        route_counter = 1

        for building in individual:
            if route_counter not in routes:
                routes[route_counter] = []

            if current_weight + building.quant > self.vehicle_cap:
                route_counter += 1
                current_weight = 0
                routes[route_counter] = []

            routes[route_counter].append(building)
            current_weight += building.quant

        return routes

    @classmethod
    def de_partition_routes(cls, individual: Dict):
        ll = []

        # This length should only be one
        for _, j in individual.items():
            ll.extend(j)
        return ll

    def select(self) -> Tuple[list, list]:
        """
        For selection, five individuals are randomly sampled. Of the five, the two with the best selected
        are chosen to become parents. We employ a form of tournament selection here.
        :return: A tuple containing parent one and parent two
        """

        # take_five is the mating pool for this generation
        take_five = r.sample(self.pop, self.selection_size)
        parent1 = self._get_value_and_remove(take_five, self.maximize_fitness)
        parent2 = self._get_value_and_remove(take_five, self.maximize_fitness)
        return parent1, parent2

    def optimized_xo(self, ind1: list, ind2: list) -> list:
        ind1_partitioned = self.partition_routes(ind1)
        ind2_partitioned = self.partition_routes(ind2)

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
        return CVRP.de_partition_routes(child)

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

    def cycle_xo(self, ind1, ind2):
        cl = self.CycleInfo(ind1, ind2).get_cycle_info()
        possible_children = []

        for i in range(15):
            o_child_opt, e_child_opt = [None] * len(ind1), [None] * len(ind1)
            binaries = [bool(r.getrandbits(1)) for _ in cl]

            all_ = len(set(binaries))
            if all_ == 1:
                ri = r.randint(0, len(binaries) - 1)
                binaries[ri] = not binaries[ri]

            bin_counter = 0
            for c in cl:
                if not binaries[bin_counter]:
                    # if 0, get from ind2
                    for allele in c:
                        ind1_idx = ind1.index(allele)
                        o_child_opt[ind1_idx] = ind2[ind1_idx]
                        e_child_opt[ind1_idx] = ind1[ind1_idx]
                else:
                    # else 1, get from ind1
                    for allele in c:
                        ind1_idx = ind1.index(allele)
                        o_child_opt[ind1_idx] = allele
                        e_child_opt[ind1_idx] = ind2[ind1_idx]
                bin_counter += 1

            possible_children.append({
                "o-child": o_child_opt,
                "e-child": e_child_opt,
                "cycles": cl,
                "binaries": binaries
            })

        # O-Child
        min_building = min(possible_children, key=lambda t: self.calc_fitness(t['o-child']))

        return min_building

    @classmethod
    def inversion_mutation(cls, child: list) -> list:
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
            CVRP._swap(child, idx1, idx2)
            idx1 += 1
            idx2 -= 1

        return child

    @staticmethod
    def _swap(ll: list, idx1: int, idx2: int) -> None:
        """
        Swaps two positions in pythonic fashion given a list.
        :param ll: The list to swap values from
        :param idx1: The first index
        :param idx2: The second index
        :return: None
        """
        ll[idx1], ll[idx2] = ll[idx2], ll[idx1]

    def replacement_strat(self, new_indiv: list) -> None:
        """
        Replaces the two worst individuals in the population with two new ones.
        :param new_indiv: The new individual to replace a candidate in the population
        :return: None
        """
        self._get_value_and_remove(self.pop, not self.maximize_fitness)
        self.pop.append(new_indiv)

    def _get_value_and_remove(self, sel_values: list, maximized: bool):
        """
        A helper method to get and remove an individual based on highest fitness level. If the fitness is being
        maximized, the lower fitness is removed and vice versa.
        :param sel_values: A list of buildings
        :param maximized: A flag indicating whether the fitness should be maximized or minimized
        :return: The individual with the highest fitness
        """

        if maximized:
            val = max(sel_values, key=lambda indiv: self.calc_fitness(indiv))
        else:
            val = min(sel_values, key=lambda indiv: self.calc_fitness(indiv))
        sel_values.remove(val)
        return val

    def run(self) -> dict:
        """
        Runs the CVRP in the following stages:
        1.) Parent Selection
        2.) Parent Recombination/Crossover
        3.) Child Mutation
        4.) Fitness Calculation
        5.) Survivor replacement
        :return: A potential solution if found or the closest optimal solution otherwise
        """
        print(f"Running {self.ngen} generations(s)...")
        t = time.process_time()
        found = False
        indiv = None

        mut_prob = r.choices([True, False], weights=(self.mutpb, 1 - self.mutpb), k=1)[0]
        cx_prob = r.choices([True, False], weights=(self.cxpb, 1 - self.cxpb), k=1)[0]

        for i in range(self.ngen):

            if self.pgen:
                print(f'{i}/{self.ngen}', end='\r')

            parent1, parent2 = self.select()
            child1 = self.optimized_xo(parent1, parent2) if cx_prob else parent1
            child1 = CVRP.inversion_mutation(child1) if mut_prob else child1
            child1_fit = self.calc_fitness(child1)

            if self.optimal_fitness is not None:
                # One of the children were found to have an optimal fitness, so I'll save that
                if child1_fit == self.optimal_fitness:
                    indiv = child1
                    found = True
                    break

            self.replacement_strat(child1)

        # Find the closest value to the optimal fitness (in case we don't find a solution)
        closest = self._get_value_and_remove(self.pop, self.maximize_fitness)
        end = time.process_time() - t

        return self._create_solution(indiv if found else closest, end)

    def _create_solution(self, individual, comp_time) -> dict:
        """
        Creates a dictionary with all of the information about the solution or closest solution
        that was found in the EA.
        :param individual: The individual that was matched as the solution or closest solution
        :param comp_time: The computation time of the algorithm
        :return: A dictionary with the information
        """
        partitioned = self.partition_routes(individual)
        return {
            "best_individual": partitioned,
            "best_individual_fitness": self.calc_fitness(individual),
            "name": type(self).__name__,
            "problem_set_name": self.problem_set_name,
            "problem_set_optimal": self.optimal_fitness,
            "time": f"{comp_time} seconds",
            "vehicles": len(partitioned.keys()),
            "vehicle_capacity": self.vehicle_cap,
            "dimension": len(individual),
            "population_size": self.population_size,
            "selection_size": self.selection_size,
            "generations": self.ngen,
            "mutpb": self.mutpb,
            "cxpb": self.cxpb,
        }
