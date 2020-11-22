import json
import random
import random as r
import time
from typing import Dict, List, Tuple, Union
from util import Building, populate_from_file

class CVRP:
    MAX_CAPACITY = 100

    def __init__(self, building_lst: List[list],
                 optimal_fitness: Union[int, None],
                 selection_size: int,
                 ngen: int,
                 mutpb: float,
                 cxpb: float,
                 maximize_fitness: bool = False):

        var_len = len(building_lst)
        self.pop = [random.sample(building_lst, var_len) for _ in range(var_len)]
        self.selection_size = selection_size
        self.optimal_fitness = optimal_fitness
        self.ngen = ngen
        self.mutpb = mutpb
        self.cxpb = cxpb
        self.maximize_fitness = maximize_fitness

        self.depot = Building("DEPOT", 1, -1, 0)

    def calc_fitness(self, individual):
        distance = 0
        partitioned_routes = CVRP.partition_routes(individual)
        for _, route in partitioned_routes.items():
            for i in range(len(route) - 1):
                h1, h2 = route[i], route[i + 1]
                distance += Building.distance(h1, h2)
            distance += Building.distance(route[len(route) - 1], self.depot)

        return distance

    @classmethod
    def partition_routes(cls, individual: list) -> Dict:
        routes = {}
        current_weight = 0
        route_counter = 1

        for building in individual:
            if route_counter not in routes:
                routes[route_counter] = []

            if current_weight + building.quant > cls.MAX_CAPACITY:
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
        take_five = random.sample(self.pop, self.selection_size)
        parent1 = self._get_value_and_remove(take_five, self.maximize_fitness)
        parent2 = self._get_value_and_remove(take_five, self.maximize_fitness)
        return parent1, parent2

    @classmethod
    def optimized_cx(cls, ind1: list, ind2: list) -> list:
        ind1_partitioned = CVRP.partition_routes(ind1)
        ind2_partitioned = CVRP.partition_routes(ind2)

        # choose a random route from individual 1
        route_number = r.choice(list(ind1_partitioned.keys()))
        route_from_ind1 = ind1_partitioned[route_number]

        # random route chosen to be replaced in individual 2 --> list
        section_from_ind1 = route_from_ind1[r.randint(0, len(route_from_ind1) - 1):]

        closest_ind2_partition = 0
        closest_ind2_building_index = 0
        closest_distance = -1

        for route_num in ind2_partitioned.keys():
            # removing duplicates
            for building in ind2_partitioned[route_num][:]:
                if building in section_from_ind1:
                    ind2_partitioned[route_num].remove(building)

        child = ind2_partitioned
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
                    closest_ind2_building_index = i
                    closest_ind2_partition = route_num
                elif distance < closest_distance:
                    closest_distance = distance
                    closest_ind2_building_index = i
                    closest_ind2_partition = route_num

        child[closest_ind2_partition][
        closest_ind2_building_index + 1:closest_ind2_building_index + 1] = section_from_ind1
        return CVRP.de_partition_routes(child)

    @classmethod
    def inversion_mutation(cls, child: list) -> list:
        """
        Mutates a child's genes by choosing two random indices between 0 and len(chromosome) - 1. From a programming
        perspective, two indices are chosen: one between 0 and the list midpoint and one between the midpoint and the
        length of the list. Every value between the two chosen indices are mirrored. This way, the values mutate
        while preserving the permutation.
        :param child: The child as a dictionary object (at this stage only containing a key to the chromosome as a list)
        :return: The mutated child
        """

        mid = (len(child) // 2) - 1
        idx1 = random.randint(0, mid)
        idx2 = random.randint(mid + 1, len(child) - 1)

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
        print("Running...")
        t = time.process_time()
        found = False
        indiv = None

        mut_prob = random.choices([True, False], weights=(self.mutpb, 1 - self.mutpb), k=1)[0]
        cx_prob = random.choices([True, False], weights=(self.cxpb, 1 - self.cxpb), k=1)[0]

        for i in range(self.ngen):

            parent1, parent2 = self.select()
            child1 = CVRP.optimized_cx(parent1, parent2) if cx_prob else parent1
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
            "name": type(self).__name__,
            "time": f"{comp_time} seconds",
            "best_individual": partitioned,
            "best_individual_fitness": self.calc_fitness(individual),
            "vehicles": len(partitioned.keys()),
            "dimension": len(individual),
            "selection_size": self.selection_size,
            "generations": self.ngen,
            "mutpb": self.mutpb,
            "cxpb": self.cxpb,
        }


if __name__ == '__main__':
    buildings = populate_from_file("buildings.txt")
    cvrp = CVRP(building_lst=buildings,
                optimal_fitness=None,
                selection_size=5,
                ngen=50000,
                mutpb=0.15,
                cxpb=0.75)

    result = cvrp.run()
    print(json.dumps(obj=result,
                     default=lambda o: o.__dict__,
                     indent=2))
