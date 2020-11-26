import random as r
import time
from typing import Dict, Tuple

import algorithms
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

    @staticmethod
    def de_partition_routes(individual: Dict):
        ll = []
        # This length should only be one
        for v in individual.values():
            ll.extend(v)
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
        print(f"Running {self.ngen} generation(s)...")
        t = time.process_time()
        found = False
        indiv = None

        mut_prob = r.choices([True, False], weights=(self.mutpb, 1 - self.mutpb), k=1)[0]
        cx_prob = r.choices([True, False], weights=(self.cxpb, 1 - self.cxpb), k=1)[0]

        for i in range(self.ngen):

            if self.pgen:
                print(f'{i}/{self.ngen}', end='\r')

            parent1, parent2 = self.select()

            child1 = algorithms.best_route_xo(parent1, parent2, self) if cx_prob else parent1
            # child1 = algorithms.cycle_xo(parent1, parent2, self)['o-child'] if cx_prob else parent1
            # child1 = algorithms.edge_recomb_xo(parent1, parent2) if cx_prob else parent1
            # child1 = algorithms.order_xo(parent1, parent2) if cx_prob else parent1

            child1 = algorithms.inversion_mutation(child1) if mut_prob else child1
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
