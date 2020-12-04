import random as r
import time
from typing import Dict, Tuple

import matplotlib.pyplot as plt

import algorithms as alg
from util import Building, Individual


class CVRP:
    DIV_THRESH_LB = 5
    DIV_THRESH_UP = 40

    def __init__(self, problem_set: dict,
                 population_size: int,
                 selection_size: int,
                 ngen: int,
                 mutpb: float,
                 cxpb: float,
                 pgen: bool,
                 agen: bool,
                 cx_algo,
                 mt_algo,
                 plot=False):

        var_len = len(problem_set["BUILDINGS"])
        self.depot = problem_set["DEPOT"]
        self.vehicle_cap = problem_set["CAPACITY"]
        self.optimal_fitness = problem_set["OPTIMAL"]
        self.problem_set_name = problem_set["NAME"]

        self.pop = []
        for _ in range(population_size):
            rpmt = r.sample(problem_set['BUILDINGS'], var_len)
            self.pop.append(Individual(rpmt, self.calc_fitness(rpmt)))

        self.population_size = population_size
        self.selection_size = selection_size
        self.ngen = ngen
        self.mutpb = mutpb
        self.cxpb = cxpb
        self.cx_algo = cx_algo.__name__
        self.mt_algo = mt_algo.__name__
        self.pgen = pgen
        self.agen = agen
        self.plot = plot

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

    def partition_routes(self, individual: Individual) -> Dict:
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
    def de_partition_routes(partitioned_routes: Dict):
        ll = []
        # This length should only be one
        for v in partitioned_routes.values():
            ll.extend(v)
        return ll

    def select(self, bad) -> Tuple[Individual, Individual]:
        """
        For selection, five individuals are randomly sampled. Of the five, the two with the best selected
        are chosen to become parents. We employ a form of tournament selection here.
        :return: A tuple containing parent one and parent two
        """

        # take_five is the mating pool for this generation
        if not bad:
            take_five = r.sample(self.pop, self.selection_size)
        else:
            i = r.choice(self.pop)
            take_five = [alg.gvr_scramble_mut(i, self) for _ in range(self.selection_size)]

        parent1 = CVRP._get_and_remove(take_five, True)
        parent2 = CVRP._get_and_remove(take_five, True)

        return parent1, parent2

    def replacement_strat(self, individual: Individual) -> None:
        """
        Replaces the two worst individuals in the population with two new ones.
        :param individual: The new individual to replace a candidate in the population
        :return: None
        """
        self._get_and_remove(self.pop, False)
        self.pop.append(individual)

    def brute_strat(self, individual: Individual) -> None:
        self.pop.remove(r.choice(self.pop))
        self.pop.append(individual)

    @staticmethod
    def _get_and_remove(sel_values, get_best):
        """
        Get the largest fitness in the GA and remove it
        :return: The chromosome with the highest fitness
        """
        if get_best:
            val = min(sel_values)
        else:
            val = max(sel_values)
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

        orig_mutpb = self.mutpb
        bad = False

        t = time.process_time()
        found = False
        indiv = None

        worst_data, best_data, avg_data = [], [], []

        for i in range(1, self.ngen + 1):

            mut_prob = r.choices([True, False], weights=(self.mutpb, 1 - self.mutpb), k=1)[0]
            cx_prob = r.choices([True, False], weights=(self.cxpb, 1 - self.cxpb), k=1)[0]

            parent1, parent2 = self.select(bad)
            if cx_prob:
                if self.cx_algo == 'best_route_xo':
                    child1 = alg.best_route_xo(parent1, parent2, self)
                    child2 = alg.best_route_xo(parent2, parent1, self)
                elif self.cx_algo == 'cycle_xo':
                    cxo = alg.cycle_xo(parent1, parent2, self)
                    child1 = cxo['o-child']
                    child2 = cxo['e-child']
                elif self.cx_algo == 'edge_recomb_xo':
                    child1 = alg.edge_recomb_xo(parent1, parent2)
                    child2 = alg.edge_recomb_xo(parent2, parent1)
                else:
                    child1 = alg.order_xo(parent1, parent2)
                    child2 = alg.order_xo(parent2, parent1)
            else:
                child1 = parent1
                child2 = parent2

            if mut_prob:
                if self.mt_algo == 'inversion_mut':
                    child1 = alg.inversion_mut(child1)
                    child2 = alg.inversion_mut(child2)
                elif self.mt_algo == 'swap_mut':
                    child1 = alg.swap_mut(child1)
                    child2 = alg.swap_mut(child2)
                else:
                    child1 = alg.gvr_scramble_mut(child1, self)
                    child2 = alg.gvr_scramble_mut(child2, self)

            """
            Only calculate fitness if a crossover or mutation occurred, or if the xover did not
            assign a fitness value. E.g. Cycle XO assigns mandates ranking fitness, so we don't
            need to calculate again.
            """
            if child1.fitness is None:
                child1.fitness = self.calc_fitness(child1)

            if child2.fitness is None:
                child2.fitness = self.calc_fitness(child2)

            # One of the children were found to have an optimal fitness, so I'll save that
            if child1.fitness == self.optimal_fitness or child2.fitness == self.optimal_fitness:
                indiv = child1 if child1.fitness == self.optimal_fitness else child2
                found = True
                break

            self.replacement_strat(child1)
            self.replacement_strat(child2)

            if self.pgen:
                print(f'GEN: {i}/{self.ngen}', end='\r')

            min_indv, max_indv, uq_indv = None, None, len(set(self.pop))

            if i % 1000 == 0 or i == 1:
                if self.agen:
                    min_indv = min(self.pop).fitness
                    max_indv = max(self.pop).fitness

                    print(f"UNIQUE FITNESSES: {len(set(self.pop))}/{self.population_size}")
                    print(f"GEN {i} BEST FITNESS: {min_indv}")
                    print(f"GEN {i} WORST FITNESS: {max_indv}\n\n")

                if self.plot:
                    min_indv = min(self.pop).fitness if min_indv is None else min_indv
                    max_indv = max(self.pop).fitness if max_indv is None else max_indv
                    best_data.append(min_indv)
                    worst_data.append(max_indv)

                    average_val = round(sum(self.pop) / self.population_size)
                    avg_data.append(average_val)

            if uq_indv <= CVRP.DIV_THRESH_LB:
                self.mutpb = 1
                bad = True
            elif uq_indv >= CVRP.DIV_THRESH_UP:
                bad = False
                self.mutpb = orig_mutpb

        # Find the closest value to the optimal fitness (in case we don't find a solution)
        closest = min(self.pop)
        end = time.process_time() - t

        return self._create_solution(indiv if found else closest, end, best_data, worst_data, avg_data)

    def _create_solution(self, individual, comp_time, best_data, worst_data, avg_data) -> dict:
        """
        Creates a dictionary with all of the information about the solution or closest solution
        that was found in the EA.
        :param individual: The chromosome that was matched as the solution or closest solution
        :param comp_time: The computation time of the algorithm
        :return: A dictionary with the information
        """

        if self.plot:
            plt.figure(figsize=(9, 7), dpi=200)
            plt.plot(worst_data, linestyle="dotted", label="Worst Fitness Values")
            plt.plot(best_data, linestyle="dotted", label="Best Fitness Values")
            plt.plot(avg_data, linestyle="dotted", label="Average Fitness Values")
            plt.title(f'{self.cx_algo}_{self.ngen}_{self.selection_size}_{self.cxpb}_{self.mutpb}__graph')
            plt.legend(bbox_to_anchor=(0.98, 0.98), borderaxespad=0)
            plt.xlabel("Generations")
            plt.ylabel("Fitness")

        partitioned = self.partition_routes(individual)

        return {
            "best_individual": partitioned,
            "best_individual_fitness": individual.fitness,
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
            "cxpb": self.cxpb,
            "mutpb": self.mutpb,
            "cx_algorithm": self.cx_algo,
            "mut_algorithm": self.mt_algo,
            "mat_plot": plt
        }
