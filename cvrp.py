import enum
import math
import random as r
import time
from typing import Dict, Tuple

import matplotlib.pyplot as plt

import algorithms as alg
from util import Building, Individual


class ReplStrat(enum.Enum):
    RAND = enum.auto()
    BEST = enum.auto()
    WORST = enum.auto()


class CVRP:

    def __init__(self, problem_set: dict,
                 population_size: int = 800,
                 selection_size: int = 5,
                 ngen: int = 100_000,
                 mutpb: float = 0.15,
                 cxpb: float = 0.85,
                 cx_algo=alg.best_route_xo,
                 mt_algo=alg.swap_mut,
                 pgen: bool = False,
                 agen: bool = False,
                 plot: bool = False,
                 verbose_routes: bool = False):

        self.var_len = len(problem_set["BUILDINGS"])
        self.depot = problem_set["DEPOT"]
        self.vehicle_cap = problem_set["CAPACITY"]
        self.optimal_fitness = problem_set["OPTIMAL"]
        self.problem_set_name = problem_set["NAME"]
        self.problem_set_buildings_orig = problem_set['BUILDINGS']

        self.pop = []
        for _ in range(population_size):
            rpmt = r.sample(self.problem_set_buildings_orig, self.var_len)
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
        self.verbose_routes = verbose_routes

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

    def select(self) -> Tuple[Individual, Individual]:
        """
        For selection, five individuals are randomly sampled. Of the five, the two with the best selected
        are chosen to become parents. We employ a form of tournament selection here.
        :return: A tuple containing parent one and parent two
        """

        # take_five is the mating pool for this generation
        take_five = r.sample(self.pop, self.selection_size)

        parent1 = self._get_and_remove(take_five, ReplStrat.BEST)
        parent2 = self._get_and_remove(take_five, ReplStrat.BEST)

        return parent1, parent2

    def replacement_strat(self, individual: Individual, rs) -> None:
        self._get_and_remove(self.pop, rs)
        self.pop.append(individual)

    @staticmethod
    def _get_nworst(sel_values, n):
        v = sel_values[:]
        v.sort()
        return v[-n:]

    @staticmethod
    def _get_and_remove(sel_values, rs):
        """
        Get the largest fitness in the GA and remove it
        :return: The chromosome with the highest fitness
        """
        if rs == rs.RAND:
            val = r.choice(sel_values)
        elif rs == rs.BEST:
            val = min(sel_values)
        else:
            val = max(sel_values)
        sel_values.remove(val)
        return val

    def reset(self):
        self.pop = []
        for _ in range(self.population_size):
            rpmt = r.sample(self.problem_set_buildings_orig, self.var_len)
            self.pop.append(Individual(rpmt, self.calc_fitness(rpmt)))

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

        best_data, avg_data = [], []

        div_thresh_lb = math.ceil(0.01 * self.population_size)
        div_picking_rng = round(0.75 * self.population_size)

        t = time.process_time()
        found = False
        indiv = None

        for i in range(1, self.ngen + 1):

            mut_prob = r.choices([True, False], weights=(self.mutpb, 1 - self.mutpb), k=1)[0]
            cx_prob = r.choices([True, False], weights=(self.cxpb, 1 - self.cxpb), k=1)[0]

            parent1, parent2 = self.select()
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

            self.replacement_strat(child1, ReplStrat.WORST)
            self.replacement_strat(child2, ReplStrat.WORST)

            if self.pgen:
                print(f'GEN: {i}/{self.ngen}', end='\r')

            uq_indv = len(set(self.pop))

            min_indv, max_indv, avg_fit = None, None, None
            if i % 250 == 0 or i == 1:
                if self.agen:
                    min_indv = min(self.pop).fitness
                    max_indv = max(self.pop).fitness
                    avg_fit = round(sum(self.pop) / self.population_size)

                    print(f"UNIQUE FITNESS CNT: {uq_indv}/{self.population_size}")
                    print(f"GEN {i} BEST FITNESS: {min_indv}")
                    print(f"GEN {i} WORST FITNESS: {max_indv}")
                    print(f"GEN {i} AVG FITNESS: {avg_fit}\n\n")

                if self.plot:
                    min_indv = min(self.pop).fitness if min_indv is None else min_indv
                    best_data.append(min_indv)

                    avg_fit = round(sum(self.pop) / self.population_size) if avg_fit is None else avg_fit
                    avg_data.append(avg_fit)

            if i % 10000 == 0 and uq_indv <= div_thresh_lb:
                print("===============DIVERSITY MAINT===============") if self.agen else None
                worst = self._get_nworst(self.pop, div_picking_rng)

                for k in range(div_picking_rng):
                    c = min(self.pop)
                    if self.mt_algo == 'inversion_mut':
                        rsamp = alg.inversion_mut(c)
                    elif self.mt_algo == 'swap_mut':
                        rsamp = alg.swap_mut(c)
                    else:
                        rsamp = alg.gvr_scramble_mut(c, self)
                    i = Individual(rsamp, self.calc_fitness(rsamp))

                    self.pop.remove(worst[k])
                    self.pop.append(i)

        # Find the closest value to the optimal fitness (in case we don't find a solution)
        closest = min(self.pop)
        end = time.process_time() - t

        return self._create_solution(indiv if found else closest, end, best_data, avg_data)

    def _create_solution(self, individual, comp_time, best_data, avg_data) -> dict:
        """
        Creates a dictionary with all of the information about the solution or closest solution
        that was found in the EA.
        :param individual: The chromosome that was matched as the solution or closest solution
        :param comp_time: The computation time of the algorithm
        :return: A dictionary with the information
        """

        if self.plot:
            plt.figure(figsize=(10, 9), dpi=200)
            plt.plot(best_data, linestyle="solid", label="Best Fitness Value")
            plt.plot(avg_data, linestyle="solid", label="Average Fitness Value")
            plt.title(f'{self.cx_algo}_{self.ngen}_{self.selection_size}_{self.cxpb}_{self.mutpb}__graph')
            plt.legend(loc='upper right')
            plt.xlabel("Generations")
            plt.ylabel("Fitness")

        partitioned = self.partition_routes(individual)

        obj = {
            "name": type(self).__name__,
            "problem_set_name": self.problem_set_name,
            "problem_set_optimal": self.optimal_fitness,
            "time": f"{comp_time} seconds",
            "vehicles": len(partitioned.keys()),
            "vehicle_capacity": self.vehicle_cap,
            "dimension": self.var_len,
            "population_size": self.population_size,
            "selection_size": self.selection_size,
            "generations": self.ngen,
            "cxpb": self.cxpb,
            "mutpb": self.mutpb,
            "cx_algorithm": self.cx_algo,
            "mut_algorithm": self.mt_algo,
            "mat_plot": plt,
            "best_individual_fitness": individual.fitness,
        }

        if self.verbose_routes:
            obj["best_individual"] = partitioned

        return obj
