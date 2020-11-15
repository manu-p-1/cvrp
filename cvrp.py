import random
from typing import Dict
from deap import creator, base, tools
from util import House, Depot, populate_from_file


class CVRP:
    MAX_CAPACITY = 20

    def __init__(self):
        self.depot = Depot()

    def eval_cvrp(self, individual):
        distance = 0
        partitioned_routes = self.partition_routes(individual)

        for _, route in partitioned_routes.items():
            for i in range(len(route) - 1):
                h1, h2 = route[i], route[i + 1]
                distance += House.distance(h1, h2)
            distance += House.distance(route[len(route) - 1], self.depot)

        return distance,

    @classmethod
    def partition_routes(cls, individual: list) -> Dict:
        routes = {}
        current_weight = 0
        route_counter = 1

        for x in individual:
            house = houses[x]
            if route_counter not in routes:
                routes[route_counter] = []

            if current_weight + house.num_packages > cls.MAX_CAPACITY:
                route_counter += 1
                current_weight = 0
            routes[route_counter].append(house)
            current_weight += house.num_packages
        return routes


houses = populate_from_file("houses.txt")
print(houses)

# Specifies the objective (minimize or maximize) and the type of the individual
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Specifies the representation of the individual and its inclusion into the population
toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(len(houses)), len(houses))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

cvrp = CVRP()
toolbox.register("evaluate", cvrp.eval_cvrp)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, indpb=0.1, mu=0, sigma=1)
toolbox.register("select", tools.selTournament, tournsize=5)

pop = toolbox.population(n=10)
print(pop)
