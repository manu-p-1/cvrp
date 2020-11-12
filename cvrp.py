import random

from deap import creator, base, tools

from util import House, Depot, populate_from_file

MAX_CAP = 20
depot = Depot()

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


def eval_cvrp(individual):
    current_weight = 0
    routes = {}
    route_counter = 1
    distance = 0

    for x in individual:

        house = houses[x]
        if route_counter not in routes:
            routes[route_counter] = []

        if current_weight + house.num_packages > MAX_CAP:
            route_counter += 1
            current_weight = 0
        routes[route_counter].append(house)
        current_weight += house.num_packages
    
    for route_number, route in routes.items():
        for i in range(0, len(route) - 1):
            h1 = route[i]
            h2 = route[i + 1]
            distance += House.distance(h1, h2)

        distance += House.distance(route[len(route) - 1], depot)
    return distance,
    
toolbox.register("evaluate", eval_cvrp)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, indpb=0.1, mu=0, sigma=1)
toolbox.register("select", tools.selTournament, tournsize=5)

pop = toolbox.population(n=10)
