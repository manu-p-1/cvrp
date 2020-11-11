import json
import math
import random
import util
from deap import creator, base, tools, algorithms

houses = util.populate_from_file("houses.txt")
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
    # Temporary
    return individual.x * 2


toolbox.register("evaluate", eval_cvrp)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, indpb=0.1, mu=0, sigma=1)
toolbox.register("select", tools.selTournament, tournsize=5)

pop = toolbox.population(n=10)
