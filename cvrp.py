import random as r
import numpy
from typing import Dict
from deap import creator, base, tools, algorithms
from util import House, Depot, populate_from_file


class CVRP:
    MAX_CAPACITY = 20

    def __init__(self):
        self.depot = Depot()

    def eval_cvrp(self, individual):
        distance = 0
        partitioned_routes = CVRP.partition_routes(individual)
        # print(partitioned_routes)
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
                routes[route_counter] = []
            
            routes[route_counter].append(house)
            current_weight += house.num_packages

        # print(routes)
        return routes

    @classmethod
    def optimized_cx(cls, ind1: list, ind2: list, mutpb):
        ind1_partitioned = CVRP.partition_routes(ind1)
        ind2_partitioned = CVRP.partition_routes(ind2)

        if True:
            
            # choose a random route from individual 1
            route_number = r.choice(list(ind1_partitioned.keys())) 
            route_from_ind1 = ind1_partitioned[route_number]
            
            # random route chosen to be replaced in individual 2 --> list
            section_from_ind1 = route_from_ind1[r.randint(0, len(route_from_ind1) - 1):] 


            closest_ind2_partition = 0
            closest_ind2_house_index = 0            
            closest_distance = -1
            # print(f'\n\n{ind2_partitioned}\n\n')
            
            for route_num in ind2_partitioned.keys():
                # removing duplicates
                for house in ind2_partitioned[route_num]:
                    if house in section_from_ind1:
                        print(house)
                        ind2_partitioned[route_num].remove(house)
                        print('removing')

            # print(f'\n\n{child}\n\n')
            child = ind2_partitioned
            for route_num in child.keys():
                for i,house in enumerate(child[route_num]):
                    # checks the closest distance from the first number in route to be inserted to route being inserted and places it at that index
                    # print('hi')
                    # print(section_from_ind1)
                    house_ind1 = section_from_ind1[0]
                    distance = House.distance(house_ind1,house)
                        
                    # initial min
                    if closest_distance == -1:
                        closest_distance = distance
                        closest_ind2_house_index = i
                        closest_ind2_partition = route_num
                    elif distance < closest_distance:
                        closest_distance = distance
                        closest_ind2_house_index = i
                        closest_ind2_partition = route_num
            
            print(f'\n\nSECTION: {section_from_ind1}\n\n')
            print(f'\n\n{child}\n\n')
            child[closest_ind2_partition][closest_ind2_house_index:closest_ind2_house_index] = section_from_ind1
            print(f'\n\n{child}\n\n')

            # print(f'PARENT 1: {ind1_partitioned}\n\n')
            # print(f'PARENT 2: {ind2_partitioned}\n\n')
            # print(f'\n\n\n\n{child}\n\n\n\n')
            return child         
                        



houses = populate_from_file("houses.txt")
# print(houses)

# Specifies the objective (minimize or maximize) and the type of the individual
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Specifies the representation of the individual and its inclusion into the population
toolbox = base.Toolbox()
toolbox.register("indices", r.sample, range(len(houses)), len(houses))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

cvrp = CVRP()

toolbox.register("evaluate", cvrp.eval_cvrp)
toolbox.register("mate", cvrp.optimized_cx, mutpb=1)
toolbox.register("mutate", tools.mutGaussian, indpb=0.1, mu=0, sigma=1)
toolbox.register("select", tools.selTournament, tournsize=5)

pop = toolbox.population(n=10)
# print(pop)

cvrp.optimized_cx(pop[0], pop[1],1)

# def main(seed=0):
#     r.seed(seed)

#     pop = toolbox.population(n=10)
#     print(pop)
#     hof = tools.HallOfFame(3)
#     stats = tools.Statistics(lambda ind: ind.fitness.values)
#     stats.register("Avg", numpy.mean)
#     stats.register("Std", numpy.std)
#     stats.register("Min", numpy.min)
#     stats.register("Max", numpy.max)
    
#     algorithms.eaSimple(pop, toolbox,cxpb=.5, mutpb=.05, ngen=100,stats=stats,halloffame=hof,verbose=True)

#     return pop,stats, hof

# if __name__ == '__main__':
#     pop,stats, hof = main()
