import argparse
import json
import sys

from cvrp import CVRP
from util import populate_from_file, BuildingEncoder

if __name__ == '__main__':

    pop = 200
    sel = 5
    ngen = 50_000
    mutpb = 0.15
    cxpb = 0.75

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", action='store_true', help="Use test set")
    parser.add_argument("-f", "--file", metavar='', type=str, help="The path to the problem set")
    parser.add_argument("-p", "--pop", metavar='', type=int, help="The population size")
    parser.add_argument("-s", "--sel", metavar='', type=int, help="The selection size")
    parser.add_argument("-g", "--gen", metavar='', type=int, help="The generation size")
    parser.add_argument("-m", "--mutpb", metavar='', type=float, help="The mutation probability")
    parser.add_argument("-c", "--cxpb", metavar='', type=float, help="The crossover probability")
    args = parser.parse_args()

    if args.test:
        p_set = populate_from_file("problems_sets/test_set.txt")
    else:
        if args.file:
            p_set = populate_from_file(sys.argv[1])
        else:
            p_set = populate_from_file("problems_sets/A-n54-k7.txt")

    pop = args.pop if args.pop is not None else pop
    sel = args.sel if args.sel is not None else sel
    ngen = args.gen if args.gen is not None else ngen
    mutpb = args.mutpb if args.mutpb is not None else mutpb
    cxpb = args.cxpb if args.cxpb is not None else cxpb

    cvrp = CVRP(problem_set=p_set,
                population_size=pop,
                selection_size=sel,
                ngen=ngen,
                mutpb=mutpb,
                cxpb=cxpb)

    result = cvrp.run()
    print(json.dumps(obj=result,
                     cls=BuildingEncoder,
                     indent=2))
