import argparse
import json

from cvrp import CVRP
from util import BuildingEncoder, parse_file

if __name__ == '__main__':
    pop, sel, ngen, mutpb, cxpb = 200, 5, 50_000, 0.15, 0.75

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", metavar='', type=str, help="The path to the problem set")
    parser.add_argument("-p", "--pop", metavar='', type=int, help="The population size")
    parser.add_argument("-s", "--sel", metavar='', type=int, help="The selection size")
    parser.add_argument("-g", "--ngen", metavar='', type=int, help="The generation size")
    parser.add_argument("-m", "--mutpb", metavar='', type=float, help="The mutation probability")
    parser.add_argument("-c", "--cxpb", metavar='', type=float, help="The crossover probability")
    args = parser.parse_args()

    p_set = parse_file("data/A-n54-k7.ocvrp") if args.file is None else parse_file(args.file)
    pop = pop if args.pop is None else args.pop
    sel = sel if args.sel is None else args.sel
    ngen = ngen if args.ngen is None else args.ngen
    mutpb = mutpb if args.mutpb is None else args.mutpb
    cxpb = cxpb if args.cxpb is None else args.cxpb

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
