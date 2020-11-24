import argparse
import json

from cvrp import CVRP
from util import BuildingEncoder, parse_file

if __name__ == '__main__':
    pop, sel, ngen, mutpb, cxpb = 200, 5, 50_000, 0.15, 0.75

    parser = argparse.ArgumentParser(description="Runs the CVRP with any of the optional arguments")
    parser.add_argument("-f", "--file", metavar='', type=str, help="the path to the problem set")
    parser.add_argument("-p", "--pop", metavar='', type=int, help="the population size")
    parser.add_argument("-s", "--sel", metavar='', type=int, help="the selection size")
    parser.add_argument("-g", "--ngen", metavar='', type=int, help="the generation size")
    parser.add_argument("-m", "--mutpb", metavar='', type=float, help="the mutation probability")
    parser.add_argument("-c", "--cxpb", metavar='', type=float, help="the crossover probability")
    parser.add_argument("-i", "--indent", metavar='', type=int, help="the indentation amount of the result string")
    parser.add_argument("-P", "--pgen", action='store_true', help="prints the current generation")
    args = parser.parse_args()

    p_set = parse_file(args.file) if args.file else parse_file("data/A-n54-k7.ocvrp")
    pop = args.pop if args.pop else pop
    sel = args.sel if args.sel else sel
    ngen = args.ngen if args.ngen else ngen
    mutpb = args.mutpb if args.mutpb else mutpb
    cxpb = args.cxpb if args.cxpb else cxpb

    cvrp = CVRP(problem_set=p_set,
                population_size=pop,
                selection_size=sel,
                ngen=ngen,
                mutpb=mutpb,
                cxpb=cxpb,
                pgen=args.pgen if args.pgen else False)

    result = cvrp.run()

    print(json.dumps(obj=result,
                     cls=BuildingEncoder,
                     indent=None if args.indent == 0 else args.indent))
