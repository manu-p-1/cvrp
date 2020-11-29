import argparse
import json

import algorithms
from cvrp import CVRP
from util import BuildingEncoder, parse_file


def pos_float(value):
    try:
        value = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError('Value must be numerical')

    if value < 0 or value > 1:
        raise argparse.ArgumentTypeError('Value must be >= 0 and <= 1')

    return value


def pos_int(value):
    try:
        value = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError('Value must be an integer')

    if value < 0:
        raise argparse.ArgumentTypeError('Value must be >= 0')
    return value


def int_ge_one(value):
    try:
        value = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError('Value must be an integer')

    if value <= 0:
        raise argparse.ArgumentTypeError('Value must be >= 1')
    return value


if __name__ == '__main__':
    pop = 200
    sel = 5
    ngen = 50_000
    mutpb = 0.15
    cxpb = 0.75
    offspring = 1
    cx_algo = algorithms.best_route_xo
    mt_algo = algorithms.inversion_mutation

    parser = argparse.ArgumentParser(description="Runs the CVRP with any of the optional arguments")
    parser.add_argument("-f", "--file", metavar='', type=str, help="the path to the problem set")
    parser.add_argument("-p", "--pop", metavar='', type=pos_int, help="the population size")
    parser.add_argument("-s", "--sel", metavar='', type=pos_int, help="the selection size")
    parser.add_argument("-g", "--ngen", metavar='', type=pos_int, help="the generation size")
    parser.add_argument("-m", "--mutpb", metavar='', type=pos_float, help="the mutation probability")
    parser.add_argument("-c", "--cxpb", metavar='', type=pos_float, help="the crossover probability")

    parser.add_argument("-r", "--run", metavar='', type=int_ge_one, help="the number of times to run the problem")
    parser.add_argument("-o", "--offspring", metavar='', type=int_ge_one, help="the number of offspring to generate")

    cx_types = parser.add_mutually_exclusive_group()
    cx_types.add_argument("--brxo", action='store_true', help="use best route crossover")
    cx_types.add_argument("--cxo", action='store_true', help="use cycle crossover")
    cx_types.add_argument("--erxo", action='store_true', help="use edge recombination crossover")
    cx_types.add_argument("--oxo", action='store_true', help="use order crossover")

    parser.add_argument("-i", "--indent", metavar='', nargs="?", type=int_ge_one, const=2,
                        help="the indentation amount of the result string")
    parser.add_argument("-P", "--pgen", action='store_true', help="prints the current generation")
    parser.add_argument("-A", "--agen", action='store_true', help="prints the average fitness every 1000 generations")
    args = parser.parse_args()

    p_set = parse_file(args.file) if args.file else parse_file("data/A-n54-k7.ocvrp")
    pop = args.pop if args.pop else pop
    sel = args.sel if args.sel else sel
    ngen = args.ngen if args.ngen else ngen
    mutpb = args.mutpb if args.mutpb else mutpb
    cxpb = args.cxpb if args.cxpb else cxpb

    offspring = args.offspring if args.offspring else offspring
    runtime = args.run if args.run else 1

    if args.cxo:
        cx_algo = algorithms.cycle_xo
    elif args.erxo:
        cx_algo = algorithms.edge_recomb_xo
    elif args.oxo:
        cx_algo = algorithms.order_xo

    cvrp = CVRP(problem_set=p_set,
                population_size=pop,
                selection_size=sel,
                ngen=ngen,
                mutpb=mutpb,
                cxpb=cxpb,
                num_offspring=offspring,
                cx_algo=cx_algo,
                mt_algo=mt_algo,
                pgen=args.pgen,
                agen=args.agen)

    for i in range(runtime):
        result = cvrp.run()

        print(json.dumps(obj=result,
                         cls=BuildingEncoder,
                         indent=args.indent))
