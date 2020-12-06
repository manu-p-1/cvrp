import argparse
import datetime
import json
import os
import sys

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


def main():
    pop = 800
    sel = 5
    ngen = 100_000
    mutpb = 0.15
    cxpb = 0.85
    cx_algo = algorithms.best_route_xo
    mt_algo = algorithms.inversion_mut

    parser = argparse.ArgumentParser(description="Runs the CVRP with any of the optional arguments")
    parser.add_argument("-f", "--file", metavar='', type=str, help="the path to the problem set")
    parser.add_argument("-p", "--pop", metavar='', type=pos_int, help="the population size")
    parser.add_argument("-s", "--sel", metavar='', type=pos_int, help="the selection size")
    parser.add_argument("-g", "--ngen", metavar='', type=pos_int, help="the generation size")
    parser.add_argument("-m", "--mutpb", metavar='', type=pos_float, help="the mutation probability")
    parser.add_argument("-c", "--cxpb", metavar='', type=pos_float, help="the crossover probability")
    parser.add_argument("-r", "--run", metavar='', type=int_ge_one, help="the number of times to run the problem")

    cx_types = parser.add_mutually_exclusive_group()
    cx_types.add_argument("-B", "--brxo", action='store_true', help="use best route crossover")
    cx_types.add_argument("-C", "--cxo", action='store_true', help="use cycle crossover")
    cx_types.add_argument("-E", "--erxo", action='store_true', help="use edge recombination crossover")
    cx_types.add_argument("-O", "--oxo", action='store_true', help="use order crossover")

    mt_types = parser.add_mutually_exclusive_group()
    mt_types.add_argument("-I", "--vmt", action='store_true', help="use inversion mutation")
    mt_types.add_argument("-W", "--swmt", action='store_true', help="use swap mutation")
    mt_types.add_argument("-G", "--gvmt", action='store_true', help="use GVR based scramble mutation")

    parser.add_argument("-i", "--indent", metavar='', nargs="?", type=int_ge_one, const=2,
                        help="the indentation amount of the result string")
    parser.add_argument("-P", "--pgen", action='store_true', help="prints the current generation")
    parser.add_argument("-A", "--agen", action='store_true', help="prints the average fitness every 1000 generations")

    parser.add_argument("-S", "--save", action="store_true", help="saves the results to a file")
    parser.add_argument("-R", "--routes", action="store_true", help="adds every route (verbose) of the best "
                                                                    "individual to the result")
    parser.add_argument("-M", "--plot", action="store_true", help="plot average fitness across generations with "
                                                                  "matplotlib")
    args = parser.parse_args()

    p_set = parse_file(args.file) if args.file else parse_file("data/A-n54-k7.ocvrp")
    pop = args.pop if args.pop else pop
    sel = args.sel if args.sel else sel
    ngen = args.ngen if args.ngen else ngen
    mutpb = args.mutpb if args.mutpb else mutpb
    cxpb = args.cxpb if args.cxpb else cxpb

    runtime = args.run if args.run else 1

    if args.cxo:
        cx_algo = algorithms.cycle_xo
    elif args.erxo:
        cx_algo = algorithms.edge_recomb_xo
    elif args.oxo:
        cx_algo = algorithms.order_xo

    if args.swmt:
        mt_algo = algorithms.swap_mut
    elif args.gvmt:
        mt_algo = algorithms.gvr_scramble_mut

    runs = {"RUNS": {}}
    for i in range(1, runtime + 1):
        result = CVRP(problem_set=p_set,
                      population_size=pop,
                      selection_size=sel,
                      ngen=ngen,
                      mutpb=mutpb,
                      cxpb=cxpb,
                      pgen=args.pgen,
                      agen=args.agen,
                      cx_algo=cx_algo,
                      mt_algo=mt_algo,
                      plot=args.plot,
                      verbose_routes=args.routes).run()

        runs["RUNS"][f"RUN_{i}"] = result

        print(f"\n\n============END RUN {i}============\n\n")

    print("...Run Complete")
    runs['BEST_RUN'] = min(runs['RUNS'], key=lambda run: runs['RUNS'][run]['best_individual_fitness'])
    runs['WORST_RUN'] = max(runs['RUNS'], key=lambda run: runs['RUNS'][run]['best_individual_fitness'])
    runs["AVG_FITNESS"] = sum(v['best_individual_fitness'] for v in runs['RUNS'].values()) / len(runs['RUNS'].keys())

    js_res = json.dumps(obj=runs,
                        cls=BuildingEncoder,
                        indent=args.indent)

    now = datetime.datetime.now().strftime("%Y%m%d__%I_%M_%S%p")
    f_name = f'{cvrp.cx_algo}_{cvrp.ngen}__{now}'

    if args.save:
        if not os.path.isdir('./results'):
            os.mkdir('./results')

        with open(f'results/{f_name}.json', 'w+') as fc:
            fc.write(js_res)

    print(js_res)

    if args.plot:
        for i, k in enumerate(runs['RUNS'].keys(), start=1):
            plt = runs['RUNS'][k]['mat_plot']
            plt.savefig(f'results/{f_name}__RUN_{i}.jpg', bbox_inches='tight')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as kms:
        print("Keyboard Interrupt")
        try:
            sys.exit(1)
        except SystemExit:
            os._exit(1)
