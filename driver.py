import json

from cvrp import CVRP
from util import populate_from_file

if __name__ == '__main__':
    prblm = populate_from_file("problems_sets/A-n54-k7.txt")
    cvrp = CVRP(problem_set=prblm,
                population_size=200,
                selection_size=5,
                ngen=50_000,
                mutpb=0.15,
                cxpb=0.75)

    result = cvrp.run()
    print(json.dumps(obj=result,
                     default=lambda o: o.__dict__,
                     indent=2))
