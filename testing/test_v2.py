"""Quick smoke tests for v2.0 features."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}")
        failed += 1

print("=== Distance Matrix ===")
from ocvrp.distance import DistanceMatrix
from ocvrp.util import OCVRPParser, Building

ps = OCVRPParser("data/A-n54-k7.ocvrp").parse()
depot = ps.get_ps_depot()
buildings = ps.get_ps_buildings()
dm = DistanceMatrix(depot, buildings)

b1, b2 = buildings[0], buildings[1]
check("dist matrix matches Building.distance",
      dm.dist(b1, b2) == Building.distance(b1, b2))
check("dist matrix symmetric",
      dm.dist(b1, b2) == dm.dist(b2, b1))
check("dist_by_id works",
      dm.dist_by_id(b1.node, b2.node) == dm.dist(b1, b2))

print("\n=== Constructive Heuristics ===")
from ocvrp.constructive import nearest_neighbor, clarke_wright_savings, sweep_heuristic

cap = ps.get_ps_capacity()

nn_routes = nearest_neighbor(depot, buildings, cap, dm)
nn_nodes = sum(len(r) for r in nn_routes)
check(f"nearest neighbor covers all nodes ({nn_nodes})", nn_nodes == len(buildings))
check("nearest neighbor capacity feasible",
      all(sum(b.quant for b in r) <= cap for r in nn_routes))

cw_routes = clarke_wright_savings(depot, buildings, cap, dm)
cw_nodes = sum(len(r) for r in cw_routes)
check(f"clarke-wright covers all nodes ({cw_nodes})", cw_nodes == len(buildings))
check("clarke-wright capacity feasible",
      all(sum(b.quant for b in r) <= cap for r in cw_routes))

sw_routes = sweep_heuristic(depot, buildings, cap, dm)
sw_nodes = sum(len(r) for r in sw_routes)
check(f"sweep covers all nodes ({sw_nodes})", sw_nodes == len(buildings))
check("sweep capacity feasible",
      all(sum(b.quant for b in r) <= cap for r in sw_routes))

print("\n=== CVRP with Distance Matrix ===")
from ocvrp.cvrp import CVRP

cvrp = CVRP("data/A-n54-k7.ocvrp", ngen=200, population_size=50)
check("dist_matrix available", cvrp.dist_matrix is not None)
result = cvrp.run()
check("run completes", "best_individual_fitness" in result)
check("fitness is positive", result["best_individual_fitness"] > 0)

print("\n=== Seeded Initialization ===")
cvrp2 = CVRP("data/A-n54-k7.ocvrp", ngen=200, population_size=50, seed_pct=0.1)
result2 = cvrp2.run()
check("seeded run completes", "best_individual_fitness" in result2)

print("\n=== Inter-route Local Search ===")
from ocvrp.local_search import relocate_mut, exchange_mut, two_opt_star_mut

cvrp3 = CVRP("data/A-n54-k7.ocvrp", ngen=100, population_size=30, mt_algo=relocate_mut)
r3 = cvrp3.run()
check("relocate_mut run completes", r3["best_individual_fitness"] > 0)

cvrp4 = CVRP("data/A-n54-k7.ocvrp", ngen=100, population_size=30, mt_algo=exchange_mut)
r4 = cvrp4.run()
check("exchange_mut run completes", r4["best_individual_fitness"] > 0)

cvrp5 = CVRP("data/A-n54-k7.ocvrp", ngen=100, population_size=30, mt_algo=two_opt_star_mut)
r5 = cvrp5.run()
check("two_opt_star_mut run completes", r5["best_individual_fitness"] > 0)

print("\n=== Algorithm Registry ===")
from ocvrp.algorithms import CROSSOVER_REGISTRY, MUTATION_REGISTRY
check("crossover registry has 5 entries", len(CROSSOVER_REGISTRY) == 5)
check("mutation registry has 7 entries", len(MUTATION_REGISTRY) == 7)

print("\n=== CLI ===")
import subprocess
r = subprocess.run([sys.executable, "driver.py", "--help"], capture_output=True, text=True)
check("--islands in help", "--islands" in r.stdout)
check("--seed-pct in help", "--seed-pct" in r.stdout)
check("--reloc in help", "--reloc" in r.stdout)
check("--exch in help", "--exch" in r.stdout)
check("--toptstar in help", "--toptstar" in r.stdout)

print(f"\n{passed} passed, {failed} failed")
sys.exit(failed)
