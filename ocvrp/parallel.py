"""
https://github.com/manu-p-1/cvrp
parallel.py

Runs several independent copies of the GA (islands) in separate processes
and keeps the best result. Islands never exchange individuals, so this is a
multi-start GA spread across CPU cores rather than a migration-based island
model.

Reference:
    Alba, E. and Tomassini, M. (2002). "Parallelism and evolutionary
    algorithms." IEEE Transactions on Evolutionary Computation, 6(5),
    443-462.

    Cantu-Paz, E. (1998). "A survey of parallel genetic algorithms."
    Calculateurs Paralleles, Reseaux et Systemes Repartis, 10(2), 141-171.
"""

import multiprocessing as mp
import random as r
from concurrent.futures import ProcessPoolExecutor


def _island_worker(problem_set_path, pop_size, ngen, seed, cvrp_kwargs):
    """Run one island. Defined at module level so it can be pickled."""
    r.seed(seed)

    from ocvrp.cvrp import CVRP
    # Output and plots from several processes would collide, so islands run quietly.
    kwargs = dict(cvrp_kwargs, pgen=False, agen=False, plot=False, plot_save_path=None)
    cvrp = CVRP(problem_set_path, population_size=pop_size, ngen=ngen, **kwargs)
    return cvrp.run()


class IslandModel:
    """Run several independent copies of the GA in parallel and keep the best.

    Each island is a separate process with its own random seed and an equal
    share of ``total_pop`` (at least 50 individuals per island).

    Usage::

        from ocvrp.parallel import IslandModel

        if __name__ == "__main__":
            model = IslandModel("data/A-n54-k7.ocvrp", n_islands=4, ngen=50000)
            result = model.run()

    The ``__main__`` guard is required on macOS and Windows, where worker
    processes re-import the main module.

    :param problem_set_path: Path to the ``.ocvrp`` problem file
    :param n_islands: Number of islands (defaults to CPU count, max 8)
    :param total_pop: Total population, split evenly across islands
    :param ngen: Generations per island
    :param cvrp_kwargs: Other keyword arguments passed to each
        :class:`~ocvrp.cvrp.CVRP` (e.g. ``cx_algo``, ``mutpb``). Operator
        functions have to be defined at module level so they can be pickled.
    """

    def __init__(self, problem_set_path, n_islands=None, total_pop=800,
                 ngen=100_000, **cvrp_kwargs):
        self._problem_set_path = problem_set_path
        self._n_islands = n_islands or min(mp.cpu_count(), 8)
        self._total_pop = total_pop
        self._ngen = ngen
        self._cvrp_kwargs = cvrp_kwargs

    def run(self):
        """Run every island and return the result dict of the best one."""
        island_pop = max(50, self._total_pop // self._n_islands)
        seeds = [r.randint(0, 2 ** 31) for _ in range(self._n_islands)]

        n = self._n_islands
        print(f"Launching {n} island(s) "
              f"({island_pop} individuals each, {self._ngen} generations)...")

        with ProcessPoolExecutor(max_workers=n) as executor:
            futures = []
            for i in range(n):
                fut = executor.submit(
                    _island_worker,
                    self._problem_set_path,
                    island_pop,
                    self._ngen,
                    seeds[i],
                    self._cvrp_kwargs,
                )
                futures.append((i, fut))

            results = []
            for i, fut in futures:
                try:
                    result = fut.result()
                    results.append(result)
                    print(f"  Island {i + 1}/{n} finished - "
                          f"fitness = {result['best_individual_fitness']}")
                except Exception as e:
                    print(f"  Island {i + 1}/{n} failed: {e}")

        if not results:
            raise RuntimeError("All islands failed")

        best = min(results, key=lambda res: res['best_individual_fitness'])
        best['n_islands'] = n
        best['island_pop'] = island_pop
        return best

