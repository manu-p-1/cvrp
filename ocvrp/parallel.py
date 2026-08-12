"""
https://github.com/manu-p-1/cvrp
parallel.py

Parallelization utilities for CVRP optimization. Provides a multi-start
island model that runs independent GA instances across CPU cores, and
helpers for parallel population initialization.

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


def _get_algo_by_name(algo_name):
    """Resolve an algorithm function by its ``__name__`` string.

    Searches both ``ocvrp.algorithms`` and ``ocvrp.local_search``.
    """
    from ocvrp import algorithms as alg
    from ocvrp import local_search as ls

    for mod in (alg, ls):
        obj = getattr(mod, algo_name, None)
        if obj is not None and callable(obj):
            return obj

    raise ValueError(f"Unknown algorithm: {algo_name}")


def _island_worker(problem_set_path, pop_size, ngen, seed,
                   cx_algo_name, mt_algo_name,
                   selection_size, mutpb, cxpb, seed_pct):
    """Execute one island's GA.  Must be at module level for pickling."""
    r.seed(seed)

    cx_fn = _get_algo_by_name(cx_algo_name)
    mt_fn = _get_algo_by_name(mt_algo_name)

    from ocvrp.cvrp import CVRP
    cvrp = CVRP(
        problem_set_path,
        population_size=pop_size,
        ngen=ngen,
        selection_size=selection_size,
        mutpb=mutpb,
        cxpb=cxpb,
        cx_algo=cx_fn,
        mt_algo=mt_fn,
        seed_pct=seed_pct,
        pgen=False,
        agen=False,
        plot=False,
    )
    return cvrp.run()


class IslandModel:
    """Multi-start parallel genetic algorithm for CVRP.

    Launches *n* independent GA instances (islands) in separate OS processes,
    each with its own random seed and sub-population. The overall best
    solution found across all islands is returned.

    This is the simplest yet highly effective form of parallel evolutionary
    search: independent runs with diverse initializations naturally explore
    different regions of the solution space.

    Usage::

        from ocvrp.parallel import IslandModel
        model = IslandModel("data/A-n54-k7.ocvrp", n_islands=4, ngen=50000)
        result = model.run()

    :param problem_set_path: Path to the ``.ocvrp`` problem file
    :param n_islands: Number of parallel islands (defaults to CPU count, max 8)
    :param total_pop: Total population distributed across islands
    :param ngen: Generations per island
    :param cvrp_kwargs: Additional keyword arguments forwarded to each
        :class:`~ocvrp.cvrp.CVRP` instance (e.g. ``cx_algo``, ``mutpb``)
    """

    def __init__(self, problem_set_path, n_islands=None, total_pop=800,
                 ngen=100_000, **cvrp_kwargs):
        self._problem_set_path = problem_set_path
        self._n_islands = n_islands or min(mp.cpu_count(), 8)
        self._total_pop = total_pop
        self._ngen = ngen
        self._cvrp_kwargs = cvrp_kwargs

    def run(self):
        """Execute the island model and return the best result dict."""
        island_pop = max(50, self._total_pop // self._n_islands)
        seeds = [r.randint(0, 2 ** 31) for _ in range(self._n_islands)]

        # Resolve algorithm callables to their names for pickling
        cx = self._cvrp_kwargs.get('cx_algo')
        cx_name = cx.__name__ if callable(cx) else (cx or 'best_route_xo')
        mt = self._cvrp_kwargs.get('mt_algo')
        mt_name = mt.__name__ if callable(mt) else (mt or 'inversion_mut')

        sel_size = self._cvrp_kwargs.get('selection_size', 5)
        mutpb = self._cvrp_kwargs.get('mutpb', 0.15)
        cxpb = self._cvrp_kwargs.get('cxpb', 0.85)
        seed_pct = self._cvrp_kwargs.get('seed_pct', 0.0)

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
                    cx_name,
                    mt_name,
                    sel_size,
                    mutpb,
                    cxpb,
                    seed_pct,
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

