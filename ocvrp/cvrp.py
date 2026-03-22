"""
https://github.com/manu-p-1/cvrp
cvrp.py

This module contains the class to run the CVRP problem optimization
"""

import enum
import math
import random as r
import time
from typing import Dict, Tuple, List, Union

from ocvrp import algorithms as alg
from ocvrp.util import Building, Individual, OCVRPParser


class ReplStrat(enum.Enum):
    """
    Enum class to represent Replacement Strategies
    """
    RAND = enum.auto()
    BEST = enum.auto()
    WORST = enum.auto()


class CVRP:
    """
    The CVRP class which is responsible for forming and running the CVRP optimization
    """

    def __init__(self, problem_set_path: str,
                 population_size: int = 800,
                 selection_size: int = 5,
                 ngen: int = 100_000,
                 mutpb: float = 0.15,
                 cxpb: float = 0.85,
                 cx_algo=alg.best_route_xo,
                 mt_algo=alg.inversion_mut,
                 pgen: bool = False,
                 agen: bool = False,
                 plot: bool = False,
                 plot_save_path: str = None,
                 verbose_routes: bool = False):
        """
        Creates a new CVRP instance based on the following parameters

        :param problem_set_path: The path to the problem set
        :param population_size: The population size
        :param selection_size: The selection size
        :param ngen: The number of generations to run the algorithm
        :param mutpb: The mutation probability
        :param cxpb: The crossover probability
        :param cx_algo: The crossover algorithm - a function reference
        :param mt_algo: The mutation algorithm - a function reference
        :param pgen: A bool to flag whether to print the current generation
        :param agen: A bool to flag whether to print generation statistics
        :param plot: A bool to flag whether to plot the information to a results folder
        :param plot_save_path: The file path to save the plot image to (enables plotting). Overrides plot if set.
        :param verbose_routes: A bool to flag whether to save the exact route information to the results
        """
        print("Loading problem set...")
        ps_strat = OCVRPParser(problem_set_path).parse()

        self._problem_set_name = ps_strat.get_ps_name()
        self._problem_set_comments = ps_strat.get_ps_comments()
        self._vehicle_cap = ps_strat.get_ps_capacity()
        self._optimal_fitness = ps_strat.get_ps_optimal()
        self._dim = ps_strat.get_ps_dim()
        self._depot = ps_strat.get_ps_depot()
        self._problem_set_buildings_orig = ps_strat.get_ps_buildings()
        self._pop = []

        self.population_size = population_size
        self.selection_size = selection_size
        self.ngen = ngen
        self.mutpb = mutpb
        self.cxpb = cxpb
        self.cx_algo = cx_algo
        self.mt_algo = mt_algo
        self.pgen = pgen
        self.agen = agen
        self.plot = plot
        self.plot_save_path = plot_save_path
        self.verbose_routes = verbose_routes

        # Create n random permutations from the problem set
        print(f"Initializing population of {self._population_size:,} individuals...")
        self.reset()
        print("Population initialized.")

    def calc_fitness(self, individual):
        """
        Calculates the fitness value by changing the representation to GVR form. The distance
        is calculated between each route. For each route, the distance from the depot to the first node and
        the distance from the last node to the depot is summed.

        :param individual: The Individual to evaluate the fitness
        :return: The fitness value
        """
        distance = 0
        partitioned_routes = self.partition_routes(individual)
        for _, route in partitioned_routes.items():
            for h1, h2 in zip(route, route[1:]):
                distance += Building.distance(h1, h2)
            distance += Building.distance(self._depot, route[0])
            distance += Building.distance(route[-1], self._depot)

        return distance

    def partition_routes(self, individual: Individual) -> Dict:
        """
        Places the individual into its GVR representation.

        The representation splits customers into different routes based off of maximum vehicle capacity.

        :param individual: The Individual to place into GVR representation
        :return: A Dict of routes and customers per route
        """
        routes = {1: []}
        current_weight = 0
        route_counter = 1

        for building in individual:
            if current_weight + building.quant > self._vehicle_cap:
                route_counter += 1
                current_weight = 0
                routes[route_counter] = []

            routes[route_counter].append(building)
            current_weight += building.quant

        return routes

    @staticmethod
    def de_partition_routes(partitioned_routes: Dict) -> List:
        """
        Returns the GVR representation back into its vector permutation form
        :param partitioned_routes: The GVR representation as a Dict
        :return: A list containing the permutation of customers
        """
        ll = []
        for v in partitioned_routes.values():
            ll.extend(v)
        return ll

    def select(self) -> Tuple[Individual, Individual]:
        """
        For selection, individuals are randomly sampled. Of those, the two with the best fitness
        are chosen to become parents. We employ a form of tournament selection here.

        :return: A tuple containing parent one and parent two
        """
        take_five = r.sample(self._pop, self._selection_size)

        parent1 = self._get_and_remove(take_five, ReplStrat.BEST)
        parent2 = self._get_and_remove(take_five, ReplStrat.BEST)

        return parent1, parent2

    def replacement_strat(self, individual: Individual, rs: ReplStrat) -> None:
        """
        Restricted tournament replacement: selects a random tournament from the
        population and replaces the worst member only if the new individual is
        at least as good.

        :param individual: The Individual to add to the population
        :param rs: The ReplStrat enum
        :return: None
        """
        if rs == ReplStrat.WORST:
            tourn_size = min(7, len(self._pop))
            tourn_indices = r.sample(range(len(self._pop)), tourn_size)
            worst_idx = max(tourn_indices, key=lambda i: self._pop[i].fitness)
            if individual.fitness <= self._pop[worst_idx].fitness:
                self._pop[worst_idx] = individual
        else:
            self._get_and_remove(self._pop, rs)
            self._pop.append(individual)

    @staticmethod
    def _get_nworst(sel_values, n):
        """
        Grabs the N worst values from an iterable collection of values.

        :param sel_values: The iterable collection
        :param n: The maximum amount of worst individuals to find
        :return: The n worst values from the iterable
        """
        v = sel_values[:]
        v.sort()
        return v[-n:]

    @staticmethod
    def _get_and_remove(sel_values, rs: ReplStrat):
        """
        Removes a value from the iterable collection of values based on a ReplStrat instance.

        :param sel_values: The iterable collection
        :param rs: The ReplStrat enum
        :return: The removed value from sel_values based on the ReplStrat enum
        """
        if rs == ReplStrat.RAND:
            val = r.choice(sel_values)
        elif rs == ReplStrat.BEST:
            val = min(sel_values)
        else:
            val = max(sel_values)
        sel_values.remove(val)
        return val

    def _unique_genotype_count(self):
        """
        Counts unique genotypes by hashing the node-ID tuple of each individual.
        """
        seen = set()
        for ind in self._pop:
            key = tuple(b.node for b in ind)
            seen.add(key)
        return len(seen)

    def _unique_genotype_ratio(self):
        """Returns the ratio of unique genotypes to population size."""
        return self._unique_genotype_count() / self._population_size

    def reset(self):
        """
        Resets this instance by reassigning this population to a random permutation of values.
        It does not reset any other operator or probability.

        :return: None
        """
        self._pop = []
        for i in range(self._population_size):
            rpmt = r.sample(self._problem_set_buildings_orig, self._dim)
            self._pop.append(Individual(rpmt, self.calc_fitness(rpmt)))
            if self._population_size >= 10000 and (i + 1) % 10000 == 0:
                print(f"  {i + 1:,}/{self._population_size:,} individuals created", end='\r')

    def run(self) -> dict:
        """
        Runs the evolutionary loop:
        1.) Parent Selection
        2.) Crossover
        3.) Mutation
        4.) Fitness Evaluation
        5.) Survivor Replacement
        6.) Diversity Maintenance

        :return: A potential solution if found or the closest optimal solution otherwise.
        """

        print(f"Running {self._ngen} generation(s)...")

        best_data, avg_data, worst_data, diversity_data, gen_data = [], [], [], [], []

        # The bound at which we start diversity maintenance (10% unique genotypes)
        div_thresh_lb = math.ceil(0.10 * self._population_size)

        # The amount of individuals to replace for diversity maintenance (30%, runs more frequently)
        div_picking_rng = round(0.30 * self._population_size)

        t = time.process_time()
        found = False
        indiv = None

        # Track the best individual ever seen (elitism safeguard)
        best_ever = min(self._pop)

        # Adaptive mutation rate starts at configured value
        effective_mutpb = self._mutpb

        # Start the generation count
        for gen in range(1, self._ngen + 1):

            cx_prob = r.random() < self._cxpb
            mut_prob = r.random() < effective_mutpb

            parent1, parent2 = self.select()

            if cx_prob:
                # All crossover functions now return (child1, child2) tuples
                child1, child2 = self._cx_func(parent1, parent2, self)
            else:
                # Clone parents to avoid in-place mutation corruption
                child1 = Individual(parent1[:], parent1.fitness)
                child2 = Individual(parent2[:], parent2.fitness)

            if mut_prob or not cx_prob:
                # Mutate if probability fires or if crossover didn't fire
                child1 = self._mt_func(child1, self)
                child2 = self._mt_func(child2, self)

            # Calculate fitness if not yet evaluated (crossover/mutation sets fitness=None)
            if child1.fitness is None:
                child1.fitness = self.calc_fitness(child1)

            if child2.fitness is None:
                child2.fitness = self.calc_fitness(child2)

            # Update best-ever tracking (elitism)
            if child1.fitness < best_ever.fitness:
                best_ever = Individual(child1[:], child1.fitness)
            if child2.fitness < best_ever.fitness:
                best_ever = Individual(child2[:], child2.fitness)

            # One of the children were found to have an optimal fitness
            if child1.fitness == self._optimal_fitness or child2.fitness == self._optimal_fitness:
                indiv = child1 if child1.fitness == self._optimal_fitness else child2
                found = True
                break

            self.replacement_strat(child1, ReplStrat.WORST)
            self.replacement_strat(child2, ReplStrat.WORST)

            if self._pgen:
                print(f'GEN: {gen}/{self._ngen}', end='\r')

            min_indv, max_indv, avg_fit = None, None, None

            # Every 250 generations, we print the statistics or plot the value if needed
            if gen % 250 == 0 or gen == 1:
                if self._agen:
                    uq_indv = self._unique_genotype_count()
                    min_indv = min(self._pop).fitness
                    max_indv = max(self._pop).fitness
                    avg_fit = round(sum(self._pop) / self._population_size)

                    print(f"UNIQUE GENOTYPES: {uq_indv}/{self._population_size}")
                    print(f"GEN {gen} BEST FITNESS: {min_indv}")
                    print(f"GEN {gen} WORST FITNESS: {max_indv}")
                    print(f"GEN {gen} AVG FITNESS: {avg_fit}\n\n")

                if self._plot:
                    min_indv = min(self._pop).fitness if min_indv is None else min_indv
                    best_data.append(min_indv)

                    max_indv = max(self._pop).fitness if max_indv is None else max_indv
                    worst_data.append(max_indv)

                    avg_fit = round(sum(self._pop) / self._population_size) if avg_fit is None else avg_fit
                    avg_data.append(avg_fit)

                    uq_indv = self._unique_genotype_count() if not self._agen else uq_indv
                    diversity_data.append(uq_indv / self._population_size)

                    gen_data.append(gen)

            # Adaptive mutation: adjust rate based on population diversity
            if gen % 250 == 0:
                uq_ratio = self._unique_genotype_ratio()
                if uq_ratio < 0.05:
                    effective_mutpb = min(1.0, self._mutpb * 4)
                elif uq_ratio < 0.15:
                    effective_mutpb = min(0.5, self._mutpb * 2)
                else:
                    effective_mutpb = self._mutpb

            # Every 2,000 generations we check diversity and restore if needed
            if gen % 2000 == 0 and self._unique_genotype_count() <= div_thresh_lb:
                if self._agen:
                    print("===============DIVERSITY MAINT===============")
                worst = self._get_nworst(self._pop, div_picking_rng)
                # Cache the top 10% pool once (avoid re-sorting inside the loop)
                top_pool = sorted(self._pop)[:max(1, self._population_size // 10)]

                # Use random perturbations for diversity restoration instead of the
                # configured mutation, which may converge to the same local optimum.
                div_mutations = [alg.displacement_mut, alg.scramble_mut, alg.inversion_mut]

                for k in range(div_picking_rng):
                    c = r.choice(top_pool)
                    # Apply 2-3 random perturbations, then re-optimize with
                    # the configured mutation (ILS-style perturb and re-optimize)
                    mutated = c
                    for _ in range(r.randint(2, 3)):
                        mut_func = r.choice(div_mutations)
                        mutated = mut_func(mutated, self)
                    mutated = self._mt_func(mutated, self)
                    if mutated.fitness is None:
                        mutated.fitness = self.calc_fitness(mutated)

                    self._pop.remove(worst[k])
                    self._pop.append(mutated)

        # Use the best individual ever seen (may have been lost during replacement)
        closest = min(self._pop)
        if best_ever.fitness < closest.fitness:
            closest = best_ever
        end = time.process_time() - t

        return self._create_solution(indiv if found else closest, end,
                                        best_data, avg_data, worst_data, diversity_data, gen_data,
                                        plot_save_path=self._plot_save_path)

    def _create_solution(self, individual, comp_time, best_data, avg_data,
                          worst_data, diversity_data, gen_data,
                          plot_save_path: str = None) -> dict:
        """
        Creates a dictionary with all of the information about the solution or closest solution
        that was found in the EA.

        :param individual: The chromosome that was matched as the solution or closest solution
        :param comp_time: The computation time of the algorithm
        :param best_data: The best fitness values from the runs as a list
        :param avg_data: The average of fitness values from the runs as a list
        :param worst_data: The worst fitness values from the runs as a list
        :param diversity_data: The population diversity ratio from the runs as a list
        :param gen_data: The generation numbers corresponding to each data point
        :param plot_save_path: The file path to save the plot image to
        :return: A dictionary with the information
        """
        fig = None
        if self._plot:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec

            fig = plt.figure(figsize=(16, 14), dpi=150)
            gs = gridspec.GridSpec(2, 2, hspace=0.30, wspace=0.28)

            title = (f"{self._problem_set_name}  |  {self._cx_algo} + {self._mt_algo}  |  "
                     f"pop={self._population_size}  gens={self._ngen}  "
                     f"cx={self._cxpb}  mut={self._mutpb}")
            fig.suptitle(title, fontsize=12, fontweight='bold', y=0.98)

            # ── Panel 1: Convergence curve ──
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(gen_data, best_data, linewidth=1.5, color='#2196F3', label='Best')
            ax1.plot(gen_data, avg_data, linewidth=1.2, color='#FF9800', alpha=0.8, label='Average')
            ax1.fill_between(gen_data, best_data, worst_data, alpha=0.10, color='#9E9E9E', label='Best–Worst range')
            if self._optimal_fitness and self._optimal_fitness > 0:
                ax1.axhline(y=self._optimal_fitness, color='#4CAF50', linestyle='--',
                            linewidth=1, label=f'Optimal ({self._optimal_fitness})')
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Fitness (distance)')
            ax1.set_title('Convergence Curve')
            ax1.legend(fontsize=8, loc='upper right')
            ax1.grid(True, alpha=0.3)

            # ── Panel 2: Population diversity ──
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(gen_data, [d * 100 for d in diversity_data],
                     linewidth=1.5, color='#9C27B0')
            ax2.axhline(y=10.0, color='#F44336', linestyle=':', linewidth=1,
                        alpha=0.7, label='Diversity maintenance threshold (10%)')
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Unique genotypes (% of population)')
            ax2.set_title('Population Diversity')
            ax2.set_ylim(bottom=0)
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)

            # ── Panel 3: Best solution route map ──
            ax3 = fig.add_subplot(gs[1, 0])
            partitioned_viz = self.partition_routes(individual)
            route_colors = plt.cm.Set2(range(len(partitioned_viz)))
            for idx, (route_num, route) in enumerate(partitioned_viz.items()):
                xs = [self._depot.x] + [b.x for b in route] + [self._depot.x]
                ys = [self._depot.y] + [b.y for b in route] + [self._depot.y]
                ax3.plot(xs, ys, 'o-', color=route_colors[idx], markersize=4,
                         linewidth=1.2, alpha=0.85, label=f'Route {route_num}')
                for b in route:
                    ax3.annotate(str(b.node), (b.x, b.y), fontsize=5,
                                 ha='center', va='bottom', textcoords='offset points',
                                 xytext=(0, 4))
            ax3.plot(self._depot.x, self._depot.y, 's', color='#F44336',
                     markersize=10, zorder=5, label='Depot')
            ax3.set_xlabel('X coordinate')
            ax3.set_ylabel('Y coordinate')
            ax3.set_title(f'Best Solution Routes  (fitness = {individual.fitness})')
            if len(partitioned_viz) <= 12:
                ax3.legend(fontsize=6, loc='best', ncol=2)
            ax3.grid(True, alpha=0.2)
            ax3.set_aspect('equal', adjustable='datalim')

            # ── Panel 4: Final population fitness distribution ──
            ax4 = fig.add_subplot(gs[1, 1])
            pop_fitness = [ind.fitness for ind in self._pop]
            ax4.hist(pop_fitness, bins=min(50, max(10, len(set(pop_fitness)))),
                     color='#607D8B', edgecolor='white', linewidth=0.5, alpha=0.85)
            ax4.axvline(x=individual.fitness, color='#2196F3', linestyle='--',
                        linewidth=1.5, label=f'Best = {individual.fitness}')
            if self._optimal_fitness and self._optimal_fitness > 0:
                ax4.axvline(x=self._optimal_fitness, color='#4CAF50', linestyle='--',
                            linewidth=1, label=f'Optimal = {self._optimal_fitness}')
            ax4.axvline(x=sum(pop_fitness) / len(pop_fitness), color='#FF9800',
                        linestyle=':', linewidth=1.2,
                        label=f'Mean = {sum(pop_fitness) / len(pop_fitness):.0f}')
            ax4.set_xlabel('Fitness (distance)')
            ax4.set_ylabel('Count')
            ax4.set_title('Final Population Fitness Distribution')
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3, axis='y')

        partitioned = self.partition_routes(individual)

        obj = {
            "name": type(self).__name__,
            "problem_set_name": self._problem_set_name,
            "problem_set_optimal": self._optimal_fitness,
            "time": f"{comp_time} seconds",
            "vehicles": len(partitioned.keys()),
            "vehicle_capacity": self._vehicle_cap,
            "dimension": self._dim,
            "population_size": self._population_size,
            "selection_size": self._selection_size,
            "generations": self._ngen,
            "cxpb": self._cxpb,
            "mutpb": self._mutpb,
            "cx_algorithm": self._cx_algo,
            "mut_algorithm": self._mt_algo,
            "best_individual_fitness": individual.fitness,
        }

        if self._plot and fig is not None:
            if plot_save_path:
                import os
                os.makedirs(os.path.dirname(plot_save_path) if os.path.dirname(plot_save_path) else '.', exist_ok=True)
                fig.savefig(plot_save_path, bbox_inches='tight')
                obj["plot_save_path"] = plot_save_path
            plt.close(fig)

        if self._verbose_routes:
            obj["best_individual"] = partitioned

        return obj

    @property
    def problem_set_name(self) -> str:
        """
        Returns the name of the problem set
        :return: The name of the problem set for this instance
        """
        return self._problem_set_name

    @property
    def problem_set_comments(self) -> Union[str, None]:
        """
        Returns the comments of the problem set or None if a COMMENTS header was not provided
        :return: The comments of the problem set or None if a COMMENTS header was not provided for this instance
        """
        return self._problem_set_comments

    @property
    def vehicle_cap(self) -> int:
        """
        Returns the vehicle capacity of the problem set
        :return: The vehicle capacity of the problem set for this instance
        """
        return self._vehicle_cap

    @property
    def optimal_fitness(self) -> int:
        """
        Returns the optimal fitness of the problem set
        :return: The optimal fitness of the problem set for this instance
        """
        return self._optimal_fitness

    @property
    def dim(self) -> int:
        """
        Returns the working dimension of the problem set (not including the depot)
        :return: The working dimension of the problem set for this instance
        """
        return self._dim

    @property
    def depot(self) -> Building:
        """
        Returns the depot location of the problem set
        :return: The depot location of the problem set as a Building object
        """
        return self._depot

    @property
    def pop(self) -> List[Individual]:
        """
        Returns the population for this instance
        :return: The entire population for this instance as a list of Individual instances
        """
        return self._pop

    @property
    def population_size(self) -> int:
        """
        Returns the population size for this instance
        :return: The population size for this instance
        """
        return self._population_size

    @population_size.setter
    def population_size(self, population_size: int) -> None:
        """
        Sets the population size for this instance. The population size must be greater
        than 5.
        :param population_size: The specified population size
        :return: None
        """
        self._is_int_ge(population_size, 5)
        self._population_size = population_size

    @property
    def selection_size(self) -> int:
        """
        Returns the selection size for this instance
        :return: The selection size for this instance
        """
        return self._selection_size

    @selection_size.setter
    def selection_size(self, selection_size: int) -> None:
        """
        Sets the selection size for this instance
        :param selection_size: The selection size for this instance
        :return: None
        """
        self._is_int_ge(selection_size, 1)
        self._selection_size = selection_size

    @property
    def ngen(self) -> int:
        """
        Returns the number of generations this instance runs for
        :return: The number of generations this instance runs for
        """
        return self._ngen

    @ngen.setter
    def ngen(self, ngen: int) -> None:
        """
        Sets the number of generations this instance runs for
        :param ngen: The number of generations this instance runs for
        :return: None
        """
        self._is_int_ge(ngen, 1)
        self._ngen = ngen

    @property
    def mutpb(self) -> float:
        """
        Returns the mutation probability for this instance
        :return: The mutation probability for this instance
        """
        return self._mutpb

    @mutpb.setter
    def mutpb(self, mutpb: float) -> None:
        """
        Sets the mutation probability for this instance
        :param mutpb: The mutation probability for this instance
        :return: None
        """
        self._is_probability(mutpb)
        self._mutpb = mutpb

    @property
    def cxpb(self) -> float:
        """
        Returns the crossover probability for this instance
        :return: The crossover probability for this instance
        """
        return self._cxpb

    @cxpb.setter
    def cxpb(self, cxpb: float) -> None:
        """
        Sets the crossover probability for this instance
        :param cxpb: The crossover probability for this instance
        :return: None
        """
        self._is_probability(cxpb)
        self._cxpb = cxpb

    @property
    def cx_algo(self) -> str:
        """
        Returns the name of the crossover algorithm used in this instance
        :return: The name of the crossover algorithm used in this instance
        """
        return self._cx_algo

    @cx_algo.setter
    def cx_algo(self, cx_algo) -> None:
        """
        Sets a reference to the crossover function that will be used in this instance. Crossover algorithms
        are found in the algorithms.py module
        :param cx_algo: The crossover algorithm that will be used in this instance
        :return: None
        """
        self._cx_func = cx_algo
        self._cx_algo = cx_algo.__name__

    @property
    def mt_algo(self) -> str:
        """
        Returns the name of the mutation algorithm used in this instance
        :return: The name of the mutation algorithm used in this instance
        """
        return self._mt_algo

    @mt_algo.setter
    def mt_algo(self, mt_algo) -> None:
        """
        Sets a reference to the mutation function that will be used in this instance. Mutation algorithms
        are found in the algorithms.py module
        :param mt_algo: The mutation algorithm that will be used in this instance
        :return: None
        """
        self._mt_func = mt_algo
        self._mt_algo = mt_algo.__name__

    @property
    def pgen(self) -> bool:
        """
        Returns a flag representing if the number of generation progress will be printed
        :return: A flag representing if the number of generation progress will be printed
        """
        return self._pgen

    @pgen.setter
    def pgen(self, pgen: bool) -> None:
        """
        Sets a flag representing if the number of generation progress will be printed
        :param pgen: A flag representing if the number of generation progress will be printed
        :return: None
        """
        self._is_bool(pgen)
        self._pgen = pgen

    @property
    def agen(self) -> bool:
        """
        Returns a flag representing if generation statistics will be printed
        :return: A flag representing if generation statistics will be printed
        """
        return self._agen

    @agen.setter
    def agen(self, agen: bool) -> None:
        """
        Sets a flag representing if generation statistics will be printed
        :param agen: A flag representing if generation statistics will be printed
        :return: None
        """
        self._is_bool(agen)
        self._agen = agen

    @property
    def plot(self) -> bool:
        """
        Returns a flag representing if generation information should be plotted
        :return: A flag representing if generation information should be plotted
        """
        return self._plot

    @plot.setter
    def plot(self, plot: bool) -> None:
        """
        Sets a flag representing if generation information should be plotted
        :param plot: A flag representing if generation information should be plotted
        :return: None
        """
        self._is_bool(plot)
        self._plot = plot

    @property
    def plot_save_path(self) -> Union[str, None]:
        """
        Returns the file path where the plot image will be saved, or None if not set
        :return: The file path where the plot image will be saved
        """
        return self._plot_save_path

    @plot_save_path.setter
    def plot_save_path(self, plot_save_path: Union[str, None]) -> None:
        """
        Sets the file path where the plot image will be saved. Setting this also enables plotting.
        :param plot_save_path: The file path to save the plot image to
        :return: None
        """
        if plot_save_path is not None and not isinstance(plot_save_path, str):
            raise AttributeError("plot_save_path must be a string or None")
        self._plot_save_path = plot_save_path
        if plot_save_path is not None:
            self._plot = True

    @property
    def verbose_routes(self) -> bool:
        """
        Returns a flag representing if individual routes of the found solution should be included in the result
        :return: A flag representing if individual routes of the found solution should be included in the result
        """
        return self._verbose_routes

    @verbose_routes.setter
    def verbose_routes(self, verbose_routes: bool) -> None:
        """
        Sets a flag representing if individual routes of the found solution should be included in the result
        :param verbose_routes: A flag representing if individual routes of the found solution should be included in
        the result
        :return: None
        """
        self._is_bool(verbose_routes)
        self._verbose_routes = verbose_routes

    @staticmethod
    def _is_probability(value):
        if (not isinstance(value, int)) and (not isinstance(value, float)):
            raise AttributeError("Probability must be numeric")

        if not 0 <= value <= 1:
            raise ValueError('Value must be >= 0 and <= 1')

    @staticmethod
    def _is_int_ge(value: int, ge: int):
        if not isinstance(value, int):
            raise AttributeError("value must be int")

        if not value >= ge:
            raise ValueError(f'Value must be >= {ge}')

    @staticmethod
    def _is_bool(value: bool):
        if not isinstance(value, bool):
            raise ValueError("Value must be bool")
