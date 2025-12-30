import numpy as np
from dataclasses import dataclass
from copy import copy
import logging
import timeit

from ..ExtProblem import ExtProblem
from .init_solutions import merge_clusters, radial_init_solution
from .ClusterState import ClusterState
from .crossovers import radial_crossover

logger = logging.getLogger(__name__)


@dataclass
class GAGenerationLog:
    generation: int
    population: list[ClusterState]


def genetic_algorithm(P: ExtProblem, population_size: int = 50, init_size: int = 10, keep_umerged_init: bool = True, radial_init_samples: int = 20, merge_samples: int = 50, generations: int = 100, log_every: int = 10, reorder_clusters_every: int = 0, reorder_samples_per_node: int = 30, p_first: float = 0.25, p_crossover: float = 0.7, debug = False, seed: int = 42):
    P.times_dict = {}

    ELITISM_NUM = 2
    np.random.seed(seed)

    # PARENTS_NUM = np.floor(population_size * 0.2).astype(int)
    # N = P.graph.number_of_nodes() - 1  # excluding depot
    # cls = cluster_solve(P, merge_clusters_method="sample", sample_loops=20, min_clusters=5)
    # first_state = [[n] for n in range(1, P.graph.number_of_nodes())]

    # baseline_sol = baseline_init_solution(P)
    # merged_sol = merge_clusters(P, baseline_sol, method="sample", sample_loops=20)

    logging.debug(f"Generating initial population, {init_size} solutions with {radial_init_samples} radial samples")
    baseline_sols = [radial_init_solution(P, num_samples=radial_init_samples) for _ in range(init_size)]
    if merge_samples > 0:
        logging.debug(f"Merging initial solutions with {merge_samples} sample merges")
        merged_sols = [
            merge_clusters(P, baseline_sol, method="sample", sample_loops=merge_samples)
            for baseline_sol in baseline_sols
        ]
        if keep_umerged_init:
            baseline_sols += merged_sols
        else:
            baseline_sols = merged_sols
    
    baseline_sols_costs = np.array([sol.cost(P) for sol in baseline_sols])
    logging.info(f"Generated initial solutions with costs: {baseline_sols_costs.min():.3f} - {baseline_sols_costs.max():.3f} (avg: {baseline_sols_costs.mean():.3f})")

    population = copy(baseline_sols)
    while len(population) < population_size:
        population += [copy(cls) for cls in baseline_sols]
    population = population[:population_size]

    best_cost = float("inf")
    best_generation = -1

    generations_log: list[GAGenerationLog] | None = [] if debug else None

    probs = np.power((1-p_first), np.arange(population_size))
    probs /= probs.sum()
    print(probs)

    for gen in range(generations):
        # t = (np.sin(gen / 30) + 1) / 2

        time = timeit.default_timer()
        pop_cost = np.array([cs.cost(P) for cs in population])
        sorted_indices = np.argsort(pop_cost)
        population: list[ClusterState] = [population[i] for i in sorted_indices]
        pop_cost = pop_cost[sorted_indices]
        P.times_dict['cost'] = P.times_dict.get('cost', 0.0) + timeit.default_timer() - time

        if pop_cost[0] < best_cost:
            best_cost = pop_cost[0]
            best_generation = gen

        next_population: list[ClusterState] = [ copy(cs) for cs in population[:ELITISM_NUM]]

        # # Map pop-cost into 0-1 range
        # pop_cost = pop_cost / (pop_cost[0] + 1e-6) - 1
        # pop_cost /= (pop_cost[-1] + 1e-6)
        # pop_cost = 1.0 - pop_cost  # Invert so that lower cost = higher value

        # probs = np.power(pop_cost, 0.7)
        # probs /= probs.sum()

        # def measure_diversity(cs: ClusterState) -> float:
        #     """Returns a diversity measure from 0.0 to 1.0"""
        #     diversity = 0.0
        #     samples = min(len(next_population), 20)
        #     for other_cs in next_population[-20:]: # type: ignore
        #         # diversity += 1.0 - rand_index(cs, other_cs)
        #         diversity += haussdorff_distance(P, cs, other_cs)

        #     return diversity / samples

        while len(next_population) < population_size:
            # pop_diversity = np.array([measure_diversity(cs) for cs in population])
            # pop_diversity /= pop_diversity.max() + 1e-6

            # probs = np.pow(np.square(pop_cost) + 5*np.square(pop_diversity), 3)
            # probs /= probs.sum()


            child: ClusterState

            if np.random.rand() < p_crossover:
                # Pick two parents
                p1, p2 = np.random.choice(
                    population, #type: ignore
                    size=2,
                    p=probs,
                    replace=False,
                )
                time = timeit.default_timer()
                child = radial_crossover(P, p1, p2, reorder_samples_per_node=reorder_samples_per_node)
                P.times_dict['crossover'] = P.times_dict.get('crossover', 0.0) + timeit.default_timer() - time
            else:
                # Clone a parent
                child = np.random.choice(
                    population, #type: ignore
                    size=1,
                    p=probs,
                )[0]
                child = copy(child)

            time = timeit.default_timer()
            # if random.getrandbits(1) == 1:
            child.mutate_random(P)
            P.times_dict['mutation'] = P.times_dict.get('mutation', 0.0) + timeit.default_timer() - time
            next_population.append(child)

        population = next_population

        if reorder_clusters_every > 0 and gen % reorder_clusters_every == 0:
            time = timeit.default_timer()
            for cs in population:
                cs.mutate_reorder_all_clusters(P, max_len_perm=5, samples_per_node=reorder_samples_per_node)
            P.times_dict['reorder_clusters'] = P.times_dict.get('reorder_clusters', 0.0) + timeit.default_timer() - time

        if gen % log_every == 0:
            if generations_log is not None:
                generations_log.append(GAGenerationLog(gen, [copy(cs) for cs in population]))
            logging.info(
                f"Generation {gen}: best cost = {best_cost:.3f} at gen {best_generation}"
            )

    pop_cost = np.array([cs.cost(P) for cs in population])
    sort_idx = np.argsort(pop_cost)
    population = [population[i] for i in sort_idx]
    pop_cost = pop_cost[sort_idx]

    
    return population, pop_cost, generations_log
