import random
import numpy as np
from dataclasses import dataclass
from copy import copy
import logging
import timeit

from ..ExtProblem import ExtProblem
from .init_solutions import merge_clusters, radial_init_solution
from .ClusterState import ClusterState, haussdorff_distance
from .crossovers import radial_crossover

logger = logging.getLogger(__name__)


@dataclass
class GAGenerationLog:
    generation: int
    population: list[ClusterState]

def build_rank_probs(p_first: float, size: int) -> np.ndarray:
    """Builds a probability distribution over ranks, where the first rank has probability p_first,
    and the rest decrease geometrically."""
    probs = np.power((1-p_first), np.arange(size))
    probs /= probs.sum()
    return probs

def genetic_algorithm(P: ExtProblem, population_size: int = 50, init_size: int = 10, keep_umerged_init: bool = True, radial_init_samples: int = 20, merge_samples: int = 50, generations: int = 100, log_every: int = 10, reorder_clusters_every: int = 0, reorder_samples_per_node: int = 30, p_first: float = 0.25, p_crossover: float = 0.7, debug = False, seed: int = 42):
    P.times_dict = {}

    ELITISM_NUM = 2
    np.random.seed(seed)

    logging.debug(f"Generating initial population, {init_size} solutions with {radial_init_samples} radial samples")
    baseline_sols = [radial_init_solution(P, num_samples=radial_init_samples)]
    while len(baseline_sols) < init_size:
        min_clusters = np.array([0] + [len(cs.state) for cs in baseline_sols]).mean().astype(int)
        max_clusters = np.array([len(cs.state) for cs in baseline_sols] + [P.graph.number_of_nodes()-1]).mean().astype(int)
        samples = max(int(radial_init_samples * (max_clusters - min_clusters + 1) / (P.graph.number_of_nodes()-1)), 2)

        baseline_sols.append(radial_init_solution(P, num_samples=samples, min_clusters=min_clusters, max_clusters=max_clusters))
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

    population: list[ClusterState] = copy(baseline_sols)

    def rank_pop(pop: list[ClusterState]):
        ranked_pop = list(pop)
        # pop_cost = np.array([cs.cost(P) for cs in ranked_pop])
        pop_cost = np.zeros(len(ranked_pop), dtype=np.float32)
        for i, cs in enumerate(ranked_pop):
            cost = cs.cost(P)
            pop_cost[i] = cost
        sort_idx = np.argsort(pop_cost)
        ranked_pop: list[ClusterState] = [ranked_pop[i] for i in sort_idx]
        return ranked_pop, pop_cost[sort_idx]

    best_cost = float("inf")
    best_generation = -1

    generations_log: list[GAGenerationLog] | None = [] if debug else None

    probs = build_rank_probs(p_first, population_size)

    for gen in range(generations):
        # t = (np.sin(gen / 30) + 1) / 2

        time = timeit.default_timer()
        population, pop_cost = rank_pop(population)
        P.times_dict['cost'] = P.times_dict.get('cost', 0.0) + timeit.default_timer() - time

        if pop_cost[0] < best_cost:
            best_cost = pop_cost[0]
            best_generation = gen

        # next_population: list[ClusterState] = population[:ELITISM_NUM]
        next_population: list[ClusterState] = []

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
                    p=probs[:len(population)]/np.sum(probs[:len(population)]),
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
                    p=probs[:len(population)]/np.sum(probs[:len(population)]),
                )[0]
                child = copy(child)

            time = timeit.default_timer()
            child.mutate_random(P)
            # mut_child = copy(child)
            # mut_child.mutate_random(P)
            # for _ in range(1):  # Number of mutations
            #     mut = copy(child)
            #     mut.mutate_random(P)
            #     if mut.cost(P) < mut_child.cost(P):
            #         mut_child = mut
            P.times_dict['mutation'] = P.times_dict.get('mutation', 0.0) + timeit.default_timer() - time
            next_population.append(child)

        next_population = list(set(next_population).union(set(population)))
        next_population, _ = rank_pop(next_population)
        probs = build_rank_probs(0.3, len(next_population))
        # population = population[:ELITISM_NUM] + np.random.choice(
        #     next_population, #type: ignore
        #     size=(population_size-ELITISM_NUM),
        #     p=probs,
        #     replace=False,
        # ).tolist()
        # survivors = []
        # while len(survivors) < population_size:

        population = next_population[:population_size]
            

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

    ranked_pop = list(population)
    pop_cost = np.array([cs.cost(P) for cs in ranked_pop])
    sort_idx = np.argsort(pop_cost)
    ranked_pop = [ranked_pop[i] for i in sort_idx]
    pop_cost = pop_cost[sort_idx]

    
    return ranked_pop, pop_cost, generations_log