from dataclasses import dataclass, field
import heapq
import time
import numpy as np
import networkx as nx
from .problem import Problem


@dataclass(order=True)
class SearchState:
    f_score: float
    path: list[tuple[int, float]] = field(compare=False)
    cum_dist: float = field(compare=False)
    remaining_gold: np.ndarray = field(compare=False)
    current_gold: float = field(compare=False)

    def last_node(self) -> int:
        return self.path[-1][0]


H_len_power = 0.65
H_scale = 40


# Keep variables global for debugging
open_heap = []
g_scores = None
best_fitness = np.inf
best_path = None
best_iteration = None


def MultiStar(
    problem: Problem,
    heap_size_limit: int = 100_000,
    heap_size_keep: int = 80_000,
    iteration_limit: int = 10_000,
    early_stop_fitness: float = None,
    random_takes: int = 2,
    print_every: int = 100,
):
    rng = np.random.default_rng(seed=42)
    N = problem.graph.number_of_nodes()
    avg_step_dist = 0.5 + (problem.alpha * 0.5 * 500.5) ** problem.beta

    precompute_paths_to_base: list[tuple[list[int], float]] = [([], 0)]
    for n in range(1, N):
        path = nx.shortest_path(problem.graph, source=n, target=0, weight="dist")
        dist = nx.path_weight(problem.graph, path, weight="dist")
        precompute_paths_to_base.append((path, dist))
    precompute_paths_from_base: list[tuple[list[int], float]] = [([], 0)]
    for n in range(1, N):
        path = nx.shortest_path(problem.graph, source=0, target=n, weight="dist")
        dist = nx.path_weight(problem.graph, path, weight="dist")
        precompute_paths_from_base.append((path, dist))

    # Heuristic function
    # Deeply intertwined with the queue cleanup strategy
    def H(
        current: SearchState,
        next_node: int,
        cum_dist: float,
        remaining_gold: np.ndarray,
        amount: float,
    ) -> float:
        # Note that the distance between cities is always between 0 and 1, and gold in each city is between 1 and 1000

        # ne_count = len(list(problem.graph.neighbors(next_node)))

        remaining_gold_cities = np.count_nonzero(remaining_gold)

        return (
            cum_dist
            # + remaining_gold.sum()
            + remaining_gold_cities**1.5 * avg_step_dist / N * 18
            # - (ne_count - 1) / N * avg_step_dist * 2
            + problem.cost(
                precompute_paths_to_base[next_node][0],
                current.current_gold + amount,
            )
            / N
            * 12
            - (
                ((current.current_gold + amount) * problem.alpha) ** (problem.beta) * 1
                if next_node == 0
                else 0
            )
            # - amount
            # if (amount > 0 and remaining_gold[next_node] == 0)
            # else 0
        )

    # problem_std = problem[np.isfinite(problem)].std()

    # The data structure for open should ideally allow
    # O(1) retrieval of the minimum f_score node,
    # and O(1) check if it contains a node by its index.
    global open_heap, best_fitness, best_path, best_iteration
    open_heap = [
        SearchState(
            0,
            [(0, 0.0)],
            0,
            np.array([n["gold"] for n in problem.graph.nodes.values()]),
            0.0,
        )
    ]
    heapq.heapify(open_heap)

    best_fitness = np.inf
    best_path = None

    i = 0

    while open_heap and i < iteration_limit:
        current = heapq.heappop(open_heap)

        if current.cum_dist >= best_fitness:
            continue

        for ne in problem.graph.neighbors(current.last_node()):
            if ne == 0:
                # Unload current gold
                new_path = current.path + [(0, 0)]
                cum_dist = current.cum_dist + problem.cost(
                    [current.last_node(), 0], current.current_gold
                )
                f_score = H(
                    current,
                    ne,
                    cum_dist,
                    current.remaining_gold,
                    0,
                )
                heapq.heappush(
                    open_heap,
                    SearchState(
                        f_score,
                        new_path,
                        cum_dist,
                        current.remaining_gold,
                        0.0,
                    ),
                )

                continue

            amounts = [0]
            if current.remaining_gold[ne] > 0:
                # How much gold can we take from this city? multiple branches: take 0, n random take, take all
                amounts = [0, current.remaining_gold[ne]]
                if current.remaining_gold[ne] > 2 and random_takes > 0:
                    amounts += list(
                        rng.uniform(
                            1, current.remaining_gold[ne] - 1, size=random_takes
                        )
                    )

            for amount in amounts:
                step_dist = problem.cost(
                    [current.last_node(), ne], current.current_gold
                )
                new_current_gold = current.current_gold + amount
                new_path = current.path + [(int(ne), amount)]

                remaining_gold = current.remaining_gold.copy()
                remaining_gold[ne] -= amount

                home_trip_dist = problem.cost(
                    precompute_paths_to_base[ne][0], new_current_gold
                ) + problem.cost(precompute_paths_from_base[ne][0], 0.0)
                if home_trip_dist < step_dist:
                    new_current_gold = 0.0
                    new_path = current.path + [
                        (n, 0.0)
                        for n in precompute_paths_to_base[current.last_node()][0][1:]
                        + precompute_paths_from_base[ne][0][1:]
                    ]
                    step_dist = home_trip_dist

                    # Maybe pick up gold on the way back to base?
                    last_node_before_base = precompute_paths_to_base[
                        current.last_node()
                    ][0][-2]
                    if remaining_gold[last_node_before_base] > 0:
                        # Pick up all gold on the way back to base
                        without_cost = problem.cost(
                            [last_node_before_base, 0], new_current_gold
                        )
                        amount_back = remaining_gold[last_node_before_base]
                        with_cost = problem.cost(
                            [last_node_before_base, 0],
                            new_current_gold + amount_back,
                        )
                        min_pickup_cost = problem.cost(
                            precompute_paths_to_base[last_node_before_base][0],
                            amount_back,
                        )
                        if with_cost - without_cost <= min_pickup_cost * 0.9:
                            new_current_gold += amount_back
                            remaining_gold[last_node_before_base] = 0
                            new_path = current.path + [
                                (n, 0.0)
                                for n in precompute_paths_to_base[current.last_node()][
                                    0
                                ][1:-2]
                                + [(last_node_before_base, amount_back)]
                                + precompute_paths_from_base[ne][0]
                            ]
                            step_dist += with_cost - without_cost

                cum_dist = current.cum_dist + step_dist

                if remaining_gold[ne] == 0 and remaining_gold.sum() == 0:
                    # Return to (0, 0)
                    return_path = precompute_paths_to_base[ne][0]
                    new_path = current.path + [(n, 0) for n in return_path[1:]]
                    cum_dist = current.cum_dist + problem.cost(
                        return_path, new_current_gold
                    )
                    # Completed solution
                    if cum_dist < best_fitness:
                        best_fitness = cum_dist
                        best_path = new_path
                        best_iteration = i
                        # print(
                        #     f"New best path with fitness {best_fitness:.8f} at iteration {i}, path len {len(best_path)}"
                        # )
                        if (
                            early_stop_fitness is not None
                            and best_fitness <= early_stop_fitness
                        ):
                            print(
                                f"Early stopping at iteration {i} with fitness {best_fitness:.8f}"
                            )
                            return best_path, best_fitness, best_iteration
                    continue

                f_score = H(current, ne, cum_dist, remaining_gold, amount)
                heapq.heappush(
                    open_heap,
                    SearchState(
                        f_score,
                        new_path,
                        cum_dist,
                        remaining_gold,
                        new_current_gold,
                    ),
                )

        i += 1
        if i % print_every == 0:
            heap_lengths = [len(s.path) + 1 for s in open_heap]
            heap_f_scores = [s.f_score for s in open_heap]
            heap_remaining_nonzero = [
                np.count_nonzero(s.remaining_gold) for s in open_heap
            ]
            heap_size = len(open_heap)
            print(
                f"Iteration {i} | Heap size: {heap_size} | "
                f"Heap lengths: {np.min(heap_lengths)}/{np.mean(heap_lengths):.1f}/{np.max(heap_lengths)} | "
                f"Heap remaining gold-cities: {np.min(heap_remaining_nonzero)}/{np.mean(heap_remaining_nonzero):.1f}/{np.max(heap_remaining_nonzero)} | "
                f"Heap f_scores: {np.min(heap_f_scores):.2f}/{np.mean(heap_f_scores):.2f}/{np.max(heap_f_scores):.2f} | "
                f"Best path: {best_fitness:.8f} with len {len(best_path) if best_path else None} at iter {best_iteration} "
            )

        if len(open_heap) > heap_size_limit:
            # Benchmark how long this takes
            curr_heap_size = len(open_heap)
            start_time = time.time()
            new_heap = []
            while len(new_heap) < heap_size_keep and open_heap:
                s = heapq.heappop(open_heap)
                if np.isfinite(best_fitness) and s.cum_dist >= best_fitness:
                    continue
                heapq.heappush(new_heap, s)
            end_time = time.time()
            print(
                f"Pruned heap from size {curr_heap_size} to {len(new_heap)} in {end_time - start_time:.2f} seconds"
            )
            open_heap = new_heap

    return best_path, best_fitness, best_iteration


# P = Problem(num_cities=50, density=0.2)
# print(f"Baseline: {P.baseline()}")
# print(f"Lower bound: {P.lower_bound()}")
# l = MultiStar(P, iteration_limit=100_000, print_every=500, random_takes=10)
# print(l)
