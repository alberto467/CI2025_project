import random
import networkx as nx
import numpy as np
import logging
from itertools import permutations
from copy import deepcopy, copy

from .problem import Problem


def softmax(x, temp=1.0):
    e_x = np.exp((x - np.max(x)) / temp)
    return e_x / e_x.sum()


def find_grand_tour(P: Problem, trials: int = 10):
    shortest_dists = nx.floyd_warshall_numpy(P.graph, weight="dist")

    def generate_greedy_tour(nodes: set[int]):
        unvisited = nodes.copy()
        unvisited.discard(0)  # Start from the depot
        tour = [0]
        total_dist = 0.0

        while unvisited:
            unvisited_list = list(unvisited)
            dists = shortest_dists[tour[-1], unvisited_list]
            probs = softmax(-dists, temp=0.7)

            next_city_idx = int(
                np.random.choice(np.arange(len(unvisited_list)), size=1, p=probs)[0]
            )
            next_city = unvisited_list[next_city_idx]
            total_dist += dists[next_city_idx]

            tour += nx.shortest_path(P.graph, tour[-1], next_city, weight="dist")[1:]
            unvisited.remove(next_city)

        # Return to the depot
        back_home = nx.shortest_path(P.graph, tour[-1], 0, weight="dist")
        total_dist += shortest_dists[tour[-1], 0]
        tour += back_home[1:]

        return tour, total_dist

    def catch_gold(tour):
        assert tour[0] == 0 and tour[-1] == 0, "Tour must start and end at the depot"

        nodes_left = set(P.graph.nodes)
        nodes_left.remove(0)

        total_cost = 0.0
        current_gold = 0.0
        out = [(0, 0)]  # (node, gold collected at this node)
        remaining_gold = np.array([P.graph.nodes[n]["gold"] for n in P.graph.nodes])

        i = -1
        while i < len(tour) - 2:
            i += 1
            u, v = tour[i], tour[i + 1]
            # Handle the depot
            if v == 0:
                total_cost += P.cost([u, v], current_gold)
                current_gold = 0.0
                out.append((v, 0))
                if not nodes_left:
                    break
                continue

            trip_to_home = nx.shortest_path(P.graph, u, 0)
            trip_from_home = nx.shortest_path(P.graph, 0, v)
            trip_back_cost = P.cost(trip_to_home, current_gold) + P.cost(
                trip_from_home, 0.0
            )

            LOOK_AHEAD = 3
            next_current_gold_cost = P.cost(tour[i : i + LOOK_AHEAD], current_gold)
            next_empty_gold_cost = P.cost(tour[i : i + LOOK_AHEAD], 0)
            tripless_lookahead_diff = next_current_gold_cost - next_empty_gold_cost

            if (
                tripless_lookahead_diff > 0
                and trip_back_cost / tripless_lookahead_diff < 0.95
            ):
                # Better to go back to depot now

                for j in range(len(trip_to_home) - 1):
                    u2, v2 = trip_to_home[j], trip_to_home[j + 1]
                    total_cost += P.cost([u2, v2], current_gold)
                    current_gold += remaining_gold[v2]
                    out.append((v2, remaining_gold[v2]))
                    remaining_gold[v2] = 0.0
                    nodes_left.discard(v2)

                current_gold = 0.0

                # We are now at the depot, let's randomly recalculate the next tour
                if not nodes_left:
                    break

                if random.random() < 0:
                    next_tours = [
                        generate_greedy_tour(nodes_left) for _ in range(trials)
                    ]
                    best_tour = min(next_tours, key=lambda x: x[1])[0]
                    i = -1
                    tour = best_tour
                    continue

                for j in range(len(trip_from_home) - 2):
                    u2, v2 = trip_from_home[j], trip_from_home[j + 1]
                    total_cost += P.cost([u2, v2], current_gold)
                    out.append((v2, 0))

            else:
                total_cost += P.cost([u, v], current_gold)

            gold = remaining_gold[v]
            out.append((v, gold))
            current_gold += gold
            remaining_gold[v] = 0.0
            nodes_left.discard(v)

        assert remaining_gold.sum() == 0.0, "Not all gold collected!"

        return out, total_cost

    tours = [generate_greedy_tour(set(P.graph.nodes)) for _ in range(trials)]

    gold_tours = [catch_gold(tour) for tour, _ in tours]
    return tours, gold_tours


class ExtProblem(Problem):
    """
    An extended problem class that precomputes and caches additional data structures
    """

    def __init__(self, P: Problem):
        self._graph = P._graph  # .copy()
        self._alpha = P._alpha
        self._beta = P._beta

        self._add_adj_values()

        self.floyd_mat = nx.floyd_warshall_numpy(self.graph, weight="dist")
        self.adj_floyd_mat = nx.floyd_warshall_numpy(self.graph, weight="adj_dist")

        N = P.graph.number_of_nodes()
        self._shortest_paths_mat = [[None for _ in range(N)] for _ in range(N)]
        self.cache_hit = 0
        self.cache_miss = 0

    def _add_adj_values(self):
        nx.set_node_attributes(
            self._graph,
            {
                n: np.pow(g, self._beta)
                for n, g in nx.get_node_attributes(self._graph, "gold").items()
            },
            "adj_gold",
        )
        nx.set_edge_attributes(
            self._graph,
            {
                (u, v): np.pow(d["dist"], self._beta)
                for u, v, d in self._graph.edges(data=True)
            },
            "adj_dist",
        )

    def cached_shortest_path(self, u: int, v: int) -> list[int]:
        assert u != v, "Shortest path requested between the same node"
        flipped = u > v
        if flipped:
            u, v = v, u

        if self._shortest_paths_mat[u][v] is not None:
            self.cache_hit += 1
            path = self._shortest_paths_mat[u][v]
            if flipped:
                path = list(reversed(path))
            return path

        self.cache_miss += 1
        path = nx.shortest_path(self.graph, u, v, weight="dist")
        self._shortest_paths_mat[u][v] = path
        if flipped:
            path = list(reversed(path))
        return path


# def cluster_cities(P: ExtProblem, k: int):
#     assert k > 0, "Number of clusters must be positive"
#     assert k < P.graph.number_of_nodes(), (
#         "Number of clusters must be less than number of cities"
#     )

#     coords = np.array([P.graph.nodes[n]["pos"] for n in P.graph.nodes if n != 0])
#     gold = np.array([P.graph.nodes[n]["gold"] for n in P.graph.nodes if n != 0])

#     kmeans = KMeans(n_clusters=k, random_state=0).fit(coords, sample_weight=gold)
#     clusters = [[] for _ in range(k)]
#     for idx, label in enumerate(kmeans.labels_):
#         clusters[label].append(idx + 1)  # +1 to skip the depot

#     return clusters, kmeans


def cover_cluster(P: ExtProblem, cluster: set[int]):
    """
    Finds the shortest path that covers all cities in the cluster and returns to the depot.
    """

    if len(cluster) == 1:
        city = list(cluster)[0]
        path_to = nx.shortest_path(P.graph, 0, city, weight="dist")
        path_from = nx.shortest_path(P.graph, city, 0, weight="dist")
        return path_to + path_from[1:]

    if len(cluster) <= 3:
        # Let's find the best permutation
        best_permutation = None
        best_cost = float("inf")
        for perm in permutations(cluster):
            total_cost = 0.0
            for u, v in zip([0] + list(perm), list(perm) + [0]):
                total_cost += P.floyd_mat[u, v]
            if total_cost < best_cost:
                best_cost = total_cost
                best_permutation = perm
        path = [0]
        for u, v in zip([0] + list(best_permutation), list(best_permutation) + [0]):
            sub_path = nx.shortest_path(P.graph, u, v, weight="dist")
            path += sub_path[1:]
        return path

    to_visit = set(cluster)
    path = [0]

    while to_visit:
        current = path[0]
        nearest_city = min(
            to_visit,
            key=lambda city: P.floyd_mat[city, current],
        )
        sub_path = nx.shortest_path(P.graph, nearest_city, current, weight="dist")
        path = sub_path[:-1] + path  # Avoid duplicating the current city
        for n in sub_path[:-1]:
            to_visit.discard(n)

    from_home = nx.shortest_path(P.graph, 0, path[0], weight="dist")
    path = from_home[:-1] + path
    return path


def cover_all_clusters(P: ExtProblem, floyd_mat: np.array, clusters: list[list[int]]):
    full_path = [0]
    for cluster in clusters:
        cluster_path = cover_cluster(P, floyd_mat, cluster)
        full_path += cluster_path[1:]
    return full_path


def catch_gold(P: ExtProblem, tour: list[int], cluster: set[int]):
    """
    Given a tour and a cluster of nodes, returns the gold collection plan and total cost.
    """
    assert tour[0] == 0 and tour[-1] == 0, "Tour must start and end at the depot"
    assert set(tour).issuperset(cluster), "Tour must cover all nodes in the cluster"

    nodes_left = cluster.copy()

    total_cost = 0.0
    current_gold = 0.0
    out = [(0, 0)]  # (node, gold collected at this node)
    remaining_gold = np.array(
        [P.graph.nodes[n]["gold"] if n in cluster else 0.0 for n in P.graph.nodes]
    )
    logging.debug(
        f"Catching gold in cluster {cluster} - total gold: {remaining_gold.sum()}"
    )

    i = -1
    while i < len(tour) - 2:
        i += 1
        u, v = tour[i], tour[i + 1]
        # Handle the depot
        if v == 0:
            total_cost += P.cost([u, v], current_gold)
            current_gold = 0.0
            out.append((v, 0))
            if not nodes_left:
                break
            continue

        # trip_to_home = nx.shortest_path(P.graph, u, 0)
        # trip_from_home = nx.shortest_path(P.graph, 0, v)
        # trip_back_cost = P.cost(trip_to_home, current_gold) + P.cost(
        #     trip_from_home, 0.0
        # )

        # LOOK_AHEAD = 3
        # next_current_gold_cost = P.cost(tour[i : i + LOOK_AHEAD], current_gold)
        # next_empty_gold_cost = P.cost(tour[i : i + LOOK_AHEAD], 0)
        # tripless_lookahead_diff = next_current_gold_cost - next_empty_gold_cost

        # if (
        #     False
        #     and tripless_lookahead_diff > 0
        #     and trip_back_cost / tripless_lookahead_diff < 0.95
        # ):
        #     # Better to go back to depot now

        #     for j in range(len(trip_to_home) - 1):
        #         u2, v2 = trip_to_home[j], trip_to_home[j + 1]
        #         total_cost += P.cost([u2, v2], current_gold)
        #         current_gold += remaining_gold[v2]
        #         out.append((v2, remaining_gold[v2]))
        #         remaining_gold[v2] = 0.0
        #         nodes_left.discard(v2)

        #     current_gold = 0.0

        #     # We are now at the depot, let's randomly recalculate the next tour
        #     if not nodes_left:
        #         break

        #     for j in range(len(trip_from_home) - 2):
        #         u2, v2 = trip_from_home[j], trip_from_home[j + 1]
        #         total_cost += P.cost([u2, v2], current_gold)
        #         out.append((v2, 0))

        # else:
        total_cost += P.cost([u, v], current_gold)

        take_gold = v in cluster and v not in set(tour[i + 3 :])

        gold_taken = remaining_gold[v] if take_gold else 0.0
        # logging.debug(f"At node {v}, take_gold={take_gold}, gold_taken={gold_taken}")

        out.append((v, gold_taken))
        current_gold += gold_taken
        remaining_gold[v] -= gold_taken
        nodes_left.discard(v)

    assert remaining_gold.sum() == 0.0, "Not all gold collected!"

    return out, total_cost


class Cluster:
    def __init__(self, nodes: list[int]):
        self.nodes = nodes
        # self.path = Cluster._get_path(nodes, P)
        # self.gold_path, self.cost = catch_gold(P, self.path, self.nodes)

    def get_path_cost(self, P: ExtProblem):
        total_cost = 0.0
        for u, v in zip([0] + self.nodes, self.nodes + [0]):
            total_cost += P.floyd_mat[u, v]
        return total_cost

    def get_gold_cost(self, P: ExtProblem):
        gold_cost = 0.0
        current_gold = 0.0
        for u, v in zip([0] + self.nodes, self.nodes + [0]):
            gold_cost += (
                P.floyd_mat[u, v]
                + np.pow(P.alpha * current_gold, P.beta) * P.adj_floyd_mat[u, v]
            )
            current_gold += P.graph.nodes[v]["gold"]

        return gold_cost

    def get_path(self, P: ExtProblem):
        # if self._path is not None:
        #     return self._path

        full_path = [0]
        for u, v in zip([0] + self.nodes, self.nodes + [0]):
            sub_path = P.cached_shortest_path(u, v)
            full_path += sub_path[1:]

        # self._path = full_path

        return full_path

    def get_gold_path(self, P: ExtProblem):
        full_path = [(0, 0.0)]
        for u, v in zip([0] + self.nodes, self.nodes + [0]):
            sub_path = P.cached_shortest_path(u, v)
            full_path += [(n, 0.0) for n in sub_path[1:-1]]
            full_path.append((v, P.graph.nodes[v]["gold"] if v != 0 else 0.0))

        return full_path

    def is_subpath(self, P: ExtProblem, other: "Cluster"):
        self_path = self.get_path(P)
        other_path = other.get_path(P)
        if len(self_path) > len(other_path):
            return False
        return set(self_path).issubset(set(other_path))

    def is_superpath(self, P: ExtProblem, other: "Cluster"):
        self_path = self.get_path(P)
        other_path = other.get_path(P)
        if len(self_path) < len(other_path):
            return False
        return set(self_path).issuperset(set(other_path))

    def merge(self, P: ExtProblem, other: "Cluster"):
        join = set().union(set(self.nodes), set(other.nodes))
        best_cost = float("inf")
        best_permutation = None
        for perm in permutations(list(join), len(join)):
            perm = list(perm)
            merged_cluster = Cluster(perm)
            merged_cluster_cost = merged_cluster.get_gold_cost(P)
            if merged_cluster_cost < best_cost:
                best_cost = merged_cluster_cost
                best_permutation = perm
        return Cluster(best_permutation), best_cost

    def shuffle_nodes(self):
        random.shuffle(self.nodes)

    def __str__(self):
        return f"Cluster(nodes={self.nodes})"

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        return Cluster(self.nodes.copy())


def cluster_solve(P: ExtProblem):
    clusters = [Cluster([i]) for i in range(1, P.graph.number_of_nodes())]
    # heapify(clusters)

    while len(clusters) > 1:
        best_saving = 0
        best_pair = None
        best_merged = None
        for (i, cli), (j, clj) in permutations(enumerate(clusters), 2):
            if i == j:
                continue
            if clj.is_superpath(P, cli):
                merged, merged_cost = clj.merge(P, cli)
                saved = cli.get_gold_cost(P) + clj.get_gold_cost(P) - merged_cost
                if saved > best_saving:
                    best_saving = saved
                    best_pair = (i, j)
                    best_merged = merged
        if best_merged is not None:
            logging.debug(
                f"Merging clusters {clusters[i]} and {clusters[j]} with saving {best_saving}"
            )
            i, j = best_pair
            clusters[i] = best_merged
            clusters.pop(j)
        else:
            break

    logging.debug(f"Final clusters ({len(clusters)}): {clusters}")

    cluster_state = ClusterState(clusters)
    cluster_state.verify(P)

    _, gold_path, _ = cluster_state.build_paths(P)

    # full_path = [0]
    # gold_path = [(0, 0)]
    # for cl in clusters:
    #     full_path += cl.path[1:]
    #     gold_path += cl.gold_path[1:]

    return gold_path, clusters, cluster_state


class ClusterState:
    def __init__(self, state: list[Cluster]):
        self.state = state
        # self.cluster_paths = [None for _ in state]

    def verify(self, P: ExtProblem):
        full_set = set()
        for cluster in self.state:
            # TODO: it might be interesting to allow overlapping
            # when it is useful to split the gold collection
            assert full_set.isdisjoint(set(cluster.nodes)), "Clusters overlap!"
            full_set.update(cluster.nodes)
        assert full_set == set(range(1, P.graph.number_of_nodes())), (
            "Clusters does not cover all nodes!"
        )

    def build_paths(self, P: ExtProblem):
        full_path = [0]
        gold_path = [(0, 0.0)]
        total_gold_cost = 0.0

        for cluster in self.state:
            full_path += cluster.get_path(P)[1:]
            gold_path += cluster.get_gold_path(P)[1:]
            total_gold_cost += cluster.get_gold_cost(P)

        return full_path, gold_path, total_gold_cost

    def cost(self, P: ExtProblem):
        return sum(cl.get_gold_cost(P) for cl in self.state)

    def mutate_random_swap(self, P: ExtProblem):
        if len(self.state) < 2:
            return

        c1, c2 = random.sample(range(len(self.state)), 2)

        n1 = random.choice(self.state[c1].nodes)
        n2 = random.choice(self.state[c2].nodes)

        self.state[c1].nodes.remove(n1)
        self.state[c1].nodes.append(n2)
        if len(self.state[c1].nodes) == 0:
            self.state.pop(c1)

        self.state[c2].nodes.remove(n2)
        self.state[c2].nodes.append(n1)
        if len(self.state[c2].nodes) == 0:
            self.state.pop(c2)

    def mutate_random_move(self, P: ExtProblem):
        if len(self.state) < 1:
            return

        c1 = random.choice(range(len(self.state)))
        assert len(self.state[c1].nodes) > 0, "Cannot move from empty cluster"
        # self.cluster_paths[c1] = None

        n1 = random.choice(self.state[c1].nodes)
        self.state[c1].nodes.remove(n1)
        if len(self.state[c1].nodes) == 0:
            self.state.pop(c1)
            # self.cluster_paths.pop(c1)

        new_cluster = random.random() < 0.3
        if new_cluster:
            self.state.append(Cluster([n1]))
            # self.cluster_paths.append(None)
        else:
            c2 = random.choice(range(len(self.state)))
            # self.cluster_paths[c2] = None
            i2 = random.randint(0, len(self.state[c2].nodes))
            self.state[c2].nodes.insert(i2, n1)

    def mutate_merge_clusters(self, P: ExtProblem):
        if len(self.state) < 2:
            return

        c1, c2 = random.sample(range(len(self.state)), 2)
        cl1, cl2 = self.state[c1], self.state[c2]

        merged, _ = cl1.merge(P, cl2)

        self.state[c1] = merged
        # self.cluster_paths[c1] = None
        self.state.pop(c2)
        # self.cluster_paths.pop(c2)

    def mutate_shuffle_cluster(self, P: ExtProblem):
        cl = random.choice(self.state)
        cl.shuffle_nodes()
        # self.cluster_paths[c1] = None

    def mutate_random(self, P: ExtProblem):
        r = random.random()
        if r < 0.8:
            self.mutate_merge_clusters(P)
        elif r < 0.9:
            self.mutate_random_swap(P)
        else:
            self.mutate_random_move(P)

    def __copy__(self):
        return ClusterState([copy(cl) for cl in self.state])


def genetic_algorithm(P: ExtProblem, population_size: int = 50, generations: int = 100):
    ELITISM_NUM = 3
    # PARENTS_NUM = np.floor(population_size * 0.2).astype(int)
    _, _, cl_state = cluster_solve(P)
    # first_state = [[n] for n in range(1, P.graph.number_of_nodes())]

    population = [copy(cl_state) for _ in range(population_size)]
    for gen in range(generations):
        pop_cost = np.array([cs.cost(P) for cs in population])
        sorted_indices = np.argsort(pop_cost)

        population = [population[i] for i in sorted_indices]
        pop_cost = pop_cost[sorted_indices]
        probs = np.pow(pop_cost, -1)
        probs /= probs.sum()

        next_population = population[:ELITISM_NUM]
        parents = np.random.choice(
            np.arange(population_size),
            size=(population_size - ELITISM_NUM),
            p=probs,
        )

        for parent_i in parents:
            child = copy(population[parent_i])
            # child.mutate_merge_clusters(P)

            child.mutate_random(P)
            # child.verify(P)

            next_population.append(child)

        # _, _, best_gold_cost = population[0].build_paths(P)

        population = next_population
        logging.info(
            f"Generation {gen}: best cost = {population[0].cost(P)} with state {population[0].state} - cache hits: {P.cache_hit}, misses: {P.cache_miss}"
        )
