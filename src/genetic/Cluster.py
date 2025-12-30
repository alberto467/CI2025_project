import numpy as np
from itertools import permutations
import logging
import timeit

from ..ExtProblem import ExtProblem

logger = logging.getLogger(__name__)


class Cluster:
    _nodes: tuple[int, ...]
    _gold_cost: float | None
    _centroid: np.ndarray | None

    def __init__(self, nodes: tuple[int, ...] | list[int]):
        self._nodes = nodes if isinstance(nodes, tuple) else tuple(nodes)
        # self.path = Cluster._get_path(nodes, P)
        # self.gold_path, self.cost = catch_gold(P, self.path, self.nodes)

        # Keep some cached values
        self._gold_cost = None
        self._centroid = None
    @property
    def nodes(self) -> list[int]:
        return list(self._nodes)

    def nodes_set(self) -> set[int]:
        return set(self._nodes)

    def is_empty(self) -> bool:
        return len(self._nodes) == 0

    def get_path_cost(self, P: ExtProblem) -> float:
        regular_dists = P.floyd_mat[[0] + self.nodes, self.nodes + [0]]
        return np.sum(regular_dists)

    def get_gold_cost(self, P: ExtProblem) -> float:
        if self._gold_cost is not None:
            return self._gold_cost

        adj_dists = P.adj_floyd_mat[[0] + self.nodes, self.nodes + [0]]
        adj_steps_gold = np.pow(np.cumsum(
            [0.0] + P.gold_amounts[self.nodes].tolist() # type: ignore
        ), P.beta)

        gold_cost = self.get_path_cost(P) + np.pow(P.alpha, P.beta) * np.sum(adj_dists * adj_steps_gold)
        self._gold_cost = gold_cost
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
            full_path.append((v, P.gold_amounts[v]))

        return full_path

    def is_subpath(self, P: ExtProblem, other: "Cluster"):
        self_path = self.get_path(P)
        other_path = other.get_path(P)
        if len(self_path) > len(other_path):
            return False
        return set(self_path).issubset(set(other_path))

    def is_superpath(self, P: ExtProblem, other: "Cluster"):
        self_path = self.get_path(P)
        if len(self_path) < len(other.nodes):
            return False
        return set(self_path).issuperset(set(other.nodes))

    def get_centroid(self, P: ExtProblem) -> np.ndarray:
        if self._centroid is not None:
            return self._centroid

        self._centroid = P.positions[self.nodes].mean(axis=0)
        return self._centroid # type: ignore

    @staticmethod
    def find_best_cluster(P: ExtProblem, nodes: set[int], max_len_perm: int = 5, samples_per_node: int = 30):
        timer = timeit.default_timer()
        best_cluster = None
        nodes_list = list(nodes)

        # it = None
        # if len(nodes) <= max_len_perm:
        #     it = (list(t) for t in permutations(nodes_list, len(nodes_list)))
        # else:
        #     it = (list(np.random.permutation(nodes_list)) for _ in range(samples))

        # for perm in it:
        #     new_cl = Cluster(perm)
        #     if best_cluster is None or new_cl.get_gold_cost(P) < best_cluster.get_gold_cost(P):
        #         best_cluster = new_cl

        def find_greedy():
            unvisited = set(nodes_list)
            path = [0] # End at depot (build backwards)
            gold_left = np.sum(P.gold_amounts[list(unvisited)])

            while unvisited:
                unvis_list = list(unvisited)
                current = path[-1]
                dists = P.floyd_mat[unvis_list, current]
                adj_dists = P.adj_floyd_mat[unvis_list, current]

                best_cost = float("inf")
                best_i = None

                for i, n in enumerate(unvis_list):
                    gold_carried = gold_left - P.gold_amounts[n]
                    cost = dists[i] + np.pow(P.alpha * gold_carried, P.beta) * adj_dists[i]
                    if cost < best_cost:
                        best_cost = cost
                        best_i = i

                assert best_i is not None
                path.insert(0, unvis_list[best_i])
                gold_left -= P.gold_amounts[unvis_list[best_i]]
                unvisited.remove(unvis_list[best_i])
            return path[:-1]  # Remove depot at end

        if len(nodes) <= max_len_perm:
            for perm in permutations(nodes_list, len(nodes_list)):
                new_cl = Cluster(list(perm))
                if best_cluster is None or new_cl.get_gold_cost(P) < best_cluster.get_gold_cost(P):
                    best_cluster = new_cl

        else:
            nodes_list = find_greedy()

            for a, b in np.random.randint(0, len(nodes_list), size=(samples_per_node * len(nodes_list), 2), dtype=np.int32):
                if a == b:
                    continue
                # Try swapping a and b
                nodes_list[a], nodes_list[b] = nodes_list[b], nodes_list[a]

                new_cl = Cluster(nodes_list)
                if best_cluster is None or new_cl.get_gold_cost(P) < best_cluster.get_gold_cost(P):
                    best_cluster = new_cl
                else:
                    # Revert swap
                    nodes_list[a], nodes_list[b] = nodes_list[b], nodes_list[a]

        assert best_cluster is not None
        time = timeit.default_timer() - timer
        P.times_dict["find_best_cluster"] = P.times_dict.get("find_best_cluster", 0.0) + time
        return best_cluster

    def insert_node_at_best_position(self, P: ExtProblem, node: int):
        best_cluster = None
        for i in range(len(self.nodes) + 1):
            new_nodes = self.nodes[:i] + [node] + self.nodes[i:]
            new_cl = Cluster(new_nodes)
            if best_cluster is None or new_cl.get_gold_cost(P) < best_cluster.get_gold_cost(P):
                best_cluster = new_cl
        assert best_cluster is not None
        return best_cluster

    def remove_node(self, node: int):
        assert node in self.nodes, "Node to remove not in cluster"
        new_nodes = list(self.nodes)
        new_nodes.remove(node)
        return Cluster(new_nodes)

    def merge(self, P: ExtProblem, other: "Cluster"):
        join = set().union(set(self.nodes), set(other.nodes))
        return Cluster.find_best_cluster(P, join)

    def __str__(self):
        return f"Cluster(nodes={self.nodes})"

    def __repr__(self):
        return self.__str__()

    def __hash__(self) -> int:
        return hash(self._nodes)

def jaccard_index(cl1: Cluster, cl2: Cluster) -> float:
    """Compute the Jaccard index as a measure of similarity between two clusters."""

    set1 = cl1.nodes_set()
    set2 = cl2.nodes_set()
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0.0
    return intersection / union