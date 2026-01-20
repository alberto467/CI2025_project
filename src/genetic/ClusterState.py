import numpy as np
import timeit
from copy import copy
import logging

from ..ExtProblem import ExtProblem
from ..utils import softmax
from .Cluster import Cluster

logger = logging.getLogger(__name__)


class ClusterState:
    """
    Representation of a valid solution as a set of clusters with gold collection plans:
    - Each cluster contains a subset of nodes (excluding the depot)
    - The union of all clusters covers all nodes (excluding the depot)
    - Each cluster has an associated gold collection plan (order of visiting collection nodes)
    """

    def __init__(self, state: list[Cluster]):
        self.state = state

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

    def cost(self, P: ExtProblem) -> float:
        return sum(cluster.get_gold_cost(P) for cluster in self.state)
    
    def get_centroids(self, P: ExtProblem) -> np.ndarray:
        return np.array([cl.get_centroid(P) for cl in self.state])

    def mutate_random_swap(self, P: ExtProblem):
        if len(self.state) < 2:
            return

        c1, c2 = np.random.choice(range(len(self.state)), size=2, replace=False)

        n1 = np.random.choice(self.state[c1].nodes)
        n2 = np.random.choice(self.state[c2].nodes)

        self.state[c1] = self.state[c1].remove_node(n1).insert_node_at_best_position(P, n2)
        self.state[c2] = self.state[c2].remove_node(n2).insert_node_at_best_position(P, n1)

    def mutate_random_move(self, P: ExtProblem):
        if len(self.state) < 1:
            return

        c1 = np.random.randint(0, len(self.state))
        assert len(self.state[c1].nodes) > 0, "Cannot move from empty cluster"

        n1 = np.random.choice(self.state[c1].nodes)
        self.state[c1] = self.state[c1].remove_node(n1)
        if self.state[c1].is_empty():
            self.state.pop(c1)

        new_cluster = np.random.rand() < 0.1
        if new_cluster:
            self.state.append(Cluster((n1,)))
        else:
            c2 = np.random.randint(0, len(self.state))
            self.state[c2] = self.state[c2].insert_node_at_best_position(P, n1)

    def mutate_merge_clusters(self, P: ExtProblem):
        if len(self.state) < 2:
            return

        # time = timeit.default_timer()
        centroids = self.get_centroids(P)
        # P.times_dict["mutate_merge_clusters-compute_centroids"] = P.times_dict.get("mutate_merge_clusters-compute_centroids", 0.0) + (timeit.default_timer() - time)

        # time = timeit.default_timer()
        i1 = np.random.randint(0, len(self.state))
        dists = np.linalg.norm(centroids - centroids[i1], axis=1)
        dists[i1] = float("inf")
        # probs = softmax(-dists, temp=0.3)
        # i2 = np.random.choice(range(len(self.state)), p=probs)
        i2 = int(np.argmin(dists))
        # P.times_dict["mutate_merge_clusters-find_pair"] = P.times_dict.get("mutate_merge_clusters-find_pair", 0.0) + (timeit.default_timer() - time)

        self.state[i1] = self.state[i1].merge(P, self.state[i2])
        self.state.pop(i2)

    def mutate_random(self, P: ExtProblem):
        r = np.random.rand()

        if r < 0.3:
            # time = timeit.default_timer()
            self.mutate_random_move(P)
            # P.times_dict["mutate_random_move"] = P.times_dict.get("mutate_random_move", 0.0) + (timeit.default_timer() - time)
        elif r < 0.6:
            # time = timeit.default_timer()
            self.mutate_merge_clusters(P)
            # P.times_dict["mutate_merge_clusters"] = P.times_dict.get("mutate_merge_clusters", 0.0) + (timeit.default_timer() - time)
        elif r < 0.8:
            # time = timeit.default_timer()
            self.mutate_split_cluster(P)
            # P.times_dict["mutate_split_cluster"] = P.times_dict.get("mutate_split_cluster", 0.0) + (timeit.default_timer() - time)
        else:
            # time = timeit.default_timer()
            self.mutate_improve_cluster_order(P, samples=1)
            # P.times_dict["mutate_reorder_random_cluster"] = P.times_dict.get("mutate_reorder_random_cluster", 0.0) + (timeit.default_timer() - time)

        # if r < 0.35:
        #     self.mutate_merge_clusters(P)
        # elif r < 0.65:
        #     self.split_cluster(P)
        # elif r < 0.75:
        #     self.mutate_random_swap(P)
        # elif r < 0.85:
        #     self.mutate_random_move(P)
        # else:
        #     self.mutate_reorder_random_cluster(P, samples=10000)

        # self.verify(P)

    def mutate_split_cluster(self, P: ExtProblem, split_tries: int = 40):
        # Randomly select a cluster based on its cost
        candidates = [idx for idx in range(len(self.state)) if len(self.state[idx].nodes) >= 2]
        if not candidates:
            return

        costs = np.array([self.state[idx].get_gold_cost(P) for idx in candidates])
        probs = softmax(costs, temp=0.5)
        cidx = np.random.choice(candidates, p=probs)
        cluster = self.state[cidx]

        # Randomly split the cluster into two
        best_split = None
        best_cost_pair = (float("inf"), float("inf"))
        for _ in range(split_tries):
            split = np.random.randint(0, 2, size=(len(cluster.nodes)))

            cl1_nodes = [n for i, n in enumerate(cluster.nodes) if split[i] == 0]
            cl2_nodes = [n for i, n in enumerate(cluster.nodes) if split[i] == 1]
            if not cl1_nodes or not cl2_nodes:
                continue

            cl1 = Cluster(cl1_nodes)
            cl2 = Cluster(cl2_nodes)
            cost1 = cl1.get_gold_cost(P)
            cost2 = cl2.get_gold_cost(P)

            if cost1 + cost2 < best_cost_pair[0] + best_cost_pair[1]:
                best_split = (cl1, cl2)
                best_cost_pair = (cost1, cost2)

        if best_split is None:
            return

        self.state[cidx] = best_split[0]
        self.state.append(best_split[1])

    def mutate_reorder_all_clusters(self, P: ExtProblem, max_len_perm: int = 5, samples_per_node: int = 30):
        for i, cl in enumerate(self.state):
            if len(cl.nodes) > 1:
                new_cl = Cluster.find_best_cluster(P, cl.nodes_set(), max_len_perm=max_len_perm, samples_per_node=samples_per_node)
                if new_cl.get_gold_cost(P) < cl.get_gold_cost(P):
                    self.state[i] = new_cl

    def mutate_reorder_random_cluster(self, P: ExtProblem, max_len_perm: int = 5, samples_per_node: int = 30):
        cidx = np.random.randint(0, len(self.state))
        if len(self.state[cidx].nodes) > 1:
            new_cl = Cluster.find_best_cluster(P, self.state[cidx].nodes_set(), max_len_perm=max_len_perm, samples_per_node=samples_per_node)
            if new_cl.get_gold_cost(P) < self.state[cidx].get_gold_cost(P):
                self.state[cidx] = new_cl

    def mutate_improve_cluster_order(self, P: ExtProblem, samples: int = 100):
        for i, cl in enumerate(self.state):
            nodes = cl.nodes
            for a, b in np.random.randint(0, len(nodes), size=(samples, 2), dtype=np.int32):
                if a == b:
                    continue
                new_nodes = nodes.copy()
                # try swap
                n = new_nodes.pop(a)
                new_nodes.insert(b, n)
                new_cl = Cluster(new_nodes)
                if new_cl.get_gold_cost(P) < cl.get_gold_cost(P):
                    self.state[i] = new_cl

    def plot(self, P: ExtProblem):
        import matplotlib.pyplot as plt
        colors = plt.cm.get_cmap('tab20', len(self.state))

        for idx, cluster in enumerate(self.state):
            color = colors(idx)
            alpha_color = (color[0], color[1], color[2], 0.3)

            cluster_nodes = cluster.nodes
            cluster_positions = np.array([P.graph.nodes[n]["pos"] for n in cluster_nodes])
            plt.scatter(cluster_positions[:, 0], cluster_positions[:, 1], color=color, label=f'Cluster {idx+1}')

            path = cluster.get_path(P)
            path_positions = np.array([P.graph.nodes[n]["pos"] for n in path])
            plt.plot(path_positions[:, 0], path_positions[:, 1], color=alpha_color)

            centroid = cluster.get_centroid(P)
            plt.scatter(centroid[0], centroid[1], color=color, marker='x', s=50)

        depot_pos = P.graph.nodes[0]["pos"]
        plt.scatter(depot_pos[0], depot_pos[1], color='black', marker='x', s=50, label='Depot')

        plt.title('Cluster State Visualization')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        # plt.legend()

        plt.show()

    def get_gold_path(self, P: ExtProblem) -> list[tuple[int, float]]:
        full_path: list[tuple[int, float]] = [(0, 0.0)]
        for cluster in self.state:
            cluster_path = cluster.get_gold_path(P)
            full_path += cluster_path[1:]
        return full_path

    def __hash__(self) -> int:
        clusters_hashes = sorted(hash(cl) for cl in self.state)
        return hash(tuple(clusters_hashes))
    
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, ClusterState):
            return False
        return hash(self) == hash(value)

    def __copy__(self):
        c = ClusterState(copy(self.state))
        return c

    def __str__(self) -> str:
        return f"ClusterState(num_clusters={len(self.state)}, clusters={self.state})"

    def __repr__(self) -> str:
        return self.__str__()
    

def rand_index(cs1: ClusterState, cs2: ClusterState) -> float:
    """
    Computes the Rand index as a measure of similarity between two ClusterStates.
    """
    N = sum(len(cl.nodes) for cl in cs1.state)
    assert N == sum(len(cl.nodes) for cl in cs2.state), "ClusterStates must cover same number of nodes"

    agree_same = 0
    agree_diff = 0

    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            same_in_cs1 = any(i in cl.nodes and j in cl.nodes for cl in cs1.state)
            same_in_cs2 = any(i in cl.nodes and j in cl.nodes for cl in cs2.state)

            if same_in_cs1 and same_in_cs2:
                agree_same += 1
            elif not same_in_cs1 and not same_in_cs2:
                agree_diff += 1

    total_pairs = N * (N - 1) / 2
    rand_index_value = (agree_same + agree_diff) / total_pairs
    return rand_index_value

def haussdorff_distance(P: ExtProblem, cs1: ClusterState, cs2: ClusterState) -> float:
    """
    Compute the Haussdorff distance as a measure of diversity
    between two ClusterStates (based on the cluster's centroids).
    """

    cs1_centroids = cs1.get_centroids(P)
    cs2_centroids = cs2.get_centroids(P)
    dist_matrix = np.linalg.norm(cs1_centroids[:, np.newaxis, :] - cs2_centroids[np.newaxis, :, :], axis=2)
    min_dists = dist_matrix.min(axis=1) # 0 - sqrt(2)
    # out = np.pow(np.mean((min_dists) ** 2), 1/2)
    # out = out / out.max() * np.max(min_dists)
    # out = 
    # return out
    
    max_dist = np.max(min_dists)
    return max_dist
