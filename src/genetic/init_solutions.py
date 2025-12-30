import numpy as np
from copy import copy
from itertools import permutations
from typing import Literal
import logging

from ..ExtProblem import ExtProblem
from .Cluster import Cluster
from .ClusterState import ClusterState

logger = logging.getLogger(__name__)


def radial_init_solution(P: ExtProblem, num_samples: int = 25) -> ClusterState:
    pos = np.array([ pos for _, pos in P.graph.nodes(data='pos') ])
    pos = pos[1:] - pos[0]
    angles = np.arctan2(pos[:,1], pos[:,0]) + np.pi
    sort_idx = np.argsort(angles)

    # node_angles = []
    # for n in range(1, P.graph.number_of_nodes()):
    #     pos = np.array(P.graph.nodes[n]["pos"])
    #     vector = pos - center
    #     angle = np.arctan2(vector[1], vector[0]) + np.pi
    #     node_angles.append((n, angle))

    # sorted_nodes = [n for n, _ in node_angles]

    def radial_split(cluster_num: int, offset: float) -> ClusterState:
        nonlocal sort_idx

        clusters: list[Cluster] = []
        current_cluster_nodes: list[int] = []
        angle_width = 2 * np.pi / cluster_num

        for i in sort_idx:
            n = i + 1
            angle = angles[i]

            if angle >= angle_width * (len(clusters) + 1 - offset):
                if current_cluster_nodes:
                    cluster = Cluster.find_best_cluster(P, set(current_cluster_nodes))
                    clusters.append(cluster)
                    current_cluster_nodes = []

            current_cluster_nodes.append(n)

        if current_cluster_nodes:
            cluster = Cluster.find_best_cluster(P, set(current_cluster_nodes))
            clusters.append(cluster)

        return ClusterState(clusters)

    # Let's do a binary search for the best number of clusters
    best_state = None
    best_cost = float("inf")
    for candidate_num, offset in zip(np.linspace(1, P.graph.number_of_nodes() - 1, num_samples), np.random.uniform(0, 0.5, num_samples)):
        candidate_num = int(candidate_num)
        state = radial_split(candidate_num, offset)
        cost = state.cost(P)
        if cost < best_cost:
            best_cost = cost
            best_state = state

    assert best_state is not None
    logger.debug(f"Radial solution with {len(best_state.state)} clusters and cost {best_cost}")

    best_state.verify(P)

    return best_state


def baseline_init_solution(P: ExtProblem) -> ClusterState:
    clusters = [Cluster([i]) for i in range(1, P.graph.number_of_nodes())]
    return ClusterState(clusters)


def merge_clusters(P: ExtProblem, cluster_state: ClusterState, method: Literal["perm", "sample"] = "perm", sample_loops = 10, min_clusters = 1):
    """
    From an initial cluster state tries to merge clusters to reduce the overall cost.

    methods: "perm" tries all permutations of cluster pairs to find the best merge at each step.
             "sample" randomly samples cluster pairs to find merges, for a given number of loops.
    """

    clusters = [copy(cl) for cl in cluster_state.state]
    while len(clusters) > min_clusters:
        best_saving = 0
        best_pair: tuple[int, int] | None = None
        best_merged = None

        def try_merge(i: int, j: int):
            nonlocal best_saving, best_pair, best_merged

            cli, clj = clusters[i], clusters[j]

            if clj.is_superpath(P, cli):
                merged = clj.merge(P, cli)
                saved = cli.get_gold_cost(P) + clj.get_gold_cost(P) - merged.get_gold_cost(P)
                if saved > best_saving:
                    best_saving = saved
                    best_pair = (i, j)
                    best_merged = merged

        if method == 'sample':
            pairs = np.arange(len(clusters))
            for _ in range(sample_loops):
                np.random.shuffle(pairs)
                for i in range(0, pairs.shape[0] - (pairs.shape[0] % 2), 2):
                    try_merge(pairs[i], pairs[i+1])

        elif method == 'perm':
            for i, j in permutations(range(len(clusters)), 2):
                if i == j:
                    continue
                try_merge(i, j)

        else:
            raise ValueError(f"Unknown merge method: {method}")

        if best_merged is not None:
            assert best_pair is not None
            i, j = best_pair
            # logging.debug(
            #     f"Merging clusters {clusters[i]} and {clusters[j]} with saving {best_saving}"
            # )
            clusters[i] = best_merged
            clusters.pop(j)
        else:
            # logging.debug(f"No more beneficial merges found, stopping with {len(clusters)} clusters.")
            break

    return ClusterState(clusters)