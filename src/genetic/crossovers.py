import numpy as np
from itertools import product
from copy import copy
from warnings import deprecated
import logging

from ..ExtProblem import ExtProblem
from .Cluster import Cluster
from .ClusterState import ClusterState

logger = logging.getLogger(__name__)


def radial_crossover(
    P: ExtProblem,
    p1: ClusterState,
    p2: ClusterState,
    outside_samples: int = 20,
    reorder_samples_per_node: int = 30,
) -> ClusterState:
    # Split nodes into two groups based on their angle, using a random split angle
    angles = P.angles()
    split_angle = np.random.uniform(0, 2 * np.pi)
    # Get node groups masks
    is_p1 = ((angles + split_angle) % (2 * np.pi)) < np.pi
    is_p2 = ~is_p1

    def process_cluster(cl: Cluster, nodes_mask: np.ndarray):
        """
        Process a cluster to determine its inside and outside nodes based on the provided mask.
        Returns a tuple of (cluster, inside_nodes, outside_nodes) or None if no inside nodes.
        """
        cl_nodes = np.array(cl.nodes)
        cl_nodes_mask = nodes_mask[cl_nodes - 1]

        if not np.any(cl_nodes_mask):
            # There are no inside nodes
            return None

        # Save the cluster with its inside and outside nodes
        inside_nodes: set[int] = set(cl_nodes[cl_nodes_mask])
        outside_nodes: set[int] = set(cl_nodes[~cl_nodes_mask])

        return cl, inside_nodes, outside_nodes

    matches1 = [cl for cl in (process_cluster(cl, is_p1) for cl in p1.state) if cl is not None]
    matches2 = [cl for cl in (process_cluster(cl, is_p2) for cl in p2.state) if cl is not None]

    outside1_set = set().union(*(outside for _, _, outside in matches1))
    outside2_set = set().union(*(outside for _, _, outside in matches2))

    contested_num = len(outside1_set) + len(outside2_set)
    # logging.debug(f"Radial crossover contested nodes length: {contested_num}")

    if contested_num == 0:
        # The clusters selected from both parents do not overlap, just add them together
        return ClusterState([cl for cl, _, _, in matches1] +
                            [cl for cl, _, _, in matches2])

    outside1 = np.array(list(outside1_set))
    outside2 = np.array(list(outside2_set))

    def exchange_group_outside_nodes(matches: list[tuple[Cluster, set[int], set[int]]], allow: set[int], discard: set[int]):
        clusters: list[Cluster] = []
        cost_diff = 0.0

        for cl, inside, outside in matches:
            # Remove inside nodes to discard
            new_nodes = inside.difference(discard)
            # Check if cluster was modified
            modified = len(new_nodes) != len(inside)
            # Add allowed outside nodes
            if outside:
                to_add = outside.intersection(allow)
                if to_add:
                    modified = True
                    new_nodes = new_nodes.union(to_add)

            if not modified:
                clusters.append(cl)

            elif new_nodes:
                new_cl = Cluster.find_best_cluster(P, new_nodes, samples_per_node=reorder_samples_per_node)
                clusters.append(new_cl)
                cost_diff += new_cl.get_gold_cost(P) - cl.get_gold_cost(P)

        return clusters, cost_diff

    def exchange_outside_nodes(allow_mask: list[bool]):
        if not any(allow_mask):
            # No changes to be made
            return [cl for cl, _, _ in matches1] + [cl for cl, _, _ in matches2], 0.0

        allow1 = set(outside1[allow_mask[:outside1.shape[0]]])
        allow2 = set(outside2[allow_mask[outside1.shape[0]:]])

        from1, cost_diff1 = exchange_group_outside_nodes(matches1, allow1, allow2)
        from2, cost_diff2 = exchange_group_outside_nodes(matches2, allow2, allow1)

        return from1 + from2, cost_diff1 + cost_diff2

    best_state = None
    best_cost = float("inf")

    if contested_num < np.log2(outside_samples * 1.5):
        # logger.debug(f"Trying all combinations for radial crossover with contested nodes: {contested_num}")
        # Try all combinations
        for select_mask in product([False, True], repeat=contested_num):
            clusters, cost_diff = exchange_outside_nodes(list(select_mask))
            # print(cost_diff)
            # cost = state.cost(P)
            if cost_diff < best_cost:
                best_cost = cost_diff
                best_state = clusters

    else:
        # logger.debug(f"Sampling combinations for radial crossover, contested nodes: {contested_num}")
        # Sample combinations
        # best_state = None
        # best_cost = float("inf")
        # for select_mask in np.random.randint(0, 2, size=(outside_samples, contested_num), dtype=bool):
        #     clusters, cost_diff = try_outside_allow(select_mask)
        #     # cost = state.cost(P)
        #     if cost_diff < best_cost:
        #         best_cost = cost_diff
        #         best_state = clusters
        # best_state = try_outside_allow([False] * contested_num)[0]

        # for select_mask in [
        #     [False]*contested_num,
        #     [True]*contested_num,
        #     [False]*outside1.shape[0] + [True]*outside2.shape[0],
        #     [True]*outside1.shape[0] + [False]*outside2.shape[0]
        # ]:
        #     clusters, cost_diff = exchange_outside_nodes(select_mask)
        #     # cost = state.cost(P)
        #     if cost_diff < best_cost:
        #         best_cost = cost_diff
        #         best_state = clusters

        # Better to start from no exchanges
        best_mask = [False] * contested_num
        # best_mask = list(np.random.randint(0, 2, size=(contested_num,), dtype=bool))

        best_state, best_cost = exchange_outside_nodes(best_mask)
        # Simulated annealing style optimization
        for flip_idx in np.random.randint(0, contested_num, size=(outside_samples,)):
            # Flip the bit
            best_mask[flip_idx] = not best_mask[flip_idx]

            # Try for improvements
            clusters, cost_diff = exchange_outside_nodes(best_mask)
            if cost_diff < best_cost:
                # logger.debug(f"Radial crossover improved cost diff {cost_diff:.3f}")
                best_cost = cost_diff
                best_state = clusters
            else:
                # Revert change
                best_mask[flip_idx] = not best_mask[flip_idx]


    assert best_state is not None

    # print(best_state)
    # best_state.plot(P)
    # best_state.verify(P)
    return ClusterState(best_state)


@deprecated("Use radial_crossover instead")
def crossover(P: ExtProblem, p1: ClusterState, p2: ClusterState) -> ClusterState:
    set1 = set([cl for cl in p1.state])
    set2 = set([cl for cl in p2.state])

    common = set1.intersection(set2)

    diff1 = list(set1.difference(common))
    diff2 = list(set2.difference(common))
    if not diff1 or not diff2:
        return copy(p1)

    np.random.shuffle(diff1) # type: ignore
    np.random.shuffle(diff2) # type: ignore

    diff_nodes: set[int] = set()
    new_diff: list[list[int]] = []

    current_pi = np.random.randint(0, 2)
    while diff1 or diff2:
        current_pi = 1 - current_pi
        current_diff = diff1 if current_pi == 0 else diff2
        if not current_diff:
            continue
        min_overlap = None
        best_idx = None
        for idx, cl in enumerate(current_diff):
            overlap = diff_nodes.intersection(cl.nodes_set())
            if min_overlap is None or len(overlap) < len(min_overlap):
                min_overlap = overlap
                best_idx = idx
            if len(min_overlap) == 0:
                break

        assert best_idx is not None and min_overlap is not None, "Crossover failed to find the best cluster"
        best_cl = current_diff.pop(best_idx)
        new_cl = best_cl.nodes_set().difference(min_overlap)
        if new_cl:
            new_diff.append([n for n in best_cl.nodes if n not in min_overlap])
            diff_nodes.update(new_cl)


    out = ClusterState(list(common) +
                       [ Cluster(nodes) for nodes in new_diff ])

    # out.verify(P)
    return out


@deprecated("Use radial_crossover instead")
def crossover_swap_near_centroids(P: ExtProblem, p1: ClusterState, p2: ClusterState) -> ClusterState:
    # Check for same clusters
    set1 = set([cl for cl in p1.state])
    set2 = set([cl for cl in p2.state])
    common = set1.intersection(set2)
    diff1 = list(set1.difference(common))
    diff2 = list(set2.difference(common))

    if not diff1 or not diff2:
        # We have exact copies
        return copy(p1)

    if len(diff1) < len(diff2):
        diff1, diff2 = diff2, diff1  # Ensure diff1 is larger

    centroids1 = np.array([cl.get_centroid(P) for cl in diff1])
    centroids2 = np.array([cl.get_centroid(P) for cl in diff2])

    for i1, (cl1, cent1) in enumerate(zip(diff1, centroids1)):
        dists = np.linalg.norm(centroids2 - cent1, axis=1)
        closest_idx = np.argmin(dists)
        cl2 = diff2[closest_idx]

        # Swap two random nodes between cl1 and cl2
        swap_i1 = np.random.randint(0, len(cl1.nodes))
        swap_i2 = np.random.randint(0, len(cl2.nodes))

        node1 = cl1.nodes[swap_i1]
        node2 = cl2.nodes[swap_i2]
        new_cl1 = cl1.remove_node(node1).insert_node_at_best_position(P, node2)
        new_cl2 = cl2.remove_node(node2).insert_node_at_best_position(P, node1)

        diff1[i1] = new_cl1
        diff2[closest_idx] = new_cl2

    out1 = ClusterState(list(common) + diff1)
    out2 = ClusterState(list(common) + diff2)

    out1.verify(P)
    out2.verify(P)

    return out1 if out1.cost(P) < out2.cost(P) else out2