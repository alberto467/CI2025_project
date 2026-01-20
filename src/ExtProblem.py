import networkx as nx
import numpy as np
import logging

from Problem import Problem

logger = logging.getLogger(__name__)


class ExtProblem(Problem):
    """
    An extended problem class that precomputes and caches additional data structures
    """

    _shortest_paths_mat: list[list[list[int] | None]]

    times_dict: dict[str, float] = dict()
    mutation_monitor: dict[str, dict[str, int | float]] = dict()

    def __init__(self, P: Problem):
        self._graph = P._graph  # .copy()
        self._alpha = P._alpha
        self._beta = P._beta

        logger.debug("Adding adjusted distance attributes...")
        self._add_adj_values()

        logger.debug("Precomputing Floyd-Warshall matrix...")
        self.floyd_mat = nx.floyd_warshall_numpy(self._graph, weight="dist")

        logger.debug("Precomputing adjusted Floyd-Warshall matrix...")
        self.adj_floyd_mat = nx.floyd_warshall_numpy(self._graph, weight="adj_dist")

        logger.debug("Extracting gold amounts...")
        self.gold_amounts = np.array(
            [gold for _, gold in self._graph.nodes(data='gold')],
            dtype=np.float32
        )

        logger.debug("Extracting positions...")
        self.positions = np.array(
            [pos for _, pos in self._graph.nodes(data='pos')],
            dtype=np.float32
        )

        logger.debug("Initializing shortest path cache...")
        N = P.graph.number_of_nodes()
        self._shortest_paths_mat = [[None for _ in range(N)] for _ in range(N)]
        self._angles = None

    def _add_adj_values(self):
        for e in self._graph.edges:
            dist = self._graph.edges[e]["dist"]
            self._graph.edges[e]["adj_dist"] = np.pow(dist, self._beta)


    def cached_shortest_path(self, u: int, v: int) -> list[int]:
        assert u != v, "Shortest path requested between the same node"
        flipped = u > v
        if flipped:
            u, v = v, u

        path = self._shortest_paths_mat[u][v]
        if path is not None:
            if flipped:
                path = list(reversed(path))
            return path

        path = nx.shortest_path(self.graph, u, v, weight="dist")
        self._shortest_paths_mat[u][v] = path
        if flipped:
            path = list(reversed(path))
        return path

    def angles(self) -> np.ndarray:
        if self._angles is not None:
            return self._angles

        pos = self.positions[1:] - self.positions[0]
        angles = np.arctan2(pos[:,1], pos[:,0]) + np.pi

        self._angles = angles
        return angles

    def gold_path_cost(self, path: list[tuple[int, float]]) -> float:
        total = 0
        gold_carried = 0.0
        for (u, gold_prev), (v, _) in zip(path[:-1], path[1:]):
            if u == 0:
                gold_carried = 0.0
            else:
                gold_carried += gold_prev

            dist = self.floyd_mat[u, v]
            total += dist + (dist * self._alpha * gold_carried) ** self._beta

        return total

    def verify_gold_path(self, path: list[tuple[int, float]]) -> bool:
        tol = 1e-3

        gold_left = self.gold_amounts.copy()
        for node, gold_taken in path:
            if gold_taken < 0:
                logger.error(f"Negative gold taken at node {node}: {gold_taken}")
                return False
            if 0 < gold_taken < 1:
                logger.error(f"Fractional gold taken at node {node}: {gold_taken}")
                return False
            if gold_taken > gold_left[node] + tol:
                logger.error(
                    f"Too much gold taken at node {node}: taken {gold_taken}, available {gold_left[node]}"
                )
                return False
            gold_left[node] -= gold_taken

        assert np.all(np.abs(gold_left) < tol), "Not all gold has been collected!"

    def lower_bound(self):
        """
        Very basic lower bound, only considering the cost relative to transporting gold
        (not very important for low alpha/beta values)
        """

        gold = self.gold_amounts[1:]
        adj_dists = self.adj_floyd_mat[1:, 0]

        return np.sum(adj_dists * np.power(self._alpha * gold, self._beta))