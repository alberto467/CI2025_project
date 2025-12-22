import networkx as nx
import numpy as np
import logging
from itertools import combinations
import matplotlib.pyplot as plt


class Problem:
    _graph: nx.Graph
    _alpha: float
    _beta: float

    def __init__(
        self,
        num_cities: int,
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
        density: float = 0.5,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        self._alpha = alpha
        self._beta = beta
        # Generate the coords of the cities
        cities = rng.random(size=(num_cities, 2))
        # First city is at (0.5, 0.5)
        cities[0, 0] = cities[0, 1] = 0.5

        self._graph = nx.Graph()
        # First city has no gold
        self._graph.add_node(0, pos=(cities[0, 0], cities[0, 1]), gold=0)
        for c in range(1, num_cities):
            # Gold between 1 and 1000, uniformly distributed
            self._graph.add_node(
                c, pos=(cities[c, 0], cities[c, 1]), gold=(1 + 999 * rng.random())
            )

        tmp = cities[:, np.newaxis, :] - cities[np.newaxis, :, :]
        d = np.sqrt(np.sum(np.square(tmp), axis=-1))
        for c1, c2 in combinations(range(num_cities), 2):
            # Possible exploit on lower densities, there is always a path between any two successive cities
            if rng.random() < density or c2 == c1 + 1:
                self._graph.add_edge(c1, c2, dist=d[c1, c2])

        assert nx.is_connected(self._graph)

    @property
    def graph(self) -> nx.Graph:
        return nx.Graph(self._graph)

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    def cost(self, path, weight):
        total = 0
        for u, v in zip(path[:-1], path[1:]):
            dist = nx.shortest_path_length(self._graph, u, v, weight="dist")
            total += dist + (self._alpha * dist * weight) ** self._beta
        return total

    def baseline(self):
        cost = 0
        full_path = [(0, 0)]
        for dest, path in nx.single_source_dijkstra_path(
            self._graph, source=0, weight="weight"
        ).items():
            if dest == 0:
                continue
            logging.debug(
                f"dummy_solution: go to {dest} ({' > '.join(str(n) for n in path)}) -- cost: {self.cost(path, 0):.2f}"
            )
            for n in path[1:-1]:
                full_path.append((n, 0))
            logging.debug(
                f"dummy_solution: grab {self._graph.nodes[dest]['gold']:.2f}kg of gold"
            )
            full_path.append((dest, self._graph.nodes[dest]["gold"]))
            logging.debug(
                f"dummy_solution: return to 0 ({' > '.join(str(n) for n in reversed(path))}) -- cost: {self.cost(path, self._graph.nodes[dest]['gold']):.2f}"
            )
            for n in reversed(path[:-1]):
                full_path.append((n, 0))
            cost += self.cost(path, 0) + self.cost(
                path, self._graph.nodes[dest]["gold"]
            )
        logging.info(f"dummy_solution: total cost: {cost:.2f}")
        assert calc_solution_cost(full_path, self) - cost < 1e-6, (
            "Baseline cost mismatch!"
        )
        return cost, full_path

    def lower_bound(self):
        lb = 0
        for n in self._graph.nodes:
            if n == 0:
                continue
            gold = self._graph.nodes[n]["gold"]
            dist = nx.shortest_path_length(self._graph, n, 0, weight="dist")
            lb += (self._alpha * dist * gold) ** self._beta
        return lb

    def plot(self):
        plt.figure(figsize=(10, 10))
        pos = nx.get_node_attributes(self._graph, "pos")
        size = [100] + [
            self._graph.nodes[n]["gold"] for n in range(1, len(self._graph))
        ]
        color = ["red"] + ["lightblue"] * (len(self._graph) - 1)
        return nx.draw(
            self._graph,
            pos,
            with_labels=True,
            node_color=color,
            node_size=size,
            arrows=True,
            arrowsize=300,
        )


def calc_solution_cost(path: list[tuple[int, float]], P: Problem) -> float:
    """
    Calculate the total cost of a solution, and verify that
    the solution is valid: all gold has been collected.
    """
    assert path[0][0] == 0 and path[-1][0] == 0, "Path must start and end at the depot"

    remaining_gold = list(nx.get_node_attributes(P.graph, "gold").values())

    total_cost = 0.0
    current_gold = 0.0
    for i in range(len(path) - 1):
        u = path[i][0]
        if u == 0:
            assert path[i][1] == 0.0, "No gold can be collected at the depot"
            current_gold = 0.0
        v = path[i + 1][0]
        gold_taken = path[i][1]
        assert gold_taken <= remaining_gold[u], (
            f"Trying to take more gold than available at node {u}"
        )
        current_gold += gold_taken
        remaining_gold[u] -= gold_taken
        total_cost += P.cost([u, v], current_gold)

    assert all(g == 0.0 for g in remaining_gold), "Not all gold has been collected!"

    return total_cost


def print_path_steps(path: list[tuple[int, float]], P: Problem):
    remaining_gold = list(nx.get_node_attributes(P.graph, "gold").values())
    total_cost = 0.0
    current_gold = 0.0
    for i in range(len(path) - 1):
        u = path[i][0]
        if u == 0:
            print(
                f"[0] At depot, all gold ({current_gold:.2f}kg) unloaded.", current_gold
            )
            current_gold = 0.0
        v = path[i + 1][0]
        gold_taken = path[i][1]
        current_gold += gold_taken
        print(
            f"[{u}] Taking {gold_taken:.2f}kg of gold - leaving {remaining_gold[u] - gold_taken:.2f}kg out of previous {remaining_gold[u]:.2f}kg"
        )
        remaining_gold[u] -= gold_taken
        step_cost = P.cost([u, v], current_gold)
        print(
            f"[{u}->{v}] Moving from {u} to {v} with {current_gold:.2f}kg of gold - step cost: {step_cost:.2f}"
        )
        total_cost += step_cost
    print(f"Total cost of the path: {total_cost:.2f}")


# example_problem = Problem(num_cities=100, density=0.8, alpha=1.0, beta=1.0)
