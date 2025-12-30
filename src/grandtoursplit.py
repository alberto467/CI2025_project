import numpy as np
import logging

logger = logging.getLogger(__name__)


# def find_grand_tour(P: Problem, trials: int = 10):
#     shortest_dists = nx.floyd_warshall_numpy(P.graph, weight="dist")

#     def generate_greedy_tour(nodes: set[int]):
#         unvisited = nodes.copy()
#         unvisited.discard(0)  # Start from the depot
#         tour = [0]
#         total_dist = 0.0

#         while unvisited:
#             unvisited_list = list(unvisited)
#             dists = shortest_dists[tour[-1], unvisited_list]
#             probs = softmax(-dists, temp=0.7)

#             next_city_idx = int(
#                 np.random.choice(np.arange(len(unvisited_list)), size=1, p=probs)[0]
#             )
#             next_city = unvisited_list[next_city_idx]
#             total_dist += dists[next_city_idx]

#             tour += nx.shortest_path(P.graph, tour[-1], next_city, weight="dist")[1:]
#             unvisited.remove(next_city)

#         # Return to the depot
#         back_home = nx.shortest_path(P.graph, tour[-1], 0, weight="dist")
#         total_dist += shortest_dists[tour[-1], 0]
#         tour += back_home[1:]

#         return tour, total_dist

#     def catch_gold(tour):
#         assert tour[0] == 0 and tour[-1] == 0, "Tour must start and end at the depot"

#         nodes_left = set(P.graph.nodes)
#         nodes_left.remove(0)

#         total_cost = 0.0
#         current_gold = 0.0
#         out = [(0, 0)]  # (node, gold collected at this node)
#         remaining_gold = np.array([P.graph.nodes[n]["gold"] for n in P.graph.nodes])

#         i = -1
#         while i < len(tour) - 2:
#             i += 1
#             u, v = tour[i], tour[i + 1]
#             # Handle the depot
#             if v == 0:
#                 total_cost += P.cost([u, v], current_gold)
#                 current_gold = 0.0
#                 out.append((v, 0))
#                 if not nodes_left:
#                     break
#                 continue

#             trip_to_home = nx.shortest_path(P.graph, u, 0)
#             trip_from_home = nx.shortest_path(P.graph, 0, v)
#             trip_back_cost = P.cost(trip_to_home, current_gold) + P.cost(
#                 trip_from_home, 0.0
#             )

#             LOOK_AHEAD = 3
#             next_current_gold_cost = P.cost(tour[i : i + LOOK_AHEAD], current_gold)
#             next_empty_gold_cost = P.cost(tour[i : i + LOOK_AHEAD], 0)
#             tripless_lookahead_diff = next_current_gold_cost - next_empty_gold_cost

#             if (
#                 tripless_lookahead_diff > 0
#                 and trip_back_cost / tripless_lookahead_diff < 0.95
#             ):
#                 # Better to go back to depot now

#                 for j in range(len(trip_to_home) - 1):
#                     u2, v2 = trip_to_home[j], trip_to_home[j + 1]
#                     total_cost += P.cost([u2, v2], current_gold)
#                     current_gold += remaining_gold[v2]
#                     out.append((v2, remaining_gold[v2]))
#                     remaining_gold[v2] = 0.0
#                     nodes_left.discard(v2)

#                 current_gold = 0.0

#                 # We are now at the depot, let's randomly recalculate the next tour
#                 if not nodes_left:
#                     break

#                 if random.random() < 0:
#                     next_tours = [
#                         generate_greedy_tour(nodes_left) for _ in range(trials)
#                     ]
#                     best_tour = min(next_tours, key=lambda x: x[1])[0]
#                     i = -1
#                     tour = best_tour
#                     continue

#                 for j in range(len(trip_from_home) - 2):
#                     u2, v2 = trip_from_home[j], trip_from_home[j + 1]
#                     total_cost += P.cost([u2, v2], current_gold)
#                     out.append((v2, 0))

#             else:
#                 total_cost += P.cost([u, v], current_gold)

#             gold = remaining_gold[v]
#             out.append((v, gold))
#             current_gold += gold
#             remaining_gold[v] = 0.0
#             nodes_left.discard(v)

#         assert remaining_gold.sum() == 0.0, "Not all gold collected!"

#         return out, total_cost

#     tours = [generate_greedy_tour(set(P.graph.nodes)) for _ in range(trials)]

#     gold_tours = [catch_gold(tour) for tour, _ in tours]
#     return tours, gold_tours


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


# def cover_cluster(P: ExtProblem, cluster: set[int]):
#     """
#     Finds the shortest path that covers all cities in the cluster and returns to the depot.
#     """

#     if len(cluster) == 1:
#         city = list(cluster)[0]
#         path_to = nx.shortest_path(P.graph, 0, city, weight="dist")
#         path_from = nx.shortest_path(P.graph, city, 0, weight="dist")
#         return path_to + path_from[1:]

#     if len(cluster) <= 3:
#         # Let's find the best permutation
#         best_permutation = None
#         best_cost = float("inf")
#         for perm in permutations(cluster, len(cluster)):
#             perm = list(perm)
#             total_cost = 0.0
#             for u, v in zip([0] + perm, perm + [0]):
#                 total_cost += P.floyd_mat[u, v]
#             if total_cost < best_cost:
#                 best_cost = total_cost
#                 best_permutation = perm

#         assert best_permutation is not None
#         path = [0]
#         for u, v in zip([0] + best_permutation, best_permutation + [0]):
#             sub_path = nx.shortest_path(P.graph, u, v, weight="dist")
#             path += sub_path[1:]
#         return path

#     to_visit = set(cluster)
#     path = [0]

#     while to_visit:
#         current = path[0]
#         nearest_city = min(
#             to_visit,
#             key=lambda city: P.floyd_mat[city, current],
#         )
#         sub_path = nx.shortest_path(P.graph, nearest_city, current, weight="dist")
#         path = sub_path[:-1] + path  # Avoid duplicating the current city
#         for n in sub_path[:-1]:
#             to_visit.discard(n)

#     from_home = nx.shortest_path(P.graph, 0, path[0], weight="dist")
#     path = from_home[:-1] + path
#     return path


# def cover_all_clusters(P: ExtProblem, floyd_mat: np.ndarray, clusters: list[list[int]]):
#     full_path = [0]
#     for cluster in clusters:
#         cluster_path = cover_cluster(P, floyd_mat, cluster)
#         full_path += cluster_path[1:]
#     return full_path


# def catch_gold(P: ExtProblem, tour: list[int], cluster: set[int]):
#     """
#     Given a tour and a cluster of nodes, returns the gold collection plan and total cost.
#     """
#     assert tour[0] == 0 and tour[-1] == 0, "Tour must start and end at the depot"
#     assert set(tour).issuperset(cluster), "Tour must cover all nodes in the cluster"

#     nodes_left = cluster.copy()

#     total_cost = 0.0
#     current_gold = 0.0
#     out = [(0, 0.0)]  # (node, gold collected at this node)
#     remaining_gold = np.array(
#         [P.graph.nodes[n]["gold"] if n in cluster else 0.0 for n in P.graph.nodes]
#     )
#     logging.debug(
#         f"Catching gold in cluster {cluster} - total gold: {remaining_gold.sum()}"
#     )

#     i = -1
#     while i < len(tour) - 2:
#         i += 1
#         u, v = tour[i], tour[i + 1]
#         # Handle the depot
#         if v == 0:
#             total_cost += P.cost([u, v], current_gold)
#             current_gold = 0.0
#             out.append((v, 0))
#             if not nodes_left:
#                 break
#             continue

#         # trip_to_home = nx.shortest_path(P.graph, u, 0)
#         # trip_from_home = nx.shortest_path(P.graph, 0, v)
#         # trip_back_cost = P.cost(trip_to_home, current_gold) + P.cost(
#         #     trip_from_home, 0.0
#         # )

#         # LOOK_AHEAD = 3
#         # next_current_gold_cost = P.cost(tour[i : i + LOOK_AHEAD], current_gold)
#         # next_empty_gold_cost = P.cost(tour[i : i + LOOK_AHEAD], 0)
#         # tripless_lookahead_diff = next_current_gold_cost - next_empty_gold_cost

#         # if (
#         #     False
#         #     and tripless_lookahead_diff > 0
#         #     and trip_back_cost / tripless_lookahead_diff < 0.95
#         # ):
#         #     # Better to go back to depot now

#         #     for j in range(len(trip_to_home) - 1):
#         #         u2, v2 = trip_to_home[j], trip_to_home[j + 1]
#         #         total_cost += P.cost([u2, v2], current_gold)
#         #         current_gold += remaining_gold[v2]
#         #         out.append((v2, remaining_gold[v2]))
#         #         remaining_gold[v2] = 0.0
#         #         nodes_left.discard(v2)

#         #     current_gold = 0.0

#         #     # We are now at the depot, let's randomly recalculate the next tour
#         #     if not nodes_left:
#         #         break

#         #     for j in range(len(trip_from_home) - 2):
#         #         u2, v2 = trip_from_home[j], trip_from_home[j + 1]
#         #         total_cost += P.cost([u2, v2], current_gold)
#         #         out.append((v2, 0))

#         # else:
#         total_cost += P.cost([u, v], current_gold)

#         take_gold = v in cluster and v not in set(tour[i + 3 :])

#         gold_taken = float(remaining_gold[v]) if take_gold else 0.0
#         # logging.debug(f"At node {v}, take_gold={take_gold}, gold_taken={gold_taken}")

#         out.append((v, gold_taken))
#         current_gold += gold_taken
#         remaining_gold[v] -= gold_taken
#         nodes_left.discard(v)

#     assert remaining_gold.sum() == 0.0, "Not all gold collected!"

#     return out, total_cost

# def cluster_solve(P: ExtProblem, merge_clusters_method: Literal["perm", "sample", None] = "perm", sample_loops = 10, min_clusters = 1):
#     baseline_sol = baseline_init_solution(P)
#     if merge_clusters_method is None:
#         return baseline_sol
#     else:
#         return merge_clusters(P, baseline_sol, method=merge_clusters_method, sample_loops=sample_loops, min_clusters=min_clusters)
