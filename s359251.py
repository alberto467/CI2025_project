import logging

from Problem import Problem
from src.genetic.algo import genetic_algorithm, trips_expansion
from src.ExtProblem import ExtProblem

logger = logging.getLogger()

debug = True
logger.setLevel(logging.DEBUG if debug else logging.INFO)

logging.getLogger("matplotlib").setLevel(logging.INFO)

logging.debug("Debugging started")

params_slow = {
    "generations": 2500,
    "log_every": 100,
    "population_size": 50,
    # Init params
    "init_size": 20,
    "radial_init_samples": 50,
}

params_medium = {
    "generations": 1750,
    "log_every": 100,
    "population_size": 40,
    # Init params
    "init_size": 8,
    "radial_init_samples": 50,
}


params_fast = {
    "generations": 1000,
    "log_every": 50,
    "population_size": 32,
    # Init params
    "init_size": 4,
    "radial_init_samples": 10,
}

params_faster = {
    "generations": 700,
    "log_every": 50,
    "population_size": 30,
    # Init params
    "init_size": 4,
    "radial_init_samples": 10,
    "p_crossover": 0.2,
}

params_base = {
    "p_crossover": 0.33,
    "p_first": 0.08,
    "p_first_survivors": 0.15,
}


def solution(P: Problem):
    P_ext = ExtProblem(P)

    N = P_ext.graph.number_of_nodes()
    if N <= 50:
        # params = params_slow
        params = {**params_base, **params_fast}
    elif N <= 200:
        params = {**params_base, **params_medium}
    elif N <= 600:
        params = {**params_base, **params_fast}
    else:
        params = {**params_base, **params_faster}

    best, cost, _ = genetic_algorithm(
        P_ext,
        **params,
        merge_init_samples=100 if N <= 100 else 0,
    )

    refined_solution = trips_expansion(P_ext, best[0])
    cost = P_ext.gold_path_cost(refined_solution)

    logger.info(f"Best solution found has cost {cost:.2f}")
    return refined_solution

# P = Problem(num_cities=1000, density=0.8, alpha=0.1, beta=2)
# sol = solution(P)
# print(sol)