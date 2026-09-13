D = 10
MIN_VAL = -100.0
MAX_VAL = 100.0
EVALUATIONS_BUDGET = 10000 * D
RUNS = 10

EXPERIMENTAL_PARAMS = {
    "pop_size": [100, 200, 500],
    "tournament_size": [3, 5, 7],
    "crossover_prob": [0.5, 0.75],
    "mutation_prob": [0.25, 0.5],
    "m_subpopulations": [5, 10],
    "r_interval": [10, 25],
    "alpha": [0.2, 0.4],
}

FUNCTIONS_IDS = [i for i in range(1, 31) if i != 2]