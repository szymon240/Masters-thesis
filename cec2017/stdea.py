import random
from utility import (
    initializePopulation,
    tournament_selection,
    crossover,
    mutate
)

def run_stdea(params, evaluations_budget, wrapper_func, D, MIN, MAX):
    """Steady-state EA with tournament selection and random deletion (CEC minimization)."""
    TOURNAMENT = params['tournament_size']
    CROSSOVER = params['crossover_prob']
    MUTATION = params['mutation_prob']
    TOTAL_POP = params['pop_size']
    ALPHA = params['alpha']

    population = initializePopulation(TOTAL_POP, MIN, MAX, wrapper_func, D)
    evaluations = TOTAL_POP

    valid_fits = [ind.fitness for ind in population if ind.fitness is not None]
    if not valid_fits:
        raise RuntimeError("Start population has no valid fitness values.")

    best_ind = min(
        (ind for ind in population if ind.fitness is not None),
        key=lambda ind: ind.fitness,
    )
    best_global_fitness = best_ind.fitness
    best_global_solution = list(best_ind.genes)

    history = []
    log_interval = max(1, evaluations_budget // 100)
    next_log_threshold = log_interval

    while evaluations >= next_log_threshold:
        history.append((next_log_threshold, best_global_fitness))
        next_log_threshold += log_interval

    while evaluations < evaluations_budget:
        actual_tourn = min(TOURNAMENT, len(population))

        p1 = tournament_selection(population, actual_tourn)
        p2 = tournament_selection(population, actual_tourn)

        child = crossover(p1, p2, CROSSOVER)
        child = mutate(child, MUTATION, ALPHA, MIN, MAX)

        wrapper_func(child)
        evaluations += 1

        if child.fitness is not None and child.fitness < best_global_fitness:
            best_global_fitness = child.fitness
            best_global_solution = list(child.genes)

        remove = random.choice(population)
        population.remove(remove)
        population.append(child)

        while evaluations >= next_log_threshold and next_log_threshold <= evaluations_budget:
            history.append((next_log_threshold, best_global_fitness))
            next_log_threshold += log_interval

    if not history or history[-1][0] < evaluations_budget:
        history.append((evaluations_budget, best_global_fitness))

    return history, best_global_solution