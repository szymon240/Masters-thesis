import random
from utility import (
    initializePopulation,
    random_deletion,
    crossover,
    mutate
)

def fuss_selection(population):
    valid_fits = [ind.fitness for ind in population if ind.fitness is not None]

    if not valid_fits:
        return random.choice(population)

    f_min = min(valid_fits)
    f_max = max(valid_fits)

    if f_min == f_max:
        return random.choice(population)

    target_fitness = random.uniform(f_min, f_max)

    best_match = None
    smallest_difference = float('inf')

    for ind in population:
        if ind.fitness is None:
            continue
        difference = abs(ind.fitness - target_fitness)
        if difference < smallest_difference:
            smallest_difference = difference
            best_match = ind

    return best_match if best_match else random.choice(population)

def run_fuss(params, evaluations_budget, wrapper_func, D, MIN, MAX):
    SIZE = params['pop_size']
    CROSSOVER = params['crossover_prob']
    MUTATION = params['mutation_prob']
    ALPHA = params['alpha']

    population = initializePopulation(SIZE, MIN, MAX, wrapper_func, D)
    evaluations = SIZE

    valid_start = [ind.fitness for ind in population if ind.fitness is not None]
    if not valid_start:
        raise RuntimeError("Initialization produced no valid fitness values.")

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
        x1 = fuss_selection(population)
        x2 = fuss_selection(population)

        child = crossover(x1, x2, CROSSOVER)
        child = mutate(child, MUTATION, ALPHA, MIN, MAX)

        wrapper_func(child)
        evaluations += 1

        chosen = random_deletion(population)
        population.remove(chosen)
        population.append(child)

        if child.fitness is not None and child.fitness < best_global_fitness:
            best_global_fitness = child.fitness
            best_global_solution = list(child.genes)

        while evaluations >= next_log_threshold and next_log_threshold <= evaluations_budget:
            history.append((next_log_threshold, best_global_fitness))
            next_log_threshold += log_interval

    if not history or history[-1][0] < evaluations_budget:
        history.append((evaluations_budget, best_global_fitness))

    return history, best_global_solution