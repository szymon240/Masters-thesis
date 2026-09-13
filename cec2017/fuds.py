import random
from utility import (
    initializePopulation,
    tournament_selection,
    crossover,
    mutate
)

def fuds_deletion(population, bins=20):
    valid_fits = [ind.fitness for ind in population if ind.fitness is not None]

    if not valid_fits:
        return random.choice(population)

    f_min = min(valid_fits)
    f_max = max(valid_fits)

    if f_min == f_max:
        return random.choice(population)

    capacity = (f_max - f_min) / bins
    if capacity == 0:
        return random.choice(population)

    bins_list = [[] for _ in range(bins)]

    for i in population:
        if i.fitness is None: continue
        index = int((i.fitness - f_min) / capacity)
        index = max(0, min(index, bins - 1))
        bins_list[index].append(i)

    most_crowded = -1
    size = -1
    for i in range(bins):
        if len(bins_list[i]) > size:
            size = len(bins_list[i])
            most_crowded = i

    if not bins_list[most_crowded]:
        return random.choice(population)

    chosen = random.choice(bins_list[most_crowded])
    return chosen

def run_fuds(params, evaluations_budget, wrapper_func, D, MIN, MAX):
    SIZE = params['pop_size']
    TOURNAMENT = params['tournament_size']
    CROSSOVER = params['crossover_prob']
    MUTATION = params['mutation_prob']
    ALPHA = params['alpha']
    BINS = 20

    population = initializePopulation(SIZE, MIN, MAX, wrapper_func, D)
    evaluations = SIZE

    valid_start = [ind.fitness for ind in population if ind.fitness is not None]
    if not valid_start:
        raise RuntimeError("FUDS initialization produced no valid fitness values.")

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
        x1 = tournament_selection(population, TOURNAMENT)
        x2 = tournament_selection(population, TOURNAMENT)

        child = crossover(x1, x2, CROSSOVER)
        child = mutate(child, MUTATION, ALPHA, MIN, MAX)

        wrapper_func(child)
        evaluations += 1

        chosen = fuds_deletion(population, BINS)

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