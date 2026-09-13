import random
from utility_frams import (
    initialize_population_frams,
    random_deletion,
    crossover_frams,
    mutate_frams
)

def clone_individual(individual, framsLib):
    cloned = crossover_frams(individual, individual, 0.0, framsLib)
    if cloned is None:
        raise RuntimeError("Unable to clone individual.")
    return cloned

def mutate_offspring(parent, mutation_prob, framsLib):
    child = clone_individual(parent, framsLib)
    result = mutate_frams(child, mutation_prob, framsLib)
    if result is not None:
        child = result
    child.fitness = None
    return child

def crossover_offspring(p1, p2, crossover_prob, framsLib):
    child = crossover_frams(p1, p2, crossover_prob, framsLib)
    if child is None:
        raise RuntimeError("crossover_frams returned None.")
    child.fitness = None
    return child

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

    for individual in population:
        if individual.fitness is None:
            continue
        difference = abs(individual.fitness - target_fitness)
        if difference < smallest_difference:
            smallest_difference = difference
            best_match = individual

    return best_match if best_match else random.choice(population)


def run_fuss(params, evaluations_budget, wrapper_func, framsLib):
    SIZE = params['pop_size']
    CROSSOVER = params['crossover_prob']
    MUTATION = params['mutation_prob']

    population = initialize_population_frams(SIZE, framsLib)
    evaluations = 0

    for ind in population:
        if ind.fitness is None:
            wrapper_func(ind)
            evaluations += 1

    valid_start = [ind.fitness for ind in population if ind.fitness is not None]
    if not valid_start:
        raise RuntimeError("FUSS initialization produced no valid fitness values.")

    best_global_fitness = max(valid_start)
    best_global_genotype = max(
        (ind for ind in population if ind.fitness is not None),
        key=lambda ind: ind.fitness,
    ).genotype
    history = []

    log_interval = 1000
    next_log_threshold = log_interval

    while evaluations >= next_log_threshold:
        history.append((next_log_threshold, best_global_fitness))
        next_log_threshold += log_interval

    while evaluations < evaluations_budget:
        x1 = fuss_selection(population)
        x2 = fuss_selection(population)

        child = crossover_offspring(x1, x2, CROSSOVER, framsLib)
        child = mutate_offspring(child, MUTATION, framsLib)

        if child.fitness is None:
            wrapper_func(child)
            evaluations += 1

        # --- KLASYCZNE USUWANIE FUSS ---
        chosen = random_deletion(population)
        population.remove(chosen)
        population.append(child)
        # -------------------------------

        if child.fitness is not None and child.fitness > best_global_fitness:
            best_global_fitness = child.fitness
            best_global_genotype = child.genotype

        while evaluations >= next_log_threshold and next_log_threshold <= evaluations_budget:
            history.append((next_log_threshold, best_global_fitness))
            print(f"Evals: {next_log_threshold}, MaxFit: {best_global_fitness:.4f}")
            next_log_threshold += log_interval

    if not history or history[-1][0] < evaluations_budget:
        history.append((evaluations_budget, best_global_fitness))

    return history, best_global_genotype