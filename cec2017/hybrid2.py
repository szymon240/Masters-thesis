import random

from utility import (
    initializePopulation,
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


def migration(populations, bins):
    individuals = []
    for p in populations:
        individuals.extend(p)

    if not individuals:
        return [[] for _ in range(bins)]

    fits = [ind.fitness for ind in individuals if ind.fitness is not None]
    if not fits:
        return [[] for _ in range(bins)]

    f_min = min(fits)
    f_max = max(fits)

    new_populations = [[] for _ in range(bins)]

    if f_min == f_max:
        new_populations[0] = individuals
        return new_populations

    width = (f_max - f_min) / float(bins)

    for ind in individuals:
        if ind.fitness is None:
            continue
        idx = int((ind.fitness - f_min) / width)
        idx = max(0, min(idx, bins - 1))
        new_populations[idx].append(ind)

    return new_populations


def run_hybrid2(params, evaluations_budget, wrapper_func, D, MIN, MAX):
    """ConvSel with equiwidth migration where parent selection inside each island uses local FUSS."""
    M = params['m_subpopulations']
    R = params['r_interval']
    CROSSOVER = params['crossover_prob']
    MUTATION = params['mutation_prob']
    TOTAL_POP = params['pop_size']
    ALPHA = params['alpha']
    SIZE = max(2, TOTAL_POP // M)

    flat_population = initializePopulation(TOTAL_POP, MIN, MAX, wrapper_func, D)
    evaluations = TOTAL_POP

    valid_start = [ind.fitness for ind in flat_population if ind.fitness is not None]
    if not valid_start:
        raise RuntimeError("Initialization produced no valid fitness values.")

    best_ind = min(
        (ind for ind in flat_population if ind.fitness is not None),
        key=lambda ind: ind.fitness,
    )
    best_global_fitness = best_ind.fitness
    best_global_solution = list(best_ind.genes)

    subpopulations = [[] for _ in range(M)]
    subpopulations[0] = flat_population
    subpopulations = migration(subpopulations, M)

    history = []
    log_interval = max(1, evaluations_budget // 100)
    next_log_threshold = log_interval

    evals_since_migration = 0
    migration_interval = TOTAL_POP * R

    while evaluations >= next_log_threshold:
        history.append((next_log_threshold, best_global_fitness))
        next_log_threshold += log_interval

    while evaluations < evaluations_budget:
        valid_tiers = [i for i in range(M) if subpopulations[i]]
        if not valid_tiers:
            break

        weights = [len(subpopulations[i]) for i in valid_tiers]
        tier_idx = random.choices(valid_tiers, weights=weights, k=1)[0]
        sub = subpopulations[tier_idx]

        p1 = fuss_selection(sub)
        p2 = fuss_selection(sub)

        child = crossover(p1, p2, CROSSOVER)
        child = mutate(child, MUTATION, ALPHA, MIN, MAX)

        wrapper_func(child)
        evaluations += 1
        evals_since_migration += 1

        if child.fitness is not None and child.fitness < best_global_fitness:
            best_global_fitness = child.fitness
            best_global_solution = list(child.genes)

        subpopulations[tier_idx].append(child)
        if len(subpopulations[tier_idx]) > SIZE:
            victim = random.choice(subpopulations[tier_idx])
            subpopulations[tier_idx].remove(victim)

        while evaluations >= next_log_threshold and next_log_threshold <= evaluations_budget:
            history.append((next_log_threshold, best_global_fitness))
            next_log_threshold += log_interval

        if evals_since_migration >= migration_interval:
            evals_since_migration = 0
            subpopulations = migration(subpopulations, M)

    if not history or history[-1][0] < evaluations_budget:
        history.append((evaluations_budget, best_global_fitness))

    return history, best_global_solution
