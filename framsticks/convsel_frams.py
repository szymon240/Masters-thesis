import random
from utility_frams import (
    initialize_population_frams,
    tournament_selection,
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

def run_convsel(params, evaluations_budget, wrapper_func, framsLib):
    M = params['m_subpopulations']
    R = params['r_interval']
    TOURNAMENT = params['tournament_size']
    CROSSOVER = params['crossover_prob']
    MUTATION = params['mutation_prob']
    TOTAL_POP = params['pop_size']
    SIZE = max(2, TOTAL_POP // M)

    flat_population = initialize_population_frams(TOTAL_POP, framsLib)
    evaluations = 0

    for ind in flat_population:
        if ind.fitness is None:
            wrapper_func(ind)
            evaluations += 1

    best_global_fitness = max(
        ind.fitness for ind in flat_population if ind.fitness is not None
    )
    best_global_genotype = max(
        (ind for ind in flat_population if ind.fitness is not None),
        key=lambda ind: ind.fitness,
    ).genotype

    subpopulations = [[] for _ in range(M)]
    subpopulations[0] = flat_population
    subpopulations = migration(subpopulations, M)

    history = []
    log_interval = 1000
    next_log_threshold = log_interval

    print(f"Start ConvSel steady-state (M={M}, R={R}, pop={TOTAL_POP})")

    evals_since_migration = 0
    migration_interval = TOTAL_POP * R

    while evaluations < evaluations_budget:
        valid_tiers = [i for i in range(M) if subpopulations[i]]
        if not valid_tiers:
            break

        weights = [len(subpopulations[i]) for i in valid_tiers]
        tier_idx = random.choices(valid_tiers, weights=weights, k=1)[0]
        sub = subpopulations[tier_idx]
        actual_tourn = min(TOURNAMENT, len(sub))

        p1 = tournament_selection(sub, actual_tourn)
        p2 = tournament_selection(sub, actual_tourn)

        child = crossover_offspring(p1, p2, CROSSOVER, framsLib)
        child = mutate_offspring(child, MUTATION, framsLib)

        if child.fitness is None:
            wrapper_func(child)
            evaluations += 1
            evals_since_migration += 1

        if child.fitness is not None and child.fitness > best_global_fitness:
            best_global_fitness = child.fitness
            best_global_genotype = child.genotype

        subpopulations[tier_idx].append(child)
        if len(subpopulations[tier_idx]) > SIZE:
            victim = random.choice(subpopulations[tier_idx])
            subpopulations[tier_idx].remove(victim)

        while evaluations >= next_log_threshold and next_log_threshold <= evaluations_budget:
            history.append((next_log_threshold, best_global_fitness))
            print(f"Evals: {next_log_threshold}, MaxFit: {best_global_fitness:.6f}")
            next_log_threshold += log_interval

        if evals_since_migration >= migration_interval:
            evals_since_migration = 0
            subpopulations = migration(subpopulations, M)

    if not history or history[-1][0] < evaluations_budget:
        history.append((evaluations_budget, best_global_fitness))

    return history, best_global_genotype