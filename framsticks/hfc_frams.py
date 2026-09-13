from __future__ import annotations

import random
import statistics
from bisect import bisect_right

from utility_frams import (
    initialize_population_frams,
    tournament_selection,
    crossover_frams,
    mutate_frams,
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

def tournament_deletion(subpopulation, tournament_size=3):
    if not subpopulation:
        return None
    sample = random.sample(subpopulation, min(tournament_size, len(subpopulation)))
    return min(sample, key=lambda x: x.fitness if x.fitness is not None else -float("inf"))

def trim_to_size(subpopulation, size, tournament_size=3):
    while len(subpopulation) > size:
        worst = tournament_deletion(subpopulation, tournament_size)
        if worst in subpopulation:
            subpopulation.remove(worst)

def compute_adm_boundaries(subpopulations, M, fixed_worst_export_threshold):
    if M <= 1:
        return []
    all_inds = [ind for sub in subpopulations for ind in sub if ind.fitness is not None]
    if not all_inds:
        return [fixed_worst_export_threshold] * (M - 1)
    fits = [ind.fitness for ind in all_inds]
    fmax = max(fits)
    sigma = statistics.pstdev(fits) if len(fits) > 1 else 0.0
    top_threshold = fmax - sigma
    if top_threshold < fixed_worst_export_threshold:
        top_threshold = fixed_worst_export_threshold
    if M == 2:
        return [top_threshold]
    step = (top_threshold - fixed_worst_export_threshold) / (M - 2)
    return [fixed_worst_export_threshold + i * step for i in range(M - 1)]

def get_tier_from_boundaries(fitness, boundaries):
    if fitness is None or not boundaries:
        return 0
    return bisect_right(boundaries, fitness)

def export_to_buffers(subpopulations, boundaries, admission_buffers):
    M = len(subpopulations)
    for tier_idx in range(M - 1):
        export_threshold = boundaries[tier_idx]
        survivors = []
        for ind in subpopulations[tier_idx]:
            if ind.fitness is not None and ind.fitness > export_threshold:
                target_tier = get_tier_from_boundaries(ind.fitness, boundaries)
                target_tier = max(target_tier, tier_idx + 1)
                target_tier = min(target_tier, M - 1)
                admission_buffers[target_tier].append(ind)
            else:
                survivors.append(ind)
        subpopulations[tier_idx] = survivors

def run_hfc(params, evaluations_budget, wrapper_func, framsLib):
    M = params["m_subpopulations"]
    R = params["r_interval"]
    TOURNAMENT = params["tournament_size"]
    CROSSOVER = params["crossover_prob"]
    MUTATION = params["mutation_prob"]
    TOTAL_POP = params["pop_size"]

    if M < 1:
        raise ValueError("m_subpopulations must be at least 1.")

    SIZE = max(2, TOTAL_POP // M)

    population = initialize_population_frams(TOTAL_POP, framsLib)
    evaluations = 0

    for ind in population:
        if ind.fitness is None:
            wrapper_func(ind)
            evaluations += 1

    valid_fits = [ind.fitness for ind in population if ind.fitness is not None]
    if not valid_fits:
        raise RuntimeError("Calibration population has no valid fitness values.")

    best_global_fitness = max(valid_fits)
    best_global_genotype = max(
        (ind for ind in population if ind.fitness is not None),
        key=lambda ind: ind.fitness,
    ).genotype
    fixed_worst_export_threshold = sum(valid_fits) / len(valid_fits)

    subpopulations = [[] for _ in range(M)]
    subpopulations[0] = population
    boundaries = compute_adm_boundaries(subpopulations, M, fixed_worst_export_threshold)

    subpopulations = [[] for _ in range(M)]
    for ind in population:
        tier = get_tier_from_boundaries(ind.fitness, boundaries)
        tier = min(max(tier, 0), M - 1)
        subpopulations[tier].append(ind)

    for i in range(M):
        trim_to_size(subpopulations[i], SIZE, TOURNAMENT)

    admission_buffers = [[] for _ in range(M)]
    history = []
    log_interval = 1000
    next_log_threshold = log_interval

    evals_since_migration = 0
    migration_interval = M * SIZE * R

    print(f"Start HFC-ADM steady-state (M={M}, R={R}, pop={TOTAL_POP}, migration every {migration_interval} evals)")

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
            worst = tournament_deletion(subpopulations[tier_idx], TOURNAMENT)
            if worst in subpopulations[tier_idx]:
                subpopulations[tier_idx].remove(worst)

        while evaluations >= next_log_threshold and next_log_threshold <= evaluations_budget:
            history.append((next_log_threshold, best_global_fitness))
            print(f"Evals: {next_log_threshold}, MaxFit: {best_global_fitness:.6f}")
            next_log_threshold += log_interval

        if evals_since_migration >= migration_interval:
            evals_since_migration = 0
            boundaries = compute_adm_boundaries(subpopulations, M, fixed_worst_export_threshold)
            export_to_buffers(subpopulations, boundaries, admission_buffers)

            moved_limit = max(1, SIZE // 2)
            for lvl in range(1, M):
                if admission_buffers[lvl]:
                    admission_buffers[lvl].sort(
                        key=lambda x: x.fitness if x.fitness is not None else -float("inf"),
                        reverse=True,
                    )
                    moved = admission_buffers[lvl][:moved_limit]
                    subpopulations[lvl].extend(moved)
                    admission_buffers[lvl] = []
                    trim_to_size(subpopulations[lvl], SIZE, TOURNAMENT)

            while len(subpopulations[0]) < SIZE and evaluations < evaluations_budget:
                recruit = initialize_population_frams(1, framsLib)[0]
                if recruit.fitness is None:
                    wrapper_func(recruit)
                    evaluations += 1
                    evals_since_migration += 1

                subpopulations[0].append(recruit)
                if recruit.fitness is not None and recruit.fitness > best_global_fitness:
                    best_global_fitness = recruit.fitness
                    best_global_genotype = recruit.genotype

            admission_buffers = [[] for _ in range(M)]
            for i in range(M):
                trim_to_size(subpopulations[i], SIZE, TOURNAMENT)

    if not history or history[-1][0] < evaluations_budget:
        history.append((evaluations_budget, best_global_fitness))

    return history, best_global_genotype