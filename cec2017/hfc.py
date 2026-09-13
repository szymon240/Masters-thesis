import random
import statistics
from bisect import bisect_left

from utility import (
    initializePopulation,
    tournament_selection,
    crossover,
    mutate
)

def tournament_deletion(subpopulation, tournament_size=3):
    if not subpopulation:
        return None
    sample = random.sample(subpopulation, min(tournament_size, len(subpopulation)))
    return max(sample, key=lambda x: x.fitness if x.fitness is not None else float("inf"))

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
    fmin = min(fits)
    sigma = statistics.pstdev(fits) if len(fits) > 1 else 0.0

    top_threshold = fmin + sigma
    if top_threshold > fixed_worst_export_threshold:
        top_threshold = fixed_worst_export_threshold

    if M == 2:
        return [top_threshold]

    step = (fixed_worst_export_threshold - top_threshold) / (M - 2)
    return [fixed_worst_export_threshold - i * step for i in range(M - 1)]

def get_tier_from_boundaries(fitness, boundaries):
    if fitness is None or not boundaries:
        return 0
    for i, b in enumerate(boundaries):
        if fitness > b:
            return i
    return len(boundaries)

def export_to_buffers(subpopulations, boundaries, admission_buffers):
    M = len(subpopulations)
    for tier_idx in range(M - 1):
        export_threshold = boundaries[tier_idx]
        survivors = []
        for ind in subpopulations[tier_idx]:
            if ind.fitness is not None and ind.fitness < export_threshold:
                target_tier = get_tier_from_boundaries(ind.fitness, boundaries)
                target_tier = max(target_tier, tier_idx + 1)
                target_tier = min(target_tier, M - 1)
                admission_buffers[target_tier].append(ind)
            else:
                survivors.append(ind)
        subpopulations[tier_idx] = survivors

def run_hfc(params, evaluations_budget, wrapper_func, D, MIN, MAX):
    M = params["m_subpopulations"]
    R = params["r_interval"]
    TOURNAMENT = params["tournament_size"]
    CROSSOVER = params["crossover_prob"]
    MUTATION = params["mutation_prob"]
    TOTAL_POP = params["pop_size"]
    ALPHA = params["alpha"]

    if M < 1:
        raise ValueError("m_subpopulations must be at least 1.")

    SIZE = max(2, TOTAL_POP // M)

    population = initializePopulation(TOTAL_POP, MIN, MAX, wrapper_func, D)
    evaluations = TOTAL_POP

    valid_fits = [ind.fitness for ind in population if ind.fitness is not None]
    if not valid_fits:
        raise RuntimeError("Calibration population has no valid fitness values.")

    best_ind = min(
        (ind for ind in population if ind.fitness is not None),
        key=lambda ind: ind.fitness,
    )
    best_global_fitness = best_ind.fitness
    best_global_solution = list(best_ind.genes)
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
    log_interval = max(1, evaluations_budget // 100)
    next_log_threshold = log_interval

    evals_since_migration = 0
    migration_interval = M * SIZE * R

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
        actual_tourn = min(TOURNAMENT, len(sub))

        p1 = tournament_selection(sub, actual_tourn)
        p2 = tournament_selection(sub, actual_tourn)
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
            worst = tournament_deletion(subpopulations[tier_idx], TOURNAMENT)
            if worst in subpopulations[tier_idx]:
                subpopulations[tier_idx].remove(worst)

        while evaluations >= next_log_threshold and next_log_threshold <= evaluations_budget:
            history.append((next_log_threshold, best_global_fitness))
            next_log_threshold += log_interval

        if evals_since_migration >= migration_interval:
            evals_since_migration = 0
            boundaries = compute_adm_boundaries(subpopulations, M, fixed_worst_export_threshold)
            export_to_buffers(subpopulations, boundaries, admission_buffers)

            moved_limit = max(1, SIZE // 2)
            for lvl in range(1, M):
                if admission_buffers[lvl]:
                    admission_buffers[lvl].sort(
                        key=lambda x: x.fitness if x.fitness is not None else float("inf"),
                        reverse=False,
                    )
                    moved = admission_buffers[lvl][:moved_limit]
                    subpopulations[lvl].extend(moved)
                    admission_buffers[lvl] = []
                    trim_to_size(subpopulations[lvl], SIZE, TOURNAMENT)

            while len(subpopulations[0]) < SIZE and evaluations < evaluations_budget:
                recruit = initializePopulation(1, MIN, MAX, wrapper_func, D)[0]
                evaluations += 1
                evals_since_migration += 1

                subpopulations[0].append(recruit)
                if recruit.fitness is not None and recruit.fitness < best_global_fitness:
                    best_global_fitness = recruit.fitness
                    best_global_solution = list(recruit.genes)

            admission_buffers = [[] for _ in range(M)]
            for i in range(M):
                trim_to_size(subpopulations[i], SIZE, TOURNAMENT)

    if not history or history[-1][0] < evaluations_budget:
        history.append((evaluations_budget, best_global_fitness))

    return history, best_global_solution