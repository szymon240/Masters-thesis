import random

import numpy as np

from utility_frams import (
    initialize_population_frams,
    tournament_selection,
    crossover_frams,
    mutate_frams,
    Individual,
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


def _serialize_subpopulations(subpopulations):
    return [
        [{"genotype": ind.genotype, "fitness": ind.fitness} for ind in sub]
        for sub in subpopulations
    ]


def _deserialize_subpopulations(serialized):
    subs = []
    for sub_data in serialized:
        sub = []
        for ind_data in sub_data:
            ind = Individual(ind_data["genotype"])
            ind.fitness = ind_data["fitness"]
            sub.append(ind)
        subs.append(sub)
    return subs


def _serialize_random_state():
    py_state = random.getstate()
    py_state_json = [py_state[0], list(py_state[1]), py_state[2]]

    np_state = np.random.get_state()
    np_state_json = [
        np_state[0],
        np_state[1].tolist(),
        int(np_state[2]),
        int(np_state[3]),
        float(np_state[4]),
    ]
    return {"py": py_state_json, "np": np_state_json}


def _deserialize_random_state(rng_state):
    py = rng_state["py"]
    random.setstate((py[0], tuple(py[1]), py[2]))

    np_s = rng_state["np"]
    np.random.set_state(
        (np_s[0], np.array(np_s[1], dtype=np.uint32), np_s[2], np_s[3], np_s[4])
    )

def run_hybrid(
    params,
    evaluations_budget,
    wrapper_func,
    framsLib,
    resume_state=None,
    mid_checkpoint_callback=None,
    mid_checkpoint_interval=50000,
):
    """ConvSel with equiwidth migration and a full random refresh of tier 0 at every migration."""
    M = params['m_subpopulations']
    R = params['r_interval']
    TOURNAMENT = params['tournament_size']
    CROSSOVER = params['crossover_prob']
    MUTATION = params['mutation_prob']
    TOTAL_POP = params['pop_size']
    SIZE = max(2, TOTAL_POP // M)

    log_interval = 1000
    migration_interval = TOTAL_POP * R

    if resume_state is not None:
        evaluations = int(resume_state["evaluations"])
        best_global_fitness = float(resume_state["best_global_fitness"])
        best_global_genotype = resume_state["best_global_genotype"]
        history = [(int(e), float(f)) for e, f in resume_state["history"]]
        evals_since_migration = int(resume_state["evals_since_migration"])
        next_log_threshold = int(resume_state["next_log_threshold"])
        subpopulations = _deserialize_subpopulations(resume_state["subpopulations"])
        _deserialize_random_state(resume_state["random_state"])
        print(
            f"Resumed Hybrid (CS+tier0-refresh) at evals={evaluations}/{evaluations_budget}, "
            f"best={best_global_fitness:.6f}"
        )

        if evaluations >= evaluations_budget:
            if not history or history[-1][0] < evaluations_budget:
                history.append((evaluations_budget, best_global_fitness))
            if mid_checkpoint_callback is not None:
                mid_checkpoint_callback({
                    "evaluations": evaluations,
                    "best_global_fitness": best_global_fitness,
                    "best_global_genotype": best_global_genotype,
                    "history": [[int(e), float(f)] for e, f in history],
                    "evals_since_migration": evals_since_migration,
                    "next_log_threshold": next_log_threshold,
                    "subpopulations": _serialize_subpopulations(subpopulations),
                    "random_state": _serialize_random_state(),
                })
            return history, best_global_genotype
    else:
        flat_population = initialize_population_frams(TOTAL_POP, framsLib)
        evaluations = 0

        for ind in flat_population:
            if ind.fitness is None:
                wrapper_func(ind)
                evaluations += 1

        valid_fits = [ind.fitness for ind in flat_population if ind.fitness is not None]
        if not valid_fits:
            raise RuntimeError("Start population has no valid fitness values.")

        best_global_fitness = max(valid_fits)
        best_global_genotype = max(
            (ind for ind in flat_population if ind.fitness is not None),
            key=lambda ind: ind.fitness,
        ).genotype

        subpopulations = [[] for _ in range(M)]
        subpopulations[0] = flat_population
        subpopulations = migration(subpopulations, M)

        history = []
        next_log_threshold = log_interval
        evals_since_migration = 0

        print(f"Start Hybrid (CS+tier0-refresh) steady-state (M={M}, R={R}, pop={TOTAL_POP})")

    last_checkpoint_evals = evaluations

    def _emit_checkpoint():
        if mid_checkpoint_callback is None:
            return
        mid_checkpoint_callback({
            "evaluations": evaluations,
            "best_global_fitness": best_global_fitness,
            "best_global_genotype": best_global_genotype,
            "history": [[int(e), float(f)] for e, f in history],
            "evals_since_migration": evals_since_migration,
            "next_log_threshold": next_log_threshold,
            "subpopulations": _serialize_subpopulations(subpopulations),
            "random_state": _serialize_random_state(),
        })

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

            refill_count = len(subpopulations[0])
            subpopulations[0] = []
            for _ in range(refill_count):
                if evaluations >= evaluations_budget:
                    break
                recruit = initialize_population_frams(1, framsLib)[0]
                if recruit.fitness is None:
                    wrapper_func(recruit)
                    evaluations += 1
                    evals_since_migration += 1

                subpopulations[0].append(recruit)
                if recruit.fitness is not None and recruit.fitness > best_global_fitness:
                    best_global_fitness = recruit.fitness
                    best_global_genotype = recruit.genotype

                while evaluations >= next_log_threshold and next_log_threshold <= evaluations_budget:
                    history.append((next_log_threshold, best_global_fitness))
                    print(f"Evals: {next_log_threshold}, MaxFit: {best_global_fitness:.6f}")
                    next_log_threshold += log_interval

        if evaluations - last_checkpoint_evals >= mid_checkpoint_interval:
            last_checkpoint_evals = evaluations
            _emit_checkpoint()

    if not history or history[-1][0] < evaluations_budget:
        history.append((evaluations_budget, best_global_fitness))

    _emit_checkpoint()

    return history, best_global_genotype
