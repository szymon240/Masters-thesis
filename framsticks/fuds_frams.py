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

def fuds_deletion(population, bins):
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


def _serialize_population(population):
    """Konwersja populacji -> JSON-serializowalna struktura."""
    return [{"genotype": ind.genotype, "fitness": ind.fitness} for ind in population]


def _deserialize_population(serialized):
    """Odtworzenie populacji z JSON do listy Individual objektow."""
    pop = []
    for ind_data in serialized:
        ind = Individual(ind_data["genotype"])
        ind.fitness = ind_data["fitness"]
        pop.append(ind)
    return pop


def _serialize_random_state():
    """Konwersja stanu RNG (Python + NumPy) do JSON-safe dict."""
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

def run_fuds(
    params,
    evaluations_budget,
    wrapper_func,
    framsLib,
    resume_state=None,
    mid_checkpoint_callback=None,
    mid_checkpoint_interval=50000,
):
    """FUDS with tournament selection and deletion from the most crowded fitness bin."""
    SIZE = params['pop_size']
    TOURNAMENT = params['tournament_size']
    CROSSOVER = params['crossover_prob']
    MUTATION = params['mutation_prob']
    BINS = 20

    log_interval = 1000

    if resume_state is not None:
        evaluations = int(resume_state["evaluations"])
        best_global_fitness = float(resume_state["best_global_fitness"])
        best_global_genotype = resume_state["best_global_genotype"]
        history = [(int(e), float(f)) for e, f in resume_state["history"]]
        next_log_threshold = int(resume_state["next_log_threshold"])
        population = _deserialize_population(resume_state["population"])
        _deserialize_random_state(resume_state["random_state"])
        print(
            f"Resumed FUDS at evals={evaluations}/{evaluations_budget}, "
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
                    "next_log_threshold": next_log_threshold,
                    "population": _serialize_population(population),
                    "random_state": _serialize_random_state(),
                })
            return history, best_global_genotype
    else:
        population = initialize_population_frams(SIZE, framsLib)
        evaluations = 0

        for ind in population:
            if ind.fitness is None:
                wrapper_func(ind)
                evaluations += 1

        best_start_ind = max(population, key=lambda ind: ind.fitness if ind.fitness is not None else -float('inf'))
        best_global_fitness = best_start_ind.fitness
        best_global_genotype = best_start_ind.genotype

        history = []
        print(f"Start FUDS (budget: {evaluations_budget}, pop: {SIZE}, bins: {BINS})")

        next_log_threshold = log_interval

        while evaluations >= next_log_threshold:
            history.append((next_log_threshold, best_global_fitness))
            next_log_threshold += log_interval

    last_checkpoint_evals = evaluations

    def _emit_checkpoint():
        if mid_checkpoint_callback is None:
            return
        mid_checkpoint_callback({
            "evaluations": evaluations,
            "best_global_fitness": best_global_fitness,
            "best_global_genotype": best_global_genotype,
            "history": [[int(e), float(f)] for e, f in history],
            "next_log_threshold": next_log_threshold,
            "population": _serialize_population(population),
            "random_state": _serialize_random_state(),
        })

    while evaluations < evaluations_budget:
        x1 = tournament_selection(population, TOURNAMENT)
        x2 = tournament_selection(population, TOURNAMENT)

        child = crossover_offspring(x1, x2, CROSSOVER, framsLib)
        child = mutate_offspring(child, MUTATION, framsLib)

        if child.fitness is None:
            wrapper_func(child)
            evaluations += 1

        chosen = fuds_deletion(population, BINS)

        population.remove(chosen)
        population.append(child)

        if child.fitness is not None and child.fitness > best_global_fitness:
            best_global_fitness = child.fitness
            best_global_genotype = child.genotype

        while evaluations >= next_log_threshold and next_log_threshold <= evaluations_budget:
            history.append((next_log_threshold, best_global_fitness))
            print(f"Evals: {next_log_threshold}, MaxFit: {best_global_fitness:.4f}")
            next_log_threshold += log_interval

        if evaluations - last_checkpoint_evals >= mid_checkpoint_interval:
            last_checkpoint_evals = evaluations
            _emit_checkpoint()

    if not history or history[-1][0] < evaluations_budget:
        history.append((evaluations_budget, best_global_fitness))

    # FINAL CHECKPOINT - zapisz stan koncowy (potrzebne do rozszerzenia budzetu pozniej)
    _emit_checkpoint()

    return history, best_global_genotype