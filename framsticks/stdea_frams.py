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

def run_stdea(params, evaluations_budget, wrapper_func, framsLib):
    """Steady-state EA with tournament selection and random deletion."""
    TOURNAMENT = params['tournament_size']
    CROSSOVER = params['crossover_prob']
    MUTATION = params['mutation_prob']
    TOTAL_POP = params['pop_size']

    # 1. Inicjalizacja i ocena populacji startowej
    population = initialize_population_frams(TOTAL_POP, framsLib)
    evaluations = 0

    for ind in population:
        if ind.fitness is None:
            wrapper_func(ind)
            evaluations += 1

    valid_fits = [ind.fitness for ind in population if ind.fitness is not None]
    if not valid_fits:
        raise RuntimeError("Start population has no valid fitness values.")

    best_global_fitness = max(valid_fits)
    best_global_genotype = max(
        (ind for ind in population if ind.fitness is not None),
        key=lambda ind: ind.fitness,
    ).genotype

    history = []
    log_interval = 1000
    next_log_threshold = log_interval

    print(f"Start StdEA steady-state (pop={TOTAL_POP}, tourn={TOURNAMENT})")

    # 2. Główna pętla ewolucyjna
    while evaluations < evaluations_budget:
        actual_tourn = min(TOURNAMENT, len(population))

        # Wybór rodziców turniejem
        p1 = tournament_selection(population, actual_tourn)
        p2 = tournament_selection(population, actual_tourn)

        child = crossover_offspring(p1, p2, CROSSOVER, framsLib)
        child = mutate_offspring(child, MUTATION, framsLib)

        # Ewaluacja dziecka
        if child.fitness is None:
            wrapper_func(child)
            evaluations += 1

        # Aktualizacja globalnego mistrza
        if child.fitness is not None and child.fitness > best_global_fitness:
            best_global_fitness = child.fitness
            best_global_genotype = child.genotype

        # Losowe usuwanie (Random Deletion) PRZED dodaniem dziecka:
        # gwarantuje, że nowo utworzony osobnik zawsze wchodzi do populacji.
        remove = random.choice(population)
        population.remove(remove)
        population.append(child)

        # Logowanie historii
        while evaluations >= next_log_threshold and next_log_threshold <= evaluations_budget:
            history.append((next_log_threshold, best_global_fitness))
            print(f"Evals: {next_log_threshold}, MaxFit: {best_global_fitness:.6f}")
            next_log_threshold += log_interval

    # Zabezpieczenie dodania ostatniego punktu do historii
    if not history or history[-1][0] < evaluations_budget:
        history.append((evaluations_budget, best_global_fitness))

    return history, best_global_genotype