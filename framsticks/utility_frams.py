import random

class Individual:
    def __init__(self, genotype):
        self.genotype = genotype
        self.fitness = None

    def __repr__(self):
        fit_str = f"{self.fitness:.6f}" if self.fitness is not None else "N/A"
        short_geno = (self.genotype[:20] + '..') if len(self.genotype) > 20 else self.genotype
        return f"Ind(Fit: {fit_str}, Geno: {short_geno})"

def initialize_population_frams(size, framsLib):
    population = []
    if size > 1:
        print(f"Initializing population (size={size}, parts=2-15, neurons=0-20, iter_max=100)")

    initial_genotype = "/*1*/" + framsLib.getSimplest("1")

    for _ in range(size):
        geno = framsLib.getRandomGenotype(
            initial_genotype=initial_genotype,
            parts_min=2,
            parts_max=15,
            neurons_min=0,
            neurons_max=20,
            iter_max=100,
            return_even_if_failed=True,
        )

        if geno is None:
            geno = "X"
        elif isinstance(geno, str) and geno.startswith("/*1*/"):
            geno = geno[5:]

        population.append(Individual(geno))
    return population

def tournament_selection(population, size):
    if not population: return None
    candidates = random.sample(population, min(size, len(population)))
    valid = [ind for ind in candidates if ind.fitness is not None]
    if not valid: return candidates[0]
    winner = max(valid, key=lambda x: x.fitness)
    return winner

def random_deletion(population):
    return random.choice(population)

def crossover_frams(parent1, parent2, prob, framsLib):
    if random.random() < prob:
        child_geno = framsLib.crossOver(f"/*1*/{parent1.genotype}", f"/*1*/{parent2.genotype}")
        if child_geno:
            if child_geno.startswith("/*1*/"):
                child_geno = child_geno[5:]
            if child_geno != framsLib.GENOTYPE_INVALID:
                return Individual(child_geno)
    return Individual(parent1.genotype)

def mutate_frams(individual, prob, framsLib):
    if random.random() < prob:
        res = framsLib.mutate([f"/*1*/{individual.genotype}"])
        if res and len(res) > 0:
            mutated = res[0]
            if mutated.startswith("/*1*/"):
                mutated = mutated[5:]
            if mutated != framsLib.GENOTYPE_INVALID:
                individual.genotype = mutated
                individual.fitness = None
    return individual