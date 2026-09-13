import math
import random


class Individual:
    def __init__(self, genes):
        self.genes = genes
        self.fitness = None

    def __repr__(self):
        fit_str = f"{self.fitness:.6f}" if self.fitness is not None else "N/A"
        return f"Individual(D={len(self.genes)}, Fitness: {fit_str})"


def drop_wave(individual):
    x1 = individual.genes[0]
    x2 = individual.genes[1]

    individual.fitness = - ((1 + math.cos(12 * math.sqrt(x1**2 + x2**2)))/(0.5 * (x1**2 + x2**2) + 2))

def shaffer_n2(individual):
    x1 = individual.genes[0]
    x2 = individual.genes[1]

    individual.fitness = 0.5 + ((math.pow(math.sin(x1 ** 2 + x2 ** 2), 2) - 0.5) / (math.pow(1 + 0.001 * (x1 ** 2 + x2 ** 2), 2)))

def shaffer_n4(individual):
    x1 = individual.genes[0]
    x2 = individual.genes[1]

    individual.fitness = 0.5 + ((math.pow(math.cos(math.sin(abs(x1 ** 2 - x2 ** 2))), 2) - 0.5) / (math.pow(1 + 0.001 * (x1 ** 2 + x2 ** 2), 2)))

def initializePopulation(size, domain_min, domain_max, fitness, D):
    population = []
    for i in range(size):
        genes = [
            random.uniform(domain_min, domain_max) for i in range(D)
        ]

        individual = Individual(genes)
        fitness(individual)
        population.append(individual)

    return population

def tournament_selection(population, size):
    tournament = []
    for i in range(size):
        chosen = random.choice(population)
        tournament.append(chosen)
    winner = min(tournament, key=lambda x: x.fitness)
    return winner

def random_deletion(population):
    chosen = random.choice(population)
    return chosen

def crossover(x1, x2, prob):
    if random.random() < prob:
        genes = []
        for i in range(len(x1.genes)):
            gene = random.choice([x1.genes[i], x2.genes[i]])
            genes.append(gene)
        child = Individual(genes)
    else:
        child = Individual(list(x1.genes))
    return child


def mutate(individual, prob, range_val, domain_min, domain_max):
    for i in range(len(individual.genes)):
        if random.random() < prob:
            change = random.uniform(-range_val, range_val)
            new_value = individual.genes[i] + change

            new_value = max(domain_min, new_value)
            new_value = min(domain_max, new_value)

            individual.genes[i] = new_value
    return individual