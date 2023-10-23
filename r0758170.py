import random as rd
import numpy as np
import Reporter
import math


def mutate_inversion(candidate):
    """Mutate a candidate solution in place using inversion mutation."""
    size = candidate.size
    if size <= 1:
        return
    first_pos = rd.randint(0, size - 2)
    second_pos = rd.randint(first_pos, size - 1)
    candidate[first_pos:second_pos + 1] = np.flip(candidate[first_pos:second_pos + 1])


def recombine_edge_crossover(parent1, parent2):
    """Use two parent candidates to produce offspring using edge crossover."""
    # 1. Construct the adjacency table.
    adj_table = create_adj_table(parent1, parent2)
    choices = [x for x in parent1]
    partial_result = []
    # 2 & 3. Pick an initial element at random and put it in the offspring.
    #        Set the variable current_element = entry.
    current_element = rd.choice(choices)
    while len(choices) != 0:
        choices.remove(current_element)
        partial_result.append(current_element)
        # 4. Remove all references to current_element from the table.
        for x, adj_list in adj_table.items():
            for y, is_common in adj_list:
                if current_element == y:
                    adj_list.remove((y, is_common))
        # 5. Examine list for current element.
        current_element_adj_list = adj_table[current_element]
        # If there is a common edge, pick that to be the next element.
        next_element = None
        for x, is_common in current_element_adj_list:
            if is_common:
                next_element = x
                break
        if not next_element:
            # Otherwise pick the entry in the list which itself has the shortest list.
            options = []
            shortest_length = math.inf
            for x, adj_list in adj_table.items():
                if x not in [y for y, _ in current_element_adj_list]:
                    continue
                if len(adj_list) <= shortest_length:
                    shortest_length = len(adj_list)
                    options.append(x)
            # 6. In the case of an empty list, the other end of the offspring is examined for extension;
            # otherwise a new element is chosen at random.
            if len(options) == 0:
                # TODO I don't know what that means? Just picking at random for now.
                if len(choices) == 0:
                    break
                next_element = rd.choice(choices)
            else:
                next_element = rd.choice(options)
        current_element = next_element
    return np.array(partial_result)


def create_adj_table(candidate1, candidate2):
    """Create an adjacency table for candidate1 and candidate2."""
    adj_table = {x: [] for x in candidate1}
    for x in adj_table:
        adj_in_parent1 = get_adj(x, candidate1)
        adj_in_parent2 = get_adj(x, candidate2)
        for y in adj_in_parent1:
            if y in adj_in_parent2:
                adj_table[x].append((y, True))
                adj_in_parent2.remove(y)
            else:
                adj_table[x].append((y, False))
        for y in adj_in_parent2:
            adj_table[x].append((y, False))
    return adj_table


def get_adj(x, candidate):
    """Returns the adjacent values of x in candidate as a list."""
    x_idx = index_of(candidate, x)
    prev_idx = x_idx-1
    next_idx = x_idx+1 if x_idx < candidate.size-1 else 0
    return [candidate[prev_idx], candidate[next_idx]]


def recombine_PMX(parent1, parent2):
    """Use two parent candidates to produce one offspring using partially mapped crossover.
    See p.70-71 in Eiben & Smith.
    """
    size = parent1.size
    offspring = np.zeros_like(parent1)
    # We must initialize offspring with -1's, to identify whether a spot is not yet filled.
    for i in range(size):
        offspring[i] = -1
    # 1.
    first_pos = rd.randint(0, size - 2)
    second_pos = rd.randint(first_pos, size - 1)
    offspring[first_pos:second_pos + 1] = parent1[first_pos:second_pos + 1]
    # 2. -  5.
    for elem in parent2[first_pos:second_pos + 1]:
        if elem in parent1[first_pos:second_pos + 1]:
            continue  # elem already occurs in offspring
        # elem is not yet in offspring, find the index to place it
        index = 0
        value = elem
        while value != -1:
            index = index_of(parent2, value)
            value = offspring[index]
        offspring[index] = elem
    # 6.
    for i in range(size):
        if offspring[i] == -1:
            offspring[i] = parent2[i]
    return offspring


def index_of(array, value):
    """Return the first index at which value occurs in array."""
    return np.where(array == value)[0][0]


def length(candidate, distance_matrix):
    """Calculate the length of the path of candidate."""
    result = 0.0
    size = candidate.size
    for i in range(size-1):
        result += distance_matrix[candidate[i]][candidate[i+1]]
    result += distance_matrix[candidate[size-1]][candidate[0]]  # Order matters.
    return result


def monte_carlo(distance_matrix, population_size):
    """Initializes the population at random."""
    population = []
    for i in range(population_size):
        gene = np.array(list(range(len(distance_matrix))))
        np.random.shuffle(gene)
        population.append(gene)
    return population


def avoid_inf(distance_matrix, population_size):
    """Initializes the population using a heuristic which tries to avoid infinite values."""
    population = []
    for i in range(population_size):
        start = rd.randrange(0, len(distance_matrix))
        candidate = [start]
        for j in range(len(distance_matrix)-1):
            possible_next = [
                x for x in range(len(distance_matrix))
                if x not in candidate and distance_matrix[candidate[-1]][x] != math.inf
            ]
            if len(possible_next) == 0:  # If there is no next city without inf distance, pick random.
                # TODO Modify the previous value / pick different start when this happens.
                possible_next = [x for x in range(len(distance_matrix)) if x not in candidate]
            next_city = rd.choice(possible_next)
            candidate.append(next_city)
        population.append(np.array(candidate))
    return population


def k_tournament(population, k, fitness_function, distance_matrix):
    """Performs a k-tournament on the population. Returns one candidate."""
    selected = []
    for i in range(k):
        selected.append(rd.choice(population))
    return max(selected, key=lambda x: fitness_function(x, distance_matrix))


# Modify the class name to match your student number.
class r0758170:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        # rd.seed(2023)  # During testing, set the seed for reproducible results.
        self.k = 5
        self.population = []
        self.population_size = 100
        self.mu = 20  # Must be even.
        self.mutate_chance = 0.05
        self.mutation_function = mutate_inversion
        self.recombine_function = recombine_PMX
        self.fitness_function = length
        self.init_function = avoid_inf
        self.selection = k_tournament

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        # Initialization
        self.population = self.init_function(distanceMatrix, self.population_size)
        # print(f'Initial pop: \n{[self.fitness_function(x, distanceMatrix) for x in self.population]}')

        # maxIt = 500
        # current_it = 0
        # while current_it < maxIt:
        while True:
            # Selection
            # Perform a certain number of k-tournaments; this depends on self.mu
            # and whether the recombination operator returns one or two offspring.
            # One offspring: need 2 * self.mu selected.
            # Two offspring: need self.mu selected.
            selected = []
            for i in range(2*self.mu):
                selected.append(k_tournament(self.population, self.k, self.fitness_function, distanceMatrix))

            # Variation
            # Recombination will produce self.mu new offspring.
            new_offspring = []
            it = iter(selected)
            for p1 in it:
                p2 = next(it)
                offspring = self.recombine_function(p1, p2)
                new_offspring.append(offspring)
            self.population.extend(new_offspring)

            # Mutation will happend on the entire population and new offspring, with a certain probability.
            for candidate in self.population:
                if rd.random() < self.mutate_chance:
                    self.mutation_function(candidate)

            # Elimination
            # Lambda + mu elimination: keep only the lambda best candidates.
            self.population.sort(key=lambda x: self.fitness_function(x, distanceMatrix))
            self.population = self.population[:self.population_size]

            # print(f'New pop: \n{[self.fitness_function(x, distanceMatrix) for x in self.population]}')

            # Recalculate mean and best.
            meanObjective = 0.0
            bestObjective = math.inf
            bestSolution = self.population[0]
            for candidate in self.population:
                candidate_fitness = self.fitness_function(candidate, distanceMatrix)
                meanObjective += candidate_fitness
                if candidate_fitness < bestObjective:
                    bestObjective = candidate_fitness
                    bestSolution = candidate
            meanObjective = meanObjective / self.population_size

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break
            # current_it += 1

        # Your code here.
        return 0
