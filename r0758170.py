import random as rd

import math
import numpy as np

import Reporter


# SEED = 2
# rd.seed(SEED)
# np.random.seed(SEED)

class Candidate:

    def __init__(self, array):
        self.array = array

    def __repr__(self):
        return str(self.array)

    def __len__(self):
        return len(self.array)

    def mutate_inversion(self):
        """Mutate a candidate solution in-place using inversion mutation."""
        size = len(self)
        first_pos = rd.randrange(0, size - 1)
        second_pos = rd.randrange(first_pos, size)
        self.array[first_pos:second_pos + 1] = np.flip(self.array[first_pos:second_pos + 1])


def mutate_swap(candidate):
    """Mutate a candidate solution in-place using swap mutation."""
    size = candidate.size
    first_pos = rd.randrange(0, size)
    second_pos = first_pos
    while second_pos == first_pos:
        second_pos = rd.randrange(0, size)
    tmp = candidate[first_pos]
    candidate[first_pos] = candidate[second_pos]
    candidate[second_pos] = tmp


def mutate_scramble(candidate):
    """Mutate a candidate solution in-place using scramble mutation."""
    size = candidate.size
    first_pos = rd.randrange(0, size - 1)
    second_pos = rd.randrange(first_pos, size)
    np.random.shuffle(candidate[first_pos:second_pos + 1])


def mutate_insert(candidate):
    """Mutate a candidate in-place using insert mutation."""
    raise NotImplementedError


def recombine_cycle_crossover(parent1, parent2):
    """Use two parent candidates to produce two offspring using cycle crossover."""
    cycles = find_cycles(parent1, parent2)
    offspring1 = np.zeros_like(parent1)
    offspring2 = np.zeros_like(parent2)
    for i, cycle in enumerate(cycles):
        if i % 2 == 0:
            for idx in cycle:
                offspring1[idx] = parent1[idx]
                offspring2[idx] = parent2[idx]
        else:
            for idx in cycle:
                offspring1[idx] = parent2[idx]
                offspring2[idx] = parent1[idx]
    return [offspring1, offspring2]


def find_cycles(parent1, parent2):
    """Returns all cycles of parent1 and parent2 as a list of lists of indices."""
    unused_idx = list(range(len(parent1)))
    cycles = []
    while len(unused_idx) != 0:
        start_idx = unused_idx[0]
        current_idx: int = start_idx
        unused_idx.remove(current_idx)
        cycle = [current_idx]
        while True:
            allele_p2 = parent2[current_idx]
            current_idx = index_of(parent1, allele_p2)
            if current_idx == start_idx:
                break
            unused_idx.remove(current_idx)
            cycle.append(current_idx)
        cycles.append(cycle)
    return cycles


def recombine_edge_crossover(parent1, parent2):
    """Use two parent candidates to produce one offspring using edge crossover."""
    adj_table = create_adj_table(parent1, parent2)
    remaining = [x for x in parent1]
    current_element = rd.choice(remaining)
    result = [current_element]
    remaining.remove(current_element)
    remove_references(adj_table, current_element)
    while len(remaining) != 0:
        try:
            current_element = pick_next_element(adj_table, current_element)
            result.append(current_element)
            remaining.remove(current_element)
            remove_references(adj_table, current_element)
        except NoNextElementException:
            try:
                next_element = pick_next_element(adj_table, result[0])
                result.insert(0, next_element)
                remaining.remove(next_element)
                remove_references(adj_table, next_element)
            except NoNextElementException:
                current_element = rd.choice(remaining)
                result.append(current_element)
                remaining.remove(current_element)
                remove_references(adj_table, current_element)
    return [np.array(result)]


class NoNextElementException(Exception):
    """Exception used in edge crossover recombination."""


def pick_next_element(adj_table, current_element):
    """Returns the next element to extend the offspring with.
    Raises NoNextElementException if there is no next element to extend with.
    """
    lst = adj_table[current_element]
    if len(lst) == 0:
        raise NoNextElementException
    for x, is_common in lst:
        if is_common:
            return x
    next_element_options = []
    shortest_len = math.inf
    for x, is_common in lst:
        x_lst_len = len(adj_table[x])
        if x_lst_len < shortest_len:
            next_element_options = [x]
            shortest_len = x_lst_len
        elif x_lst_len == shortest_len:
            next_element_options.append(x)
    next_element = rd.choice(next_element_options)
    return next_element


def remove_references(adj_table, value):
    """Removes all references of value in the lists of adj_table."""
    for x, lst in adj_table.items():
        if x == value:
            continue  # We can skip this case because value cannot be adjacent to itself.
        for y, is_common in lst:
            if value == y:
                lst.remove((y, is_common))
                break


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
    prev_idx = x_idx - 1
    next_idx = x_idx + 1 if x_idx < candidate.size - 1 else 0
    return [candidate[prev_idx], candidate[next_idx]]


def recombine_PMX(parent1, parent2):
    """Use two parent candidates to produce one offspring using partially mapped crossover."""
    # TODO Refactor this to produce two offspring.
    size = parent1.size
    offspring = np.zeros_like(parent1)
    # We must initialize offspring with -1's, to identify whether a spot is not yet filled.
    for i in range(size):
        offspring[i] = -1
    first_pos = rd.randint(0, size - 2)
    second_pos = rd.randint(first_pos, size - 1)
    offspring[first_pos:second_pos + 1] = parent1[first_pos:second_pos + 1]
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
    for i in range(size):
        if offspring[i] == -1:
            offspring[i] = parent2[i]
    return [offspring]


def recombine_order_crossover(parent1, parent2):
    """Use two parent candidates to produce one offspring using order crossover."""
    size = parent1.size
    offspring = np.zeros_like(parent1)

    first_pos = rd.randint(0, size - 2)
    second_pos = rd.randint(first_pos, size - 1)
    offspring[first_pos:second_pos + 1] = parent1[first_pos:second_pos + 1]

    raise NotImplemented


def index_of(array, value):
    """Return the first index at which value occurs in array.
    This is just a convenience function for numpy arrays, which behaves like list.index(value).
    """
    return int(np.where(array == value)[0][0])


def fitness_length(candidate, distance_matrix):
    """Calculate the length of the path of candidate."""
    result = 0.0
    size = candidate.size
    for i in range(size - 1):
        # Order is important for the distance matrix.
        result += distance_matrix[candidate[i]][candidate[i + 1]]
    result += distance_matrix[candidate[size - 1]][candidate[0]]
    return result


def init_monte_carlo(distance_matrix, population_size):
    """Initializes the population at random."""
    population = []
    for i in range(population_size):
        candidate = np.array(list(range(len(distance_matrix))))
        np.random.shuffle(candidate)
        population.append(candidate)
    return population


def init_avoid_inf_heuristic(distance_matrix, population_size):
    """Initializes the population using a heuristic which avoids infinite values."""
    population = []
    for i in range(population_size):
        choices = list(range(population_size))
        candidate = []
        while len(choices) != 0:
            if len(candidate) == 0:  # The first element is picked at random.
                choice = rd.choice(choices)
                candidate.append(choice)
                choices.remove(choice)
                continue
            possible_next = [
                x for x in choices
                if x not in candidate and distance_matrix[candidate[-1]][x] != math.inf
            ]
            if len(possible_next) == 0:
                # This leads to a dead end, backtrack the last choice and try again.
                choices.append(candidate[-1])
                candidate.remove(candidate[-1])
            else:
                # The path can be extended, pick a next element with preference for the greedy choice.
                if rd.random() < 0.15:
                    choice = min(possible_next, key=lambda x: distance_matrix[candidate[-1]][x])
                else:
                    choice = rd.choice(possible_next)
                candidate.append(choice)
                choices.remove(choice)
        population.append(np.array(candidate))
    return population


def select_k_tournament(population, k, fitness_function, distance_matrix):
    """Performs a k-tournament on the population. Returns one candidate."""
    selected = []
    for i in range(k):
        selected.append(rd.choice(population))
    return min(selected, key=lambda x: fitness_function(x, distance_matrix))


# Modify the class name to match your student number.
class r0758170:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.k = 5
        self.population = []
        self.population_size = 100
        self.nr_offspring = 100  # Must be even.
        self.mutate_chance = 0.05
        self.mutation_function = mutate_scramble
        self.recombine_function = recombine_edge_crossover
        self.fitness_function = fitness_length
        self.init_function = init_avoid_inf_heuristic
        self.selection = select_k_tournament

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        # Initialization
        self.population = self.init_function(distanceMatrix, self.population_size)

        best_solution = self.population[0]
        best_objective = self.fitness_function(best_solution, distanceMatrix)
        while True:
            # Selection
            # Perform a certain number of k-tournaments; this depends on self.mu
            # and whether the recombination operator returns one or two offspring.
            # One offspring: need 2 * self.mu selected.
            # Two offspring: need self.mu selected.
            selected = []
            for i in range(2 * self.nr_offspring):
                selected.append(select_k_tournament(self.population, self.k, self.fitness_function, distanceMatrix))

            # Variation
            # Recombination will produce new offspring using the selected candidates.
            new_offspring = []
            it = iter(selected)
            for p1 in it:
                p2 = next(it)
                offspring = self.recombine_function(p1, p2)
                new_offspring.extend(offspring)
            self.population.extend(new_offspring)

            # Mutation will happend on the entire population and new offspring, with a certain probability.
            for candidate in self.population:
                if rd.random() < self.mutate_chance:
                    self.mutation_function(candidate)

            # Elimination
            # Lambda + mu elimination: keep only the best candidates.
            self.population.sort(key=lambda x: self.fitness_function(x, distanceMatrix))
            self.population = self.population[:self.population_size]

            # Recalculate mean and best.
            mean_objective = 0.0
            current_best_solution = self.population[0]
            current_best_objective = self.fitness_function(current_best_solution, distanceMatrix)
            for candidate in self.population:
                candidate_fitness = self.fitness_function(candidate, distanceMatrix)
                mean_objective += candidate_fitness
                if candidate_fitness < current_best_objective:
                    current_best_objective = candidate_fitness
                    current_best_solution = candidate
            mean_objective = mean_objective / self.population_size
            if current_best_objective < best_objective:
                best_objective = current_best_objective
                best_solution = current_best_solution

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            # print(f'{current_it:5} | mean: {mean_objective:.2f} | best:{best_objective:.2f}')
            timeLeft = self.reporter.report(mean_objective, best_objective, best_solution)
            if timeLeft < 0:
                break

        # Your code here.
        return 0
