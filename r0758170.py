import random as rd

import math
import numpy as np
from numpy.typing import NDArray

import Reporter


class Candidate:
    def __init__(self, array):
        self.array = array
        self.size = array.size
        self.fitness = 0

    def __repr__(self):
        return str(self.array)

    def __len__(self):
        return len(self.array)

    def __iter__(self):
        return iter(self.array)

    def __getitem__(self, item):
        return self.array[item]

    def __setitem__(self, key, value):
        self.array[key] = value


class Parameters:
    def __init__(self):
        self.k = 5
        self.pop_size = 100
        self.nr_offspring = 20
        self.mutate_chance = 0.20
        self.mutate_func = mutate_inversion
        self.recombine_func = recombine_PMX
        self.fitness_func = fitness_length
        self.init_func = init_avoid_inf_heuristic
        self.select_func = select_k_tournament


def mutate_inversion(candidate: Candidate) -> None:
    """Mutate in-place using inversion mutation."""
    size = candidate.size
    first_pos = rd.randrange(0, size - 1)
    second_pos = rd.randrange(first_pos, size)
    candidate[first_pos:second_pos + 1] = np.flip(candidate[first_pos:second_pos + 1])


def mutate_swap(candidate: Candidate) -> None:
    """Mutate in-place using swap mutation."""
    size = candidate.size
    first_pos = rd.randrange(0, size)
    second_pos = first_pos
    while second_pos == first_pos:
        second_pos = rd.randrange(0, size)
    tmp = candidate[first_pos]
    candidate[first_pos] = candidate[second_pos]
    candidate[second_pos] = tmp


def mutate_scramble(candidate: Candidate) -> None:
    """Mutate in-place using scramble mutation."""
    size = candidate.size
    first_pos = rd.randrange(0, size - 1)
    second_pos = rd.randrange(first_pos, size)
    np.random.shuffle(candidate[first_pos:second_pos + 1])


def mutate_insert(candidate: Candidate) -> None:
    """Mutate in-place using insert mutation."""
    raise NotImplementedError


def fitness_length(candidate: Candidate, distance_matrix: NDArray) -> float:
    """Return the length of the path."""
    result = 0.0
    size = candidate.size
    for i in range(size - 1):
        # Order is important for the distance matrix.
        result += distance_matrix[candidate[i]][candidate[i + 1]]
    result += distance_matrix[candidate[size - 1]][candidate[0]]
    return result


class NoNextElementException(Exception):
    """Exception used in edge crossover recombination."""


def recombine_cycle_crossover(parent1: Candidate, parent2: Candidate) -> list[Candidate]:
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
    return [Candidate(offspring1), Candidate(offspring2)]


def find_cycles(parent1: Candidate, parent2: Candidate) -> list[list[int]]:
    """Returns all cycles of the parents using indices."""
    unused_idx = list(range(len(parent1)))
    cycles = []
    while len(unused_idx) != 0:
        start_idx = unused_idx[0]
        current_idx: int = start_idx
        unused_idx.remove(current_idx)
        cycle = [current_idx]
        while True:
            allele_p2 = int(parent2[current_idx])
            current_idx = index_of(parent1, allele_p2)
            if current_idx == start_idx:
                break
            unused_idx.remove(current_idx)
            cycle.append(current_idx)
        cycles.append(cycle)
    return cycles


def recombine_edge_crossover(parent1: Candidate, parent2: Candidate) -> list[Candidate]:
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
    return [Candidate(np.array(result))]


def pick_next_element(adj_table: dict[int, list[tuple[int, bool]]], current_element: int) -> int:
    """Returns the next element to extend the offspring with.
    Raises NoNextElementException if there is no next element to extend with.
    """
    lst = adj_table[current_element]
    if len(lst) == 0:
        raise NoNextElementException
    # TODO can be sped up with bin. search?
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


def remove_references(adj_table: dict[int, list[tuple[int, bool]]], value: int):
    """Removes all references of value in the lists of adj_table."""
    for x, lst in adj_table.items():
        if x == value:
            continue  # We can skip this case because value cannot be adjacent to itself.
        # TODO can be sped up with bin. search?
        for y, is_common in lst:
            if value == y:
                lst.remove((y, is_common))
                break


def create_adj_table(candidate1: Candidate, candidate2: Candidate) -> dict[int, list[tuple[int, bool]]]:
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


def get_adj(x: int, candidate: Candidate) -> list[int]:
    """Returns the adjacent values of x in candidate as a list."""
    x_idx = index_of(candidate, x)
    prev_idx = x_idx - 1
    next_idx = x_idx + 1 if x_idx < candidate.size - 1 else 0
    return [int(candidate[prev_idx]), int(candidate[next_idx])]


def recombine_PMX(parent1: Candidate, parent2: Candidate) -> list[Candidate]:
    """Use two parent candidates to produce one offspring using partially mapped crossover."""
    size = len(parent1)
    all_offspring = []
    first_pos = rd.randint(0, size - 2)
    second_pos = rd.randint(first_pos, size - 1)
    for p1, p2 in [(parent1, parent2), (parent2, parent1)]:
        offspring = np.zeros_like(p1)
        # We must initialize offspring with -1's, to identify whether a spot is not yet filled.
        for i in range(size):
            offspring[i] = -1
        offspring[first_pos:second_pos + 1] = p1[first_pos:second_pos + 1]
        for elem in p2[first_pos:second_pos + 1]:
            if elem in p1[first_pos:second_pos + 1]:
                continue  # elem already occurs in offspring
            # elem is not yet in offspring, find the index to place it
            index = 0
            value = elem
            while value != -1:
                index = index_of(p2, value)
                value = offspring[index]
            offspring[index] = elem
        for i in range(size):
            if offspring[i] == -1:
                offspring[i] = p2[i]
        all_offspring.append(Candidate(offspring))
    return all_offspring


def recombine_order_crossover(parent1: Candidate, parent2: Candidate) -> list[Candidate]:
    """Use two parent candidates to produce one offspring using order crossover."""
    raise NotImplementedError


def index_of(candidate: Candidate, value: int) -> int:
    """Return the first index at which value occurs in candidate.
    This is just a convenience function for numpy arrays, which behaves like list.index(value).
    This also works straight on Candidate objects.
    """
    return int(np.where(candidate.array == value)[0][0])


def init_monte_carlo(distance_matrix: np.ndarray, population_size: int) -> [Candidate]:
    """Initializes the population at random."""
    population = []
    sample = list(range(len(distance_matrix)))
    for i in range(population_size):
        array = np.array(sample)
        np.random.shuffle(array)
        population.append(array)
    return population


def init_avoid_inf_heuristic(distance_matrix: np.ndarray, population_size: int) -> list[Candidate]:
    """Initializes the population using a heuristic which avoids infinite values."""
    population = []
    for _ in range(population_size):
        choices = list(range(len(distance_matrix)))
        candidate = []
        while len(choices) != 0:
            if len(candidate) == 0:  # The first element is picked at random.
                choice = rd.choice(choices)
                choices.remove(choice)
                candidate.append(choice)
                continue
            possible_next = [x for x in choices if distance_matrix[candidate[-1]][x] != math.inf]
            if len(possible_next) == 0:
                # Pick the first choice because all next choices lead to inf anyway.
                next_element = choices[0]
            else:
                # Pick the best choice with a small probability, random otherwise.
                if rd.random() < 0.05:
                    next_element = min(possible_next, key=lambda x: distance_matrix[candidate[-1]][x])
                else:
                    next_element = rd.choice(possible_next)
            candidate.append(next_element)
            choices.remove(next_element)
        population.append(Candidate(np.array(candidate)))
    return population


def select_k_tournament(population: list[Candidate], k: int) -> Candidate:
    """Performs a k-tournament on the population. Returns the best candidate among k random samples."""
    selected = []
    for i in range(k):
        selected.append(rd.choice(population))
    return min(selected, key=lambda x: x.fitness)


def select_top_k(population: list[Candidate], k: int) -> Candidate:
    """Performs top-k selection on the population. Returns a random candidate among the k best candidates.
    Assumes that population is already sorted from best to worst; this is the case
    when using (lambda+mu) elimination.
    """
    return rd.choice(population[:k])


# Modify the class name to match your student number.
def recalculate_fitness(population: list[Candidate], fitness_func, distance_matrix):
    for candidate in population:
        candidate.fitness = fitness_func(candidate, distance_matrix)


class r0758170:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        distance_matrix = np.loadtxt(file, delimiter=",")
        file.close()

        # Get parameters
        p = Parameters()

        # Initialization
        population = p.init_func(distance_matrix, p.pop_size)
        recalculate_fitness(population, p.fitness_func, distance_matrix)

        current_it = 1
        best_solution = population[0]
        best_objective = best_solution.fitness
        while True:
            # Selection
            # Perform a certain number of k-tournaments; this depends on self.mu
            # and whether the recombination operator returns one or two offspring.
            # One offspring: need 2 * self.mu selected.
            # Two offspring: need self.mu selected.
            selected = []
            for i in range(2 * p.nr_offspring):
                selected.append(p.select_func(population, p.k))

            # Variation
            # Recombination will produce new offspring using the selected candidates.
            new_offspring = []
            it = iter(selected)
            for p1 in it:
                p2 = next(it)
                offspring = p.recombine_func(p1, p2)
                new_offspring.extend(offspring)
            population.extend(new_offspring)

            # Mutation will happend on the entire population and new offspring, with a certain probability.
            for candidate in population:
                if rd.random() < p.mutate_chance:
                    p.mutate_func(candidate)

            recalculate_fitness(population, p.fitness_func, distance_matrix)

            # Elimination
            # Lambda + mu elimination: keep only the best candidates.
            population.sort(key=lambda x: x.fitness)
            population = population[:p.pop_size]

            # Recalculate mean and best.
            mean_objective = 0.0
            current_best_solution = population[0]
            current_best_objective = current_best_solution.fitness
            for candidate in population:
                mean_objective += candidate.fitness
                if candidate.fitness < current_best_objective:
                    current_best_objective = candidate.fitness
                    current_best_solution = candidate
            mean_objective = mean_objective / p.pop_size
            if current_best_objective < best_objective:
                best_objective = current_best_objective
                best_solution = current_best_solution

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            print(f'{current_it:6} | mean: {mean_objective:10.2f} | best:{best_objective:10.2f}')
            timeLeft = self.reporter.report(mean_objective, best_objective, best_solution)
            if timeLeft < 0:
                break
            current_it += 1

        # Your code here.
        return 0
