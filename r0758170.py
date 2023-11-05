import random as rd

import math
import numpy as np

import Reporter


class Parameters:
    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix
        self.k_selection = 3
        self.k_crowding = 5
        self.pop_size = 50
        self.offspring_size = 100
        self.mutate_chance = 0.05
        self.recombine_chance = 0.95
        self.lso_chance = 0.0
        self.recombine_func = recombine_PMX
        self.mutate_func = mutate_inversion
        self.lso_func = lso_insert
        self.init_func = init_avoid_inf_heuristic
        self.select_func = select_k_tournament
        self.elim_func = elim_lambda_plus_mu
        self.fitness_func = path_length
        self.distance_func = distance
        if self.recombine_func.__name__ in [recombine_edge_crossover.__name__]:
            self.nr_off_per_recombine = 1
        else:
            self.nr_off_per_recombine = 2


def mutate_inversion(candidate) -> None:
    """Mutate in-place using inversion mutation."""
    first_pos = rd.randrange(0, candidate.size - 1)
    second_pos = rd.randrange(first_pos + 1, candidate.size)
    candidate[first_pos:second_pos + 1] = np.flip(candidate[first_pos:second_pos + 1])


def mutate_swap(candidate) -> None:
    """Mutate in-place using swap mutation."""
    first_pos = rd.randrange(0, candidate.size)
    second_pos = first_pos
    while second_pos == first_pos:
        second_pos = rd.randrange(0, candidate.size)
    tmp = candidate[first_pos]
    candidate[first_pos] = candidate[second_pos]
    candidate[second_pos] = tmp


def mutate_scramble(candidate) -> None:
    """Mutate in-place using scramble mutation."""
    first_pos = rd.randrange(0, candidate.size - 1)
    second_pos = rd.randrange(first_pos + 1, candidate.size)
    np.random.shuffle(candidate[first_pos:second_pos + 1])


def mutate_insert(candidate) -> None:
    """Mutate in-place using insert mutation."""
    first_pos = rd.randrange(0, candidate.size - 1)
    second_pos = rd.randrange(first_pos + 1, candidate.size)
    tmp = candidate[second_pos]
    candidate[first_pos + 2:second_pos + 1] = candidate[first_pos + 1:second_pos]
    candidate[first_pos + 1] = tmp


def path_length(candidate, p: Parameters) -> float:
    """Return the length of the path."""
    result = 0.0
    for i in range(candidate.size - 1):
        # Order is important for the distance matrix.
        result += p.distance_matrix[candidate[i]][candidate[i + 1]]
    result += p.distance_matrix[candidate[candidate.size - 1]][candidate[0]]
    return result


def distance(candidate1, candidate2) -> int:
    """Return the distance between two candidates.
    Uses Hamming distance; so the distance is the nr. of elements that are different.
    """
    # We must first align the candidates so that they start with the same element.
    c1_aligned = np.copy(candidate1)
    offset = np.where(candidate1 == candidate2[0])[0][0]
    c1_aligned = np.roll(c1_aligned, shift=-offset)
    dist = 0
    for i in range(len(candidate1)):
        if c1_aligned[i] != candidate2[i]:
            dist += 1
    return dist


def recombine_cycle_crossover(parent1, parent2, *offspring) -> None:
    """Use two parent candidates to produce two offspring using cycle crossover."""
    cycles = find_cycles(parent1, parent2)
    offspring1, offspring2 = offspring
    for i, cycle in enumerate(cycles):
        if i % 2 == 0:
            for idx in cycle:
                offspring1[idx] = parent1[idx]
                offspring2[idx] = parent2[idx]
        else:
            for idx in cycle:
                offspring1[idx] = parent2[idx]
                offspring2[idx] = parent1[idx]


def find_cycles(parent1, parent2) -> list[list[int]]:
    """Returns all cycles of the parents using indices."""
    unused_idx = list(range(len(parent1)))
    cycles = []
    while len(unused_idx) != 0:
        start_idx = unused_idx[0]
        current_idx: int = start_idx
        unused_idx.remove(current_idx)
        cycle = [current_idx]
        while True:
            value_p2 = parent2[current_idx]
            current_idx = index(parent1, value_p2)
            if current_idx == start_idx:
                break
            unused_idx.remove(current_idx)
            cycle.append(current_idx)
        cycles.append(cycle)
    return cycles


class NoNextElementException(Exception):
    """Exception used in edge crossover recombination."""


def recombine_edge_crossover(parent1, parent2, offspring) -> None:
    """Use two parent candidates to produce one offspring using edge crossover."""
    # TODO Creates invalid offspring sometimes...
    adj_table = create_adj_table(parent1, parent2)
    remaining = list(range(len(parent1)))
    current_element = rd.choice(remaining)
    offspring[0] = current_element
    idx_off = 1
    remaining.remove(current_element)
    remove_references(adj_table, current_element)
    while len(remaining) != 0:
        try:
            current_element = pick_next_element(adj_table, current_element)
            offspring[idx_off] = current_element
            idx_off += 1
            remaining.remove(current_element)
            remove_references(adj_table, current_element)
        except NoNextElementException:
            try:
                next_element = pick_next_element(adj_table, offspring[0])
                offspring = np.roll(offspring, shift=1)
                offspring[0] = next_element
                idx_off += 1
                remaining.remove(next_element)
                remove_references(adj_table, next_element)
            except NoNextElementException:
                current_element = rd.choice(remaining)
                offspring[idx_off] = current_element
                idx_off += 1
                remaining.remove(current_element)
                remove_references(adj_table, current_element)


def pick_next_element(adj_table: dict[int, list[tuple[int, bool]]], current_element) -> int:
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


def create_adj_table(candidate1, candidate2) -> dict[int, list[tuple[int, bool]]]:
    """Create an adjacency table for candidate1 and candidate2."""
    # TODO replace table with array and use indexing instead of by key
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


def get_adj(x: int, candidate) -> list[int]:
    """Returns the adjacent values of x in candidate as a list."""
    x_idx = index(candidate, x)
    prev_idx = x_idx - 1
    next_idx = x_idx + 1 if x_idx < candidate.size - 1 else 0
    return [int(candidate[prev_idx]), int(candidate[next_idx])]


def recombine_PMX(parent1, parent2, *offspring) -> None:
    """Use two parent candidates to produce two offspring using partially mapped crossover."""
    size = parent1.size
    offspring1, offspring2 = offspring
    first_pos = rd.randrange(0, size - 1)
    second_pos = rd.randrange(first_pos, size)
    for off, (p1, p2) in zip([offspring1, offspring2], [(parent1, parent2), (parent2, parent1)]):
        # We must initialize offspring with -1's, to identify whether a spot is not yet filled.
        off.fill(-1)

        off[first_pos:second_pos + 1] = p1[first_pos:second_pos + 1]
        for elem in p2[first_pos:second_pos + 1]:
            if elem in p1[first_pos:second_pos + 1]:
                continue  # elem already occurs in offspring
            # elem is not yet in offspring, find the index to place it
            idx = 0
            value = elem
            while value != -1:
                idx = index(p2, value)
                value = off[idx]
            off[idx] = elem
        for i in range(size):
            if off[i] == -1:
                off[i] = p2[i]


def recombine_order_crossover(parent1, parent2, *offspring) -> None:
    """Use two parent candidates to produce two offspring using order crossover."""
    size = parent1.size
    offspring1, offspring2 = offspring
    first_pos = rd.randrange(0, size - 1)
    second_pos = rd.randrange(first_pos, size)
    for off, (p1, p2) in zip([offspring1, offspring2], [(parent1, parent2), (parent2, parent1)]):
        off[first_pos:second_pos + 1] = p1[first_pos:second_pos + 1]
        idx_p2 = second_pos + 1 if second_pos < size - 1 else 0
        idx_off = idx_p2
        while idx_off != first_pos:
            if p2[idx_p2] not in off[first_pos:second_pos + 1]:
                off[idx_off] = p2[idx_p2]
                idx_off = 0 if idx_off + 1 >= size else idx_off + 1
            idx_p2 = 0 if idx_p2 + 1 >= size else idx_p2 + 1


def index(array, value: int) -> int:
    """Return the first index at which value occurs in array.
    This is just a convenience function for numpy arrays, which behaves like list.index(value).
    """
    return int(np.where(array == value)[0][0])


def init_monte_carlo(population, p: Parameters) -> None:
    """Initializes the population at random."""
    sample = np.array(list(range(len(p.distance_matrix))), dtype=int)
    for i in range(p.pop_size):
        population[i][:] = sample[:]
        np.random.shuffle(population[i])


def init_avoid_inf_heuristic(population, p: Parameters) -> None:
    """Initializes the population using a heuristic which tries to avoid infinite values."""
    for i in range(p.pop_size):
        choices = list(range(len(p.distance_matrix)))
        rd.shuffle(choices)
        idx = 0
        while len(choices) != 0:
            # The first element is picked at random.
            if len(population[i]) == 0:
                choice = choices[0]
                choices.remove(choice)
                population[i][idx] = choice
                idx += 1
                continue
            # Extend with the first element which does not lead to inf.
            next_element = None
            for x in choices:
                if p.distance_matrix[population[i][idx - 1]][x] != math.inf:
                    next_element = x
                    break
            if next_element is None:
                next_element = choices[0]
            population[i][idx] = next_element
            idx += 1
            choices.remove(next_element)


def select_k_tournament(population, population_fit, p: Parameters):
    """Performs a k-tournament on the population. Returns the best candidate among k random samples."""
    best_fit = math.inf
    selected = None
    for _ in range(p.k_selection):
        idx = rd.randrange(0, len(population))
        if selected is None or population_fit[idx] < best_fit:
            best_fit = population_fit[idx]
            selected = population[idx]
    return selected


def select_top_k(population, population_fit, p: Parameters):
    """Performs top-k selection on the population. Returns a random candidate among the k best candidates.
    Assumes that population is already sorted from best to worst; this is the case
    when using (lambda+mu) elimination.
    """
    raise NotImplementedError


def elim_lambda_plus_mu(population, population_fit, offspring, offspring_fit, p: Parameters):
    """Performs (lambda+mu)-elimination."""
    both = np.concatenate((population, offspring))
    both_fit = np.concatenate((population_fit, offspring_fit))
    sorting_idx = np.argsort(both_fit)
    both = np.reshape(both[sorting_idx], both.shape)
    both_fit = np.reshape(both_fit[sorting_idx], both_fit.shape)
    population[:] = both[:p.pop_size]


def elim_lambda_plus_mu_crowding(population, population_fit, offspring, offspring_fit, p: Parameters):
    """Performs (lambda+mu)-elimination with crowding for diversity promotion."""
    both = np.concatenate((population, offspring))
    both_fit = np.concatenate((population_fit, offspring_fit))
    sorting_idx = np.argsort(both_fit)
    both = np.reshape(both[sorting_idx], both.shape)
    both_fit = np.reshape(both_fit[sorting_idx], both_fit.shape)
    idx_next = 0
    can_choose = np.ones(len(both), dtype=bool)
    for i in range(len(population)):
        # Pick the best next candidate for promotion.
        population[i] = both[idx_next]
        can_choose[idx_next] = False

        # Select a candidate to eliminate (crowding).
        idx_selected = None
        smallest_dist = math.inf
        for _ in range(p.k_crowding):
            j = idx_next
            while can_choose[j] is False:
                j = rd.randrange(0, len(both))
            dist = p.distance_func(population[i], both[j])
            if idx_selected is None or dist < smallest_dist:
                smallest_dist = dist
                idx_selected = j
        can_choose[idx_selected] = False

        # The next choice for the population must be choose-able.
        while can_choose[idx_next] is False:
            idx_next += 1


def elim_lambda_comma_mu(population, population_fit, offspring, offspring_fit, p: Parameters):
    """Performs (lambda,mu)-elimination."""
    sorting_idx = np.argsort(offspring_fit)
    offspring = np.reshape(offspring[sorting_idx], offspring.shape)
    population[:] = offspring[:p.pop_size]


def elim_lambda_comma_mu_crowding(population, population_fit, offspring, offspring_fit, p: Parameters):
    """Performs (lambda,mu)-elimination with crowding for diversity promotion."""
    sorting_idx = np.argsort(offspring_fit)
    offspring = np.reshape(offspring[sorting_idx], offspring.shape)
    idx_next = 0
    can_choose = np.ones(len(offspring), dtype=bool)
    for i in range(len(population)):
        # Pick the best next candidate for promotion.
        population[i] = offspring[idx_next]
        can_choose[idx_next] = False

        # Select a candidate to eliminate (crowding).
        idx_selected = None
        smallest_dist = math.inf
        for _ in range(p.k_crowding):
            j = idx_next
            while can_choose[j] is False:
                j = rd.randrange(0, len(offspring))
            dist = p.distance_func(population[i], offspring[j])
            if idx_selected is None or dist < smallest_dist:
                smallest_dist = dist
                idx_selected = j
        can_choose[idx_selected] = False

        # The next choice for the population must be choose-able.
        while can_choose[idx_next] is False:
            idx_next += 1


def elim_age_based(population, population_fit, offspring, offspring_fit, p: Parameters):
    """Performs age-based elimination. Requires pop_size == off_size."""
    population[:] = offspring[:]


def elim_k_tournament(population, population_fit, offspring, offspring_fit, p: Parameters):
    """Performs k-tournament elimination."""
    new_population = np.copy(population)
    both = np.concatenate((population, offspring))
    both_fit = np.concatenate((population_fit, offspring_fit))
    for i in range(p.pop_size):
        new_population[i] = select_k_tournament(both, both_fit, p)
    population[:] = new_population[:]


def lso_insert(candidate, candidate_fit, p: Parameters):
    """Performs a local search using one insertion.
    Candidate is updated in-place if a better candidate was found.
    """
    tmp = np.copy(candidate)
    new_candidate = np.copy(candidate)
    best_fit = candidate_fit
    for i in range(1, len(candidate)):
        x = tmp[i]
        tmp[1: i + 1] = tmp[:i]
        tmp[0] = x
        new_fit = p.fitness_func(tmp, p)
        if new_fit < best_fit:
            best_fit = new_fit
            new_candidate[:] = tmp[:]
        tmp[:] = candidate[:]
    if best_fit < candidate_fit:
        candidate[:] = new_candidate[:]


def recalculate_fitness(population, population_fit, p: Parameters):
    for i, candidate in enumerate(population):
        population_fit[i] = p.fitness_func(candidate, p)


def is_valid_tour(candidate) -> bool:
    """Returns True if the candidate represents a valid tour, False otherwise.
    A tour is valid if every city appears in it exactly once. Note that it does
    not matter whether the length of the tour is infinite in this test.
    """
    present = np.zeros(len(candidate), dtype=bool)
    for x in candidate:
        present[x] = True
    return np.all(present)


class r0758170:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        distance_matrix = np.loadtxt(file, delimiter=",")
        file.close()

        p = Parameters(distance_matrix)
        population = np.zeros(shape=(p.pop_size, len(distance_matrix)), dtype=int)
        offspring = np.zeros(shape=(p.offspring_size, len(distance_matrix)), dtype=int)
        population_fit = np.zeros(p.pop_size, dtype=float)
        offspring_fit = np.zeros(p.offspring_size, dtype=float)

        # Initialization
        p.init_func(population, p)
        for x in population:
            assert is_valid_tour(x)
        recalculate_fitness(population, population_fit, p)

        current_it = 1
        best_solution = population[0]
        best_objective = population_fit[0]
        while True:
            # Selection and recombination
            i = 0
            while i < p.offspring_size:
                p1 = p.select_func(population, population_fit, p)
                p2 = p.select_func(population, population_fit, p)
                if rd.random() < p.recombine_chance:
                    p.recombine_func(p1, p2, offspring[i], offspring[i + 1])
                else:
                    offspring[i][:] = p1[:]
                    offspring[i + 1][:] = p2[:]
                i += 2

            for x in population:
                assert is_valid_tour(x)
            for x in offspring:
                assert is_valid_tour(x)

            # Local search & Mutation
            for i in range(p.pop_size):
                if rd.random() < p.lso_chance:
                    p.lso_func(population[i], population_fit[i], p)
                if rd.random() < p.mutate_chance:
                    p.mutate_func(population[i])
            for i in range(p.offspring_size):
                if rd.random() < p.lso_chance:
                    p.lso_func(offspring[i], offspring_fit[i], p)
                if rd.random() < p.mutate_chance:
                    p.mutate_func(offspring[i])

            for x in population:
                assert is_valid_tour(x)
            for x in offspring:
                assert is_valid_tour(x)

            recalculate_fitness(population, population_fit, p)
            recalculate_fitness(offspring, offspring_fit, p)

            # Elimination
            p.elim_func(population, population_fit, offspring, offspring_fit, p)

            recalculate_fitness(population, population_fit, p)

            for x in population:
                assert is_valid_tour(x)

            # Recalculate mean and best
            mean_objective = 0.0
            current_best_solution = population[0]
            current_best_objective = population_fit[0]
            for i in range(p.pop_size):
                mean_objective += population_fit[i]
                if population_fit[i] < current_best_objective:
                    current_best_objective = population_fit[i]
                    current_best_solution = population[i]
            mean_objective = mean_objective / p.pop_size
            if current_best_objective < best_objective:
                best_objective = current_best_objective
                best_solution = current_best_solution

            assert is_valid_tour(best_solution)

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            print(f'{current_it:6} | mean: {mean_objective:10.2f} | best: {best_objective:10.2f}')
            timeLeft = self.reporter.report(mean_objective, best_objective, best_solution)
            if timeLeft < 0:
                break
            current_it += 1

        return 0
