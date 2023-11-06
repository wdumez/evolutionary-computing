from __future__ import annotations
import random as rd

import math
import numpy as np
import Reporter
import copy
from numpy.typing import NDArray


class Candidate:

    @staticmethod
    def sort(candidates: list[Candidate], reverse=False) -> None:
        """Sort a list of candidates in-place according to their fitness."""
        candidates.sort(key=lambda x: x.fitness, reverse=reverse)

    def __init__(self, array: NDArray[int]):
        self.array = array
        self.fitness = 0.0
        self.mutate_prob = 0.05
        self.recombine_prob = 1.0
        self.local_search_prob = 0.0
        self.mutate_func = mutate_inversion
        self.recombine_func = recombine_order_crossover
        self.local_search_func = local_search_insert
        self.fitness_func = path_length
        self.distance_func = distance

    def __eq__(self, other):
        return np.array_equal(self.array, other.array)

    def __repr__(self) -> str:
        return str(self.array)

    def __len__(self) -> int:
        return len(self.array)

    def __iter__(self):
        return iter(self.array)

    def __getitem__(self, item):
        return self.array[item]

    def __setitem__(self, key, value):
        self.array[key] = value

    @property
    def size(self):
        return self.array.size

    def shuffle(self):
        """Shuffle self."""
        np.random.shuffle(self.array)

    def index(self, value: int) -> int:
        """Return the first index where value appears."""
        return int(np.where(self.array == value)[0][0])

    def mutate(self, *args) -> None:
        """Mutate in-place with a probability of self.mutate_prob."""
        if rd.random() < self.mutate_prob:
            self.mutate_func(self, *args)

    def recombine(self, *args) -> None:
        """Recombine with another parent to produce offspring, with a probability of self.recombine_prob.
        Otherwise, the offspring will be copies of the parents.
        The offspring must be provided because they will be changed in-place.
        """
        if rd.random() < self.recombine_prob:
            self.recombine_func(self, *args)
        else:
            recombine_copy(self, *args)

    def local_search(self, distance_matrix: NDArray[float]) -> None:
        """Perform a local search with a probability of self.local_search_prob."""
        if rd.random() < self.local_search_prob:
            self.local_search_func(self, distance_matrix)

    def recalculate_fitness(self, distance_matrix: NDArray[float]) -> None:
        """Recalculate the fitness."""
        self.fitness = self.fitness_func(self, distance_matrix)

    def distance(self, other_candidate: Candidate) -> float:
        """Return the distance to another candidate."""
        return self.distance_func(self, other_candidate)


def mutate_inversion(candidate: Candidate, first_pos: int | None = None, second_pos: int | None = None) -> None:
    """Mutate in-place using inversion mutation."""
    first_pos = rd.randrange(0, candidate.size - 1) if first_pos is None else first_pos
    second_pos = rd.randrange(first_pos + 1, candidate.size) if second_pos is None else second_pos
    candidate[first_pos:second_pos + 1] = np.flip(candidate[first_pos:second_pos + 1])


def mutate_swap(candidate: Candidate, first_pos: int | None = None, second_pos: int | None = None) -> None:
    """Mutate in-place using swap mutation."""
    first_pos = rd.randrange(0, candidate.size) if first_pos is None else first_pos
    second_pos = first_pos if second_pos is None else second_pos
    while second_pos == first_pos:
        second_pos = rd.randrange(0, candidate.size)
    tmp = candidate[first_pos]
    candidate[first_pos] = candidate[second_pos]
    candidate[second_pos] = tmp


def mutate_scramble(candidate: Candidate, first_pos: int | None = None, second_pos: int | None = None) -> None:
    """Mutate in-place using scramble mutation."""
    first_pos = rd.randrange(0, candidate.size - 1) if first_pos is None else first_pos
    second_pos = rd.randrange(first_pos + 1, candidate.size) if second_pos is None else second_pos
    np.random.shuffle(candidate[first_pos:second_pos + 1])


def mutate_insert(candidate: Candidate, first_pos: int | None = None, second_pos: int | None = None) -> None:
    """Mutate in-place using insert mutation."""
    first_pos = rd.randrange(0, candidate.size - 1) if first_pos is None else first_pos
    second_pos = rd.randrange(first_pos + 1, candidate.size) if second_pos is None else second_pos
    tmp = candidate[second_pos]
    candidate[first_pos + 2:second_pos + 1] = candidate[first_pos + 1:second_pos]
    candidate[first_pos + 1] = tmp


def path_length(candidate: Candidate, distance_matrix: NDArray[float]) -> float:
    """Return the length of the path."""
    result = 0.0
    for i in range(candidate.size - 1):
        # Order is important for the distance matrix.
        result += distance_matrix[candidate[i]][candidate[i + 1]]
    result += distance_matrix[candidate[candidate.size - 1]][candidate[0]]
    return result


def distance(candidate1: Candidate, candidate2: Candidate) -> float:
    """Return the distance between two candidates.
    Uses Hamming distance; so the distance is the nr. of elements that are different.
    """
    # We must first align the candidates so that they start with the same element.
    c1_aligned = copy.deepcopy(candidate1.array)
    offset = np.where(candidate1.array == candidate2.array[0])[0][0]
    c1_aligned = np.roll(c1_aligned, shift=-offset)
    dist = 0
    for i in range(len(candidate1)):
        if c1_aligned[i] != candidate2[i]:
            dist += 1
    return float(dist)


def recombine_copy(parent1: Candidate, parent2: Candidate,
                   offspring1: Candidate, offspring2: Candidate, *args) -> None:
    """Dummy recombine function which copies the parents into the offspring."""
    offspring1[:] = parent1[:]
    offspring2[:] = parent2[:]


def recombine_cycle_crossover(parent1: Candidate, parent2: Candidate,
                              offspring1: Candidate, offspring2: Candidate, *args) -> None:
    """Use two parent candidates to produce two offspring using cycle crossover."""
    cycles = find_cycles(parent1, parent2)
    for i, cycle in enumerate(cycles):
        if i % 2 == 0:
            for idx in cycle:
                offspring1[idx] = parent1[idx]
                offspring2[idx] = parent2[idx]
        else:
            for idx in cycle:
                offspring1[idx] = parent2[idx]
                offspring2[idx] = parent1[idx]


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
            value_p2 = parent2[current_idx]
            current_idx = parent1.index(value_p2)
            if current_idx == start_idx:
                break
            unused_idx.remove(current_idx)
            cycle.append(current_idx)
        cycles.append(cycle)
    return cycles


class NoNextElementException(Exception):
    """Exception used in edge crossover recombination."""


# TODO Performance is terrible, try changing adj. list to matrix?
#      Also, does not yet work!
def recombine_edge_crossover(parent1: Candidate, parent2: Candidate,
                             offspring1: Candidate, offspring2: Candidate, *args) -> None:
    """Use two parent candidates to produce two offspring using edge crossover.
    Since edge crossover only creates one offspring per recombination, the second
    offspring is the same."""
    adj_table = create_adj_table(parent1, parent2)
    remaining = list(range(len(parent1)))
    current_element = rd.choice(remaining)
    offspring1[0] = current_element
    idx_off = 1
    remaining.remove(current_element)
    remove_references(adj_table, current_element)
    while len(remaining) != 0:
        try:
            current_element = pick_next_element(adj_table, current_element)
            offspring1[idx_off] = current_element
            idx_off += 1
            remaining.remove(current_element)
            remove_references(adj_table, current_element)
        except NoNextElementException:
            try:
                next_element = pick_next_element(adj_table, offspring1[0])
                offspring1 = np.roll(offspring1, shift=1)
                offspring1[0] = next_element
                idx_off += 1
                remaining.remove(next_element)
                remove_references(adj_table, next_element)
            except NoNextElementException:
                current_element = rd.choice(remaining)
                offspring1[idx_off] = current_element
                idx_off += 1
                remaining.remove(current_element)
                remove_references(adj_table, current_element)
    offspring2[:] = offspring1[:]


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


def get_adj(x: int, candidate: Candidate) -> list[int]:
    """Returns the adjacent values of x in candidate as a list."""
    x_idx = candidate.index(x)
    prev_idx = x_idx - 1
    next_idx = x_idx + 1 if x_idx < candidate.size - 1 else 0
    return [int(candidate[prev_idx]), int(candidate[next_idx])]


def recombine_PMX(parent1: Candidate, parent2: Candidate,
                  offspring1: Candidate, offspring2: Candidate,
                  first_pos: int | None = None, second_pos: int | None = None) -> None:
    """Use two parent candidates to produce two offspring using partially mapped crossover."""
    first_pos = rd.randrange(0, parent1.size - 1) if first_pos is None else first_pos
    second_pos = rd.randrange(first_pos, parent1.size) if second_pos is None else second_pos
    for off, (p1, p2) in zip([offspring1, offspring2], [(parent1, parent2), (parent2, parent1)]):
        # We must initialize offspring with -1's, to identify whether a spot is not yet filled.
        off.array.fill(-1)

        off[first_pos:second_pos + 1] = p1[first_pos:second_pos + 1]
        for elem in p2[first_pos:second_pos + 1]:
            if elem in p1[first_pos:second_pos + 1]:
                continue  # elem already occurs in offspring
            # elem is not yet in offspring, find the index to place it
            idx = 0
            value = elem
            while value != -1:
                idx = p2.index(value)
                value = off[idx]
            off[idx] = elem
        for i in range(parent1.size):
            if off[i] == -1:
                off[i] = p2[i]


def recombine_order_crossover(parent1: Candidate, parent2: Candidate,
                              offspring1: Candidate, offspring2: Candidate,
                              first_pos: int | None = None, second_pos: int | None = None) -> None:
    """Use two parent candidates to produce two offspring using order crossover."""
    first_pos = rd.randrange(0, parent1.size - 1) if first_pos is None else first_pos
    second_pos = rd.randrange(first_pos, parent1.size) if second_pos is None else second_pos
    for off, (p1, p2) in zip([offspring1, offspring2], [(parent1, parent2), (parent2, parent1)]):
        off[first_pos:second_pos + 1] = p1[first_pos:second_pos + 1]
        idx_p2 = second_pos + 1 if second_pos < parent1.size - 1 else 0
        idx_off = idx_p2
        while idx_off != first_pos:
            if p2[idx_p2] not in off[first_pos:second_pos + 1]:
                off[idx_off] = p2[idx_p2]
                idx_off = 0 if idx_off + 1 >= parent1.size else idx_off + 1
            idx_p2 = 0 if idx_p2 + 1 >= parent1.size else idx_p2 + 1


def init_monte_carlo(size: int, distance_matrix: NDArray[float]) -> list[Candidate]:
    """Initializes the population at random."""
    sample = Candidate(np.array(list(range(len(distance_matrix))), dtype=int))
    population = []
    for i in range(size):
        sample.shuffle()
        population.append(copy.deepcopy(sample))
    return population


def init_avoid_inf_heuristic(size: int, distance_matrix: NDArray[float]) -> list[Candidate]:
    """Initializes the population using a heuristic which tries to avoid infinite values."""
    population = []
    for i in range(size):
        choices = list(range(len(distance_matrix)))
        rd.shuffle(choices)
        idx = 0
        candidate = Candidate(np.zeros(len(distance_matrix), dtype=int))
        while len(choices) != 0:
            # The first element is picked at random.
            if len(choices) == len(distance_matrix):
                choice = choices[0]
                choices.remove(choice)
                candidate[idx] = choice
                idx += 1
                continue
            # Extend with the first element which does not lead to inf.
            next_element = None
            for x in choices:
                if distance_matrix[candidate[idx - 1]][x] != math.inf:
                    next_element = x
                    break
            if next_element is None:
                next_element = choices[0]
            candidate[idx] = next_element
            idx += 1
            choices.remove(next_element)
        population.append(candidate)
    return population


def select_k_tournament(population: list[Candidate], k: int) -> Candidate:
    """Performs a k-tournament on the population. Returns the best candidate among k random samples."""
    selected = rd.choice(population)
    for _ in range(k - 1):
        maybe = rd.choice(population)
        if maybe.fitness < selected.fitness:
            selected = maybe
    return selected


def select_top_k(population: list[Candidate], k: int) -> Candidate:
    """Performs top-k selection on the population. Returns a random candidate among the k best candidates.
    """
    Candidate.sort(population)
    return rd.choice(population[:k])


def elim_lambda_plus_mu(population: list[Candidate],
                        offspring: list[Candidate]) -> tuple[list[Candidate], list[Candidate]]:
    """Performs (lambda+mu)-elimination. Returns the new population and offspring."""
    lamda = len(population)
    population.extend(offspring)
    Candidate.sort(population)
    return population[:lamda], population[lamda:]


def elim_lambda_plus_mu_crowding():
    """Performs (lambda+mu)-elimination with crowding for diversity promotion."""
    raise NotImplementedError


def elim_lambda_comma_mu(population: list[Candidate],
                         offspring: list[Candidate]) -> tuple[list[Candidate], list[Candidate]]:
    """Performs (lambda,mu)-elimination. Returns the new population and offspring."""
    lamda = len(population)
    Candidate.sort(offspring)
    offspring.extend(population)
    return offspring[:lamda], offspring[lamda:]


def elim_lambda_comma_mu_crowding():
    """Performs (lambda,mu)-elimination with crowding for diversity promotion."""
    raise NotImplementedError


def elim_age_based(population: list[Candidate],
                   offspring: list[Candidate]) -> tuple[list[Candidate], list[Candidate]]:
    """Performs age-based elimination. Returns the new population and offspring.
    Requires the population and offspring to be the same length.
    """
    return offspring, population


def elim_k_tournament(population: list[Candidate], offspring: list[Candidate], k: int) -> list[Candidate]:
    """Performs k-tournament elimination. Returns the new population."""
    both = []
    both.extend(population)
    both.extend(offspring)
    new_population = []
    for i in range(len(population)):
        new_population.append(select_k_tournament(both, k))
    return new_population


def local_search_insert(candidate: Candidate, distance_matrix: NDArray[float]) -> None:
    """Performs a local search using one insertion.
    Candidate is updated in-place if a better candidate was found.
    """
    tmp = copy.deepcopy(candidate)
    new_candidate = copy.deepcopy(candidate)
    best_fit = candidate.fitness
    for i in range(1, len(candidate)):
        x = tmp[i]
        tmp[1: i + 1] = tmp[:i]
        tmp[0] = x
        tmp.recalculate_fitness(distance_matrix)
        if tmp.fitness < best_fit:
            best_fit = tmp.fitness
            new_candidate[:] = tmp[:]
        tmp[:] = candidate[:]
    if best_fit < candidate.fitness:
        candidate[:] = new_candidate[:]


def is_valid_tour(candidate: Candidate) -> bool:
    """Returns True if the candidate represents a valid tour, False otherwise.
    A tour is valid if every city appears in it exactly once. Note that it does
    not matter whether the length of the tour is infinite in this test.
    """
    present = np.zeros(len(candidate), dtype=bool)
    for x in candidate:
        present[x] = True
    return np.all(present)


def assert_valid_tour(candidate: Candidate):
    assert is_valid_tour(candidate), f'Got invalid tour: {candidate}'


def assert_valid_tours(candidates: list[Candidate]):
    for x in candidates:
        assert_valid_tour(x)


class r0758170:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        distance_matrix = np.loadtxt(file, delimiter=",")
        file.close()

        # Parameters
        # TODO These should eventually be moved into the Candidate class,
        #      so they can be used for self-adaptivity.
        k = 5
        lamda = 100
        mu = 100

        # Initialization
        population = init_avoid_inf_heuristic(lamda, distance_matrix)
        offspring = init_monte_carlo(mu, distance_matrix)  # This is just to fill up the list.
        for x in population:
            x.recalculate_fitness(distance_matrix)

        assert_valid_tours(population)
        assert_valid_tours(offspring)

        current_it = 1
        best_solution = population[0]
        best_objective = best_solution.fitness
        while True:
            # Selection and recombination
            for i in range(0, len(offspring), 2):
                p1 = select_k_tournament(population, k)
                p2 = select_k_tournament(population, k)
                p1.recombine(p2, offspring[i], offspring[i + 1])

            assert_valid_tours(population)
            assert_valid_tours(offspring)

            # Local search & Mutation
            for x in population:
                x.local_search(distance_matrix)
                x.mutate()
                x.recalculate_fitness(distance_matrix)
            for x in offspring:
                x.local_search(distance_matrix)
                x.mutate()
                x.recalculate_fitness(distance_matrix)

            assert_valid_tours(population)
            assert_valid_tours(offspring)

            # Elimination
            population, offspring = elim_age_based(population, offspring)

            assert_valid_tours(population)
            assert_valid_tours(offspring)

            for x in population:
                x.recalculate_fitness(distance_matrix)

            # Recalculate mean and best
            mean_objective = 0.0
            current_best_solution = population[0]
            current_best_objective = current_best_solution.fitness
            for x in population:
                mean_objective += x.fitness
                if x.fitness < current_best_objective:
                    current_best_objective = x.fitness
                    current_best_solution = x
            mean_objective = mean_objective / len(population)
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
