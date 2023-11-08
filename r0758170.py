from __future__ import annotations
import random as rd
import itertools
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

    @staticmethod
    def stats(candidates: list[Candidate]) -> tuple[float, Candidate]:
        """Return the mean fitness and best candidate from a list of candidates."""
        assert candidates != [], 'Cannot get stats of an empty list.'
        mean = 0.0
        best = candidates[0]
        for x in candidates:
            mean += x.fitness
            if x.fitness < best.fitness:
                best = x
        mean = mean / len(candidates)
        return mean, best

    def __init__(self, array: NDArray[int]):
        self.array = array
        self.fitness = 0.0
        self.nr_mutations = 5
        self.mutate_func = mutate_inversion
        self.recombine_func = recombine_edge_crossover
        self.local_search_func = local_search_insert
        self.fitness_func = path_length
        self.distance_func = distance_hamming

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

    def mutate(self) -> None:
        """Mutate in-place self.nr_mutations times."""
        for _ in range(self.nr_mutations):
            self.mutate_func(self)

    def recombine(self, other_parent) -> list[Candidate]:
        """Recombine with another parent to produce offspring."""
        return self.recombine_func(self, other_parent)

    def local_search(self, distance_matrix: NDArray[float]) -> None:
        """Perform a local search."""
        self.local_search_func(self, distance_matrix)

    def recalculate_fitness(self, distance_matrix: NDArray[float]) -> None:
        """Recalculate the fitness."""
        self.fitness = self.fitness_func(self, distance_matrix)

    def distance(self, other_candidate: Candidate) -> float:
        """Return the distance to another candidate."""
        return self.distance_func(self, other_candidate)


def mutate_inversion(candidate: Candidate) -> None:
    """Mutate in-place using inversion mutation."""
    first_pos = rd.randrange(0, candidate.size - 1)
    second_pos = rd.randrange(first_pos + 1, candidate.size)
    candidate[first_pos:second_pos + 1] = np.flip(candidate[first_pos:second_pos + 1])


def mutate_swap(candidate: Candidate) -> None:
    """Mutate in-place using swap mutation."""
    first_pos = rd.randrange(0, candidate.size)
    second_pos = first_pos
    while second_pos == first_pos:
        second_pos = rd.randrange(0, candidate.size)
    tmp = candidate[first_pos]
    candidate[first_pos] = candidate[second_pos]
    candidate[second_pos] = tmp


def mutate_scramble(candidate: Candidate) -> None:
    """Mutate in-place using scramble mutation."""
    first_pos = rd.randrange(0, candidate.size - 1)
    second_pos = rd.randrange(first_pos + 1, candidate.size)
    np.random.shuffle(candidate[first_pos:second_pos + 1])


def mutate_insert(candidate: Candidate) -> None:
    """Mutate in-place using insert mutation."""
    first_pos = rd.randrange(0, candidate.size - 1)
    second_pos = rd.randrange(first_pos + 1, candidate.size)
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


def distance_hamming(candidate1: Candidate, candidate2: Candidate) -> float:
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


def recombine_copy(parent1: Candidate, parent2: Candidate) -> list[Candidate]:
    """Dummy recombine function which copies the parents into the offspring."""
    return [copy.deepcopy(parent1), copy.deepcopy(parent2)]


def recombine_cycle_crossover(parent1: Candidate, parent2: Candidate) -> list[Candidate]:
    """Use two parent candidates to produce two offspring using cycle crossover."""
    offspring1 = copy.deepcopy(parent1)
    offspring2 = copy.deepcopy(parent2)
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
    return [offspring1, offspring2]


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
def recombine_edge_crossover(parent1: Candidate, parent2: Candidate) -> list[Candidate]:
    """Use two parent candidates to produce two offspring using edge crossover.
    Since edge crossover only creates one offspring per recombination, the second
    offspring is the same."""
    offspring = copy.deepcopy(parent1)
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
                offspring.array = np.roll(offspring.array, shift=1)
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
    return [offspring, copy.deepcopy(offspring)]


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


def recombine_PMX(parent1: Candidate, parent2: Candidate) -> list[Candidate]:
    """Use two parent candidates to produce two offspring using partially mapped crossover."""
    first_pos = rd.randrange(0, parent1.size - 1)
    second_pos = rd.randrange(first_pos, parent1.size)
    offspring = []
    for p1, p2 in [(parent1, parent2), (parent2, parent1)]:
        # We must initialize offspring with -1's, to identify whether a spot is not yet filled.
        off = copy.deepcopy(parent1)
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
        offspring.append(off)
    assert_valid_tours(offspring)
    return offspring


def recombine_order_crossover(parent1: Candidate, parent2: Candidate) -> list[Candidate]:
    """Use two parent candidates to produce two offspring using order crossover."""
    first_pos = rd.randrange(0, parent1.size - 1)
    second_pos = rd.randrange(first_pos, parent1.size)
    offspring = []
    for p1, p2 in [(parent1, parent2), (parent2, parent1)]:
        off = copy.deepcopy(parent1)
        off[first_pos:second_pos + 1] = p1[first_pos:second_pos + 1]
        idx_p2 = second_pos + 1 if second_pos < parent1.size - 1 else 0
        idx_off = idx_p2
        while idx_off != first_pos:
            if p2[idx_p2] not in off[first_pos:second_pos + 1]:
                off[idx_off] = p2[idx_p2]
                idx_off = 0 if idx_off + 1 >= parent1.size else idx_off + 1
            idx_p2 = 0 if idx_p2 + 1 >= parent1.size else idx_p2 + 1
        offspring.append(off)
    assert_valid_tours(offspring)
    return offspring


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
                        offspring: list[Candidate]) -> list[Candidate]:
    """Performs (lambda+mu)-elimination. Returns the new population."""
    lamda = len(population)
    population.extend(offspring)
    Candidate.sort(population)
    return population[:lamda]


# TODO After a random nr. of iterations, this leads to invalid tours. Some values are missing, others are copied.
#      Just like last time, the cause seems to be that (somehow) the returned population and offspring can have
#      the same candidate in it, so during recombination >= 1 parent and offspring are actually the same object.
#      Maybe try rewriting so it's more clear which objects go where?
def elim_lambda_plus_mu_crowding(population: list[Candidate],
                                 offspring: list[Candidate],
                                 crowding_factor: int) -> list[Candidate]:
    """Performs (lambda+mu)-elimination with a crowding strategy for diversity promotion.
    This crowding scheme is similar but not the same as the one shown in the lecture.
    If the crowding_factor is 0, then this is the same as regular (lambda+mu)-elimination.
    """
    lamda = len(population)
    mu = len(offspring)
    assert (lamda + mu) % 2 == 0, f'The sum of lamda and mu must be even, got: {lamda} + {mu}.'
    both = [x for x in itertools.chain(population, offspring)]
    Candidate.sort(both)
    new_population = []
    removed = []
    while len(new_population) < lamda and len(removed) < mu:
        choice = both[0]
        most_similar = both[-1]  # If crowding_factor is 0, then remove worst as usual.
        closest_dist = math.inf
        for _ in range(crowding_factor):
            sample = choice
            while sample is choice:
                sample = rd.choice(both)
            dist = choice.distance(sample)
            if dist < closest_dist:
                closest_dist = dist
                most_similar = sample
        assert choice is not most_similar, "choice is most_similar, this can't be!"
        new_population.append(choice)
        removed.append(most_similar)
        both.remove(choice)
        both.remove(most_similar)
    if len(new_population) < lamda:
        new_population.extend(both)
    return new_population


def elim_lambda_comma_mu(population: list[Candidate],
                         offspring: list[Candidate]) -> list[Candidate]:
    """Performs (lambda,mu)-elimination. Returns the new population."""
    lamda = len(population)
    mu = len(offspring)
    assert lamda <= mu, \
        f'(lambda,mu)-elimination requires lambda <= mu, got: {lamda} > {mu}'
    Candidate.sort(offspring)
    return offspring[:lamda]


def elim_lambda_comma_mu_crowding(population: list[Candidate],
                                  offspring: list[Candidate],
                                  crowding_factor: int) -> list[Candidate]:
    """Performs (lambda,mu)-elimination with crowding for diversity promotion."""
    lamda = len(population)
    mu = len(offspring)
    assert 2 * lamda <= mu, \
        f'(lambda,mu)-elimination with crowding requires 2*lambda <= mu, got: {2 * lamda} > {mu}'
    Candidate.sort(offspring)
    new_population = []
    for _ in range(lamda):
        choice = offspring[0]
        most_similar = offspring[-1]
        best_dist = math.inf
        for _ in range(crowding_factor):
            sample = choice
            while sample is choice:
                sample = rd.choice(offspring)
            dist = choice.distance(sample)
            if dist < best_dist:
                best_dist = dist
                most_similar = sample
        new_population.append(choice)
        offspring.remove(choice)
        offspring.remove(most_similar)
    return new_population


def elim_age_based(population: list[Candidate],
                   offspring: list[Candidate]) -> list[Candidate]:
    """Performs age-based elimination. Returns the new population and offspring.
    Requires the population and offspring to be the same length.
    This is really just a specific case of (lamda,mu)-elimination.
    """
    assert len(population) == len(offspring), \
        f'Age based elimination requires lambda == mu, got: {len(population)} != {len(offspring)}'
    return elim_lambda_comma_mu(population, offspring)


def elim_k_tournament():
    """Performs k-tournament elimination. Returns the new population."""
    raise NotImplementedError


def local_search_insert(candidate: Candidate, distance_matrix: NDArray[float]) -> None:
    """Performs a 1-opt local search using one insertion.
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
        tmp[:i] = tmp[1:i + 1]
        tmp[i] = x
    if best_fit < candidate.fitness:
        candidate[:] = new_candidate[:]


def local_search_inversion(candidate: Candidate, distance_matrix: NDArray[float], inv_length: int) -> None:
    """Performs a 1-opt local search using one inversion, of inv_length elements.
    Candidate is updated in-place if a better candidate was found.
    """
    assert 1 < inv_length < len(candidate), f'Got bad inversion length: {inv_length}'
    tmp = copy.deepcopy(candidate)
    new_candidate = copy.deepcopy(candidate)
    best_fit = candidate.fitness
    for i in range(candidate.size - inv_length + 1):
        tmp[i:i + inv_length] = np.flip(tmp.array[i:i + inv_length])
        tmp.recalculate_fitness(distance_matrix)
        if tmp.fitness < best_fit:
            best_fit = tmp.fitness
            new_candidate[:] = tmp[:]
        tmp[i:i + inv_length] = np.flip(tmp.array[i:i + inv_length])
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
        k = 2
        lamda = 50
        mu = 100
        mutation_prob = 0.05

        # Initialization
        population = init_avoid_inf_heuristic(lamda, distance_matrix)
        for x in population:
            x.recalculate_fitness(distance_matrix)

        assert_valid_tours(population)

        current_it = 1
        best_solution = copy.deepcopy(population[0])
        while True:
            offspring = []

            # Selection and recombination
            for i in range(0, mu, 2):
                p1 = select_k_tournament(population, k)
                p2 = select_k_tournament(population, k)
                offspring.extend(p1.recombine(p2))

            assert len(offspring) == mu, f'Nr. offspring ({len(offspring)}) does not match mu ({mu})'

            assert_valid_tours(population)
            assert_valid_tours(offspring)

            # Local search & Mutation
            for x in itertools.chain(population, offspring):
                if rd.random() < mutation_prob:
                    x.mutate()
                    x.local_search(distance_matrix)
                    x.recalculate_fitness(distance_matrix)

            assert_valid_tours(population)
            assert_valid_tours(offspring)

            # Elimination
            population = elim_lambda_comma_mu_crowding(population, offspring, 5)

            assert_valid_tours(population)

            for x in population:
                x.recalculate_fitness(distance_matrix)

            # Recalculate mean and best
            mean_objective, current_best_solution = Candidate.stats(population)
            if current_best_solution.fitness < best_solution.fitness:
                best_solution = copy.deepcopy(current_best_solution)

            assert is_valid_tour(best_solution)

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            print(f'{current_it:6} | mean: {mean_objective:10.2f} | best: {best_solution.fitness:10.2f}')
            timeLeft = self.reporter.report(mean_objective, best_solution.fitness, best_solution)
            if timeLeft < 0:
                break
            current_it += 1

        return 0
