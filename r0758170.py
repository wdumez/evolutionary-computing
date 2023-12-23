# r0758170.py
#
# For the course: Genetic algorithms and evolutionary computing
#
# William Dumez (r0758170)

from __future__ import annotations
import functools
import copy
import random as rd
import sys
import math
import numpy as np
from numpy.typing import NDArray

import Reporter


class Candidate:

    @staticmethod
    def distance_stats(candidates: list[Candidate]) -> tuple[float, float, float]:
        """Return the min, avg and max distance of one candidate to all other candidates."""
        counter = 0
        avg_dist = 0.0
        min_dist = math.inf
        max_dist = 0.0
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                dist = candidates[i].distance(candidates[j])
                avg_dist += dist
                min_dist = min(min_dist, dist)
                max_dist = max(max_dist, dist)
                counter += 1
        avg_dist = avg_dist / counter
        return min_dist, avg_dist, max_dist

    @staticmethod
    def sort(candidates: list[Candidate], reverse=False) -> None:
        """Sort a list of candidates in-place according to their fitness."""
        candidates.sort(key=lambda x: x.fitness, reverse=reverse)

    @staticmethod
    def most_fit(candidates: list[Candidate]) -> Candidate:
        """Return the most fit candidate."""
        return min(candidates, key=lambda x: x.fitness)

    @staticmethod
    def stats(candidates: list[Candidate], include_inf=True) -> tuple[float, Candidate]:
        """Return the mean fitness and best candidate from a list of candidates.
        If include_inf is False, then infinite values are ignored.
        """
        assert candidates != [], 'Cannot get stats of an empty list.'
        mean = 0.0
        best = candidates[0]
        nr_included = 0
        for x in candidates:
            if not include_inf and x.fitness == math.inf:
                continue
            mean += x.fitness
            if x.fitness < best.fitness:
                best = x
            nr_included += 1
        if nr_included == 0:
            mean = 0.0
        else:
            mean = mean / nr_included
        return mean, best

    def __init__(self, tour: list[int]):
        self.tour = tour
        self.fitness = 0.0
        self.original_fitness = self.fitness
        self.mutate_options = [swap, insert, inversion, scramble]
        self.recombine_func = recombine_order_crossover
        self.lso_options = [swap, insert]
        self.fitness_func = path_length
        self.distance_func = distance_edges_cached

    def __repr__(self) -> str:
        return str(self.tour)

    def __len__(self) -> int:
        return len(self.tour)

    def __iter__(self):
        return iter(self.tour)

    def __getitem__(self, item):
        return self.tour[item]

    def __setitem__(self, key, value):
        self.tour[key] = value

    def mutate(self) -> None:
        """Mutate self in-place."""
        move_func = rd.choice(self.mutate_options)
        self.tour = mutate(self.tour, move_func)

    def recombine(self, other: Candidate) -> list[Candidate]:
        """Recombine with another parent to produce offspring."""
        offspring_tours = [x for x in self.recombine_func(self.tour, other.tour)]
        offspring = []
        for tour in offspring_tours:
            offspring.append(Candidate(tour))
        return offspring

    def local_search(self, distance_matrix: NDArray[float], fast=True) -> None:
        """Perform a local search and update the candidate in-place.
        If fast is True, then the search is not exhaustive for performance reasons.
        """
        move_func = rd.choice(self.lso_options)
        local_search(self, distance_matrix, 1, move_func, fast)

    def recalculate_fitness(self, distance_matrix: NDArray[float]) -> None:
        """Recalculate the fitness."""
        self.fitness = self.fitness_func(self.tour, distance_matrix)

    def distance(self, to: Candidate) -> float:
        """Return the distance to another candidate."""
        return self.distance_func(self.tour, to.tour)

    def closest_to(self, others: list[Candidate]) -> Candidate:
        """Returns the closest from others using the distance metric from self."""
        return min(others, key=lambda x: self.distance(x))

    def sigma_neighborhood(self, others: list[Candidate], sigma: float) -> list[Candidate]:
        """Returns all candidates from others that are within sigma-distance from self."""
        # Important: by demanding that "x is not self", we still count
        # other candidates which have the same tour as self (but are different objects).
        return [x for x in others if self.distance(x) <= sigma and x is not self]


def align(tour: list[int], first_pos: int, second_pos: int) -> tuple[list[int], int]:
    """Align a tour so that it starts with the index of first_pos,
    where [first_pos:second_pos) is a range of indices in tour.
    Also returns the new second_pos.
    The new first_pos is 0, and so is not returned.
    Examples:
        input: tour = [0,1,2,3,4,5,6,7,8,9], first_pos = 3, second_pos = 8
        output: ([3,4,5,6,7,8,9,0,1,2], 5)

        input: tour = [0,1,2,3,4,5,6,7,8,9], first_pos = 6, second_pos = 3
        output: ([6,7,8,9,0,1,2,3,4,5], 7)

        input: tour = [0,1,2,3,4,5,6,7,8,9], first_pos = 6, second_pos = 4
        output: ([6,7,8,9,0,1,2,3,4,5], 8)
    """
    tour = tour[first_pos:] + tour[:first_pos]
    if first_pos < second_pos:
        second_pos = second_pos - first_pos
    else:
        second_pos = len(tour) - first_pos + second_pos
    return tour, second_pos


def swap(tour: list[int], a: int, b: int) -> list[int]:
    """Swap the values of indices a and b in tour, and return the new tour."""
    tmp = tour[a]
    tour[a] = tour[b]
    tour[b] = tmp
    return tour


def insert(tour: list[int], a: int, b: int) -> list[int]:
    """Insert the value of index (b-1) before the value of index a in tour, and return the new tour.
    Example:
        insert([0,1,2,3,4], 1, 4) returns [3,1,2,4,0]
    """
    tour, b = align(tour, a, b)
    value_b = tour[b - 1]
    tour = [value_b] + tour[:b - 1] + tour[b:]
    return tour


def inversion(tour: list[int], a: int, b: int) -> list[int]:
    """Invert the range [a:b) of tour, and return the new tour."""
    tour, b = align(tour, a, b)
    tour[:b] = np.flip(tour[:b])
    return tour


def scramble(tour: list[int], a: int, b: int) -> list[int]:
    """Scramble the range [a:b) of tour, and return the new tour."""
    tour, b = align(tour, a, b)
    tmp = tour[:b]
    rd.shuffle(tmp)
    tour[:b] = tmp
    return tour


def mutate(tour: list[int], move_func) -> list[int]:
    """Mutate a tour with a movement function (swap, inversion, insert or scramble)."""
    a, b = rd.sample(range(len(tour)), 2)
    return move_func(tour, a, b)


def path_length(tour: list[int], distance_matrix: NDArray[float]) -> float:
    """Return the length of a tour."""
    result = 0.0
    for i in range(-1, len(tour) - 1):
        # Order is important for the distance matrix.
        result += distance_matrix[tour[i]][tour[i + 1]]
    return result


def distance_edges(tour1: list[int], tour2: list[int]) -> float:
    """Return the distance between two tours.
    The distance is the nr. of edges that are different.
    """
    edges1 = get_edges(tour1)
    edges2 = get_edges(tour2)
    return float(len(edges1 - edges2))


def get_edges(tour: list[int]) -> set[tuple[int, int]]:
    """Return all edges of the tour in a set."""
    edges = []
    for i in range(-1, len(tour) - 1):
        edges.append((tour[i], tour[i + 1]))
    return set(edges)


@functools.lru_cache(maxsize=1000)
def get_edges_cached(tour: tuple[int]) -> set[tuple[int, int]]:
    """Memoized version of get_edges. The tour is now a tuple because it must be immutable. """
    return get_edges(list(tour))


def distance_edges_cached(tour1: list[int], tour2: list[int]) -> float:
    """Memoized version of distance_edges."""

    @functools.lru_cache(maxsize=1000)
    def _distance_edges_cached(tour1_: tuple[int], tour2_: tuple[int]) -> float:
        edges1 = get_edges_cached(tour1_)
        edges2 = get_edges_cached(tour2_)
        return float(len(edges1 - edges2))

    return _distance_edges_cached(tuple(tour1), tuple(tour2))


def recombine_cycle_crossover(parent1: list[int], parent2: list[int]) -> list[list[int]]:
    """Use two parent tours to produce two offspring tours using cycle crossover."""
    offspring1 = copy.deepcopy(parent1)
    offspring2 = copy.deepcopy(parent2)
    cycles = get_cycles(parent1, parent2)
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


def get_cycles(tour1: list[int], tour2: list[int]) -> list[list[int]]:
    """Returns all cycles of the two tours using indices."""
    unused_idx = list(range(len(tour1)))
    cycles = []
    while len(unused_idx) != 0:
        start_idx = unused_idx[0]
        current_idx: int = start_idx
        unused_idx.remove(current_idx)
        cycle = [current_idx]
        while True:
            value_p2 = tour2[current_idx]
            current_idx = tour1.index(value_p2)
            if current_idx == start_idx:
                break
            unused_idx.remove(current_idx)
            cycle.append(current_idx)
        cycles.append(cycle)
    return cycles


class NoNextElementException(Exception):
    """Exception used in edge crossover recombination."""


def recombine_edge_crossover(parent1: list[int], parent2: list[int]) -> list[list[int]]:
    """Use two parent tours to produce two offspring tours using edge crossover.
    Since edge crossover only creates one offspring per recombination, the second
    offspring is the same.
    """
    edge_table = create_edge_table(parent1, parent2)
    remaining = list(range(len(parent1)))
    current_element = rd.choice(remaining)
    offspring = [current_element]
    remaining.remove(current_element)
    remove_references(edge_table, current_element)
    while len(remaining) != 0:
        try:
            current_element = pick_next_element(edge_table, current_element)
            offspring.append(current_element)
            remaining.remove(current_element)
            remove_references(edge_table, current_element)
        except NoNextElementException:
            try:
                next_element = pick_next_element(edge_table, offspring[0])
                offspring.insert(0, next_element)
                remaining.remove(next_element)
                remove_references(edge_table, next_element)
            except NoNextElementException:
                current_element = rd.choice(remaining)
                offspring.append(current_element)
                remaining.remove(current_element)
                remove_references(edge_table, current_element)
    return [offspring, copy.deepcopy(offspring)]


def pick_next_element(edge_table: dict[int, list[tuple[int, bool]]], current_element: int) -> int:
    """Returns the next element to extend the offspring with.
    Raises NoNextElementException if there is no next element to extend with.
    """
    lst = edge_table[current_element]
    if len(lst) == 0:
        raise NoNextElementException
    for x, is_common in lst:
        if is_common:
            return x
    next_element_options = []
    shortest_len = math.inf
    for x, is_common in lst:
        x_lst_len = len(edge_table[x])
        if x_lst_len < shortest_len:
            next_element_options = [x]
            shortest_len = x_lst_len
        elif x_lst_len == shortest_len:
            next_element_options.append(x)
    next_element = rd.choice(next_element_options)
    return next_element


def remove_references(edge_table: dict[int, list[tuple[int, bool]]], value: int):
    """Removes all references of value in the lists of edge_table."""
    for x, lst in edge_table.items():
        if x == value:
            continue  # We can skip this case because value cannot be adjacent to itself.
        for y, is_common in lst:
            if value == y:
                lst.remove((y, is_common))
                break


def create_edge_table(tour1: list[int], tour2: list[int]) -> dict[int, list[tuple[int, bool]]]:
    """Create an edge table for two tours."""
    edge_table = {x: [] for x in tour1}
    for x in edge_table:
        adj_in_parent1 = get_adj(x, tour1)
        adj_in_parent2 = get_adj(x, tour2)
        for y in adj_in_parent1:
            if y in adj_in_parent2:
                edge_table[x].append((y, True))
                adj_in_parent2.remove(y)
            else:
                edge_table[x].append((y, False))
        for y in adj_in_parent2:
            edge_table[x].append((y, False))
    return edge_table


def get_adj(x: int, tour: list[int]) -> list[int]:
    """Returns the adjacent values of x in the tour as a list."""
    x_idx = tour.index(x)
    prev_idx = x_idx - 1
    next_idx = x_idx + 1 if x_idx < len(tour) - 1 else 0
    return [int(tour[prev_idx]), int(tour[next_idx])]


def recombine_PMX(parent1: list[int], parent2: list[int]) -> list[list[int]]:
    """Use two parent tours to produce two offspring tours using partially mapped crossover."""
    first_pos, second_pos = rd.sample(range(len(parent1)), 2)
    parent1, _ = align(parent1, first_pos, second_pos)
    parent2, second_pos = align(parent2, first_pos, second_pos)
    offspring = []
    for p1, p2 in [(parent1, parent2), (parent2, parent1)]:
        # We must initialize offspring with -1's, to identify whether a spot is not yet filled.
        off = copy.deepcopy(parent1)
        for i in range(len(off)):
            off[i] = -1

        off[:second_pos] = p1[:second_pos]
        for elem in p2[:second_pos]:
            if elem in p1[:second_pos]:
                continue  # elem already occurs in offspring.
            # elem is not yet in offspring, so find the index to place it.
            idx = 0
            value = elem
            while value != -1:
                idx = p2.index(value)
                value = off[idx]
            off[idx] = elem
        for i in range(len(parent1)):
            if off[i] == -1:
                off[i] = p2[i]
        offspring.append(off)
    return offspring


def recombine_order_crossover(parent1: list[int], parent2: list[int]) -> list[list[int]]:
    """Use two parent tours to produce two offspring tours using order crossover."""
    first_pos, second_pos = rd.sample(range(len(parent1)), 2)
    parent1, _ = align(parent1, first_pos, second_pos)
    parent2, second_pos = align(parent2, first_pos, second_pos)
    offspring = []
    for p1, p2 in [(parent1, parent2), (parent2, parent1)]:
        off = copy.deepcopy(parent1)
        off[:second_pos] = p1[:second_pos]
        # Now copy the remaining values of p2 into off, starting from second_pos.
        idx_p2 = second_pos
        idx_off = idx_p2
        while idx_off != 0:
            if p2[idx_p2] not in off[:second_pos]:
                off[idx_off] = p2[idx_p2]
                idx_off = 0 if idx_off + 1 >= len(parent1) else idx_off + 1
            idx_p2 = 0 if idx_p2 + 1 >= len(parent1) else idx_p2 + 1
        offspring.append(off)
    return offspring


def local_search(candidate: Candidate,
                 distance_matrix: NDArray[float],
                 depth: int,
                 move_func,
                 fast=True) -> None:
    """Perform a local search on a candidate. It gets updated in-place if a better fitness was found.
    The search is not exhaustive and only considers log(len(distance_matrix)) candidates, for performance reasons.
    Move_func determines the neighborhood structure (accepts: swap, insert, scramble, inversion).
    """
    best_candidate = copy.deepcopy(candidate)
    current_candidate = copy.deepcopy(candidate)
    local_search_recursive(best_candidate, current_candidate, distance_matrix, depth, move_func, fast)
    if best_candidate.fitness < candidate.fitness:
        candidate.tour = copy.deepcopy(best_candidate.tour)
        candidate.fitness = best_candidate.fitness


def local_search_recursive(best_candidate: Candidate,
                           current_candidate: Candidate,
                           distance_matrix: NDArray[float],
                           depth: int,
                           move_func,
                           fast):
    """Recursive function used in local_search."""
    if depth == 0:
        return
    for neighbor in lso_neighborhood(current_candidate, move_func, fast):
        neighbor.recalculate_fitness(distance_matrix)
        if neighbor.fitness < best_candidate.fitness:
            best_candidate.fitness = neighbor.fitness
            best_candidate.tour = copy.deepcopy(neighbor.tour)
        local_search_recursive(best_candidate, neighbor, distance_matrix, depth - 1, move_func, fast)


def lso_neighborhood(candidate: Candidate, move_func, fast=True):
    """Generator which generates all candidates within
    1-distance of candidate, according to a movement function.
    The neighbor's fitness has not been calculated yet.
    If fast is True, then only consider log(len(candidate) random neighbors.
    """
    neighbor = copy.deepcopy(candidate)
    size = len(candidate)
    options = list(range(size))
    a_options = copy.deepcopy(options)
    b_options = copy.deepcopy(options)
    if fast:
        rd.shuffle(a_options)
        rd.shuffle(b_options)
        nr = math.ceil(math.log(size))
        a_options = a_options[:nr]
        b_options = b_options[:nr]
    for a in a_options:
        for b in b_options:
            if a == b:
                continue
            neighbor.tour = move_func(candidate.tour, a, b)
            yield copy.deepcopy(neighbor)


def init_monte_carlo(size: int, distance_matrix: NDArray[float]) -> list[Candidate]:
    """Initializes the population at random.
    The resulting candidates may include edges that are missing in the distance matrix.
    """
    sample = list(range(len(distance_matrix)))
    population = []
    for i in range(size):
        rd.shuffle(sample)
        candidate = Candidate(copy.deepcopy(sample))
        candidate.recalculate_fitness(distance_matrix)
        population.append(candidate)
    return population


def init_heuristic(size: int, distance_matrix: NDArray[float],
                   fast: bool = True, greediness: float = 0.5,
                   duplicates=False) -> list[Candidate]:
    """Initializes the population with a heuristic.
    All resulting candidates are guaranteed to not use missing edges.
    greediness is the probability of taking the greed choice each step.
    If duplicates is False, then no duplicate tours are allowed.
    """
    population = []
    while len(population) != size:
        candidate = Candidate(heuristic_solution(distance_matrix, fast, greediness))
        if not duplicates and any([is_same_tour(candidate.tour, y.tour) for y in population]):
            continue  # Duplicate so go again.
        candidate.recalculate_fitness(distance_matrix)
        population.append(candidate)
    return population


def heuristic_solution(distance_matrix: NDArray[float], fast: bool = True, greediness: float = 0.5) -> list[int]:
    """Uses a greedy heuristic to find a solution.
    If fast is True, then only one random starting position is tried; otherwise all starting positions
    in a random permutation are tried.
    greediness is a value between 0 and 1 which indicates the probability of taking the greedy next
    step instead of a random next step. In the case of a random step, it is still guaranteed that
    the total fitness is not infinite.
    """
    # We need to temporarily up the recursion limit because the nr. of recursions is just
    # slightly higher than the problem size (e.g. tour200 recurses 200-210 times).
    # The default limit is 1000, so for tour1000 this *just* becomes a problem.
    # Therefore, we increase the recursion limit in this function only.
    recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(10 * len(distance_matrix))

    choices = list(range(len(distance_matrix)))
    rd.shuffle(choices)  # Important: start with a random permutation.
    results = []
    for start in choices:
        starting_choices = copy.deepcopy(choices)
        starting_choices.remove(start)
        result = heuristic_recursive(starting_choices, [start], distance_matrix, greediness)
        if result is not False:
            results.append(result)
            if fast:
                break
    results.sort(key=lambda x: path_length(x, distance_matrix))
    candidate = results[0]

    sys.setrecursionlimit(recursion_limit)
    return candidate


def heuristic_recursive(choices: list[int], current_result: list[int],
                        distance_matrix: NDArray[float],
                        greediness: float = 0.5) -> list[int] | bool:
    """Recursive function used in heuristic_solution."""
    if len(choices) == 0:
        if distance_matrix[current_result[-1]][current_result[0]] == math.inf:
            # The last choice cannot connect to the first choice, so backtrack.
            return False
        # All edges are valid, so the answer has been found.
        return current_result

    if rd.random() < greediness:
        # This choice is greedy, so sort the given choices in increasing greediness.
        choices.sort(key=lambda x: distance_matrix[current_result[-1]][x])
    else:
        # This choice is not greedy, so randomize the choices.
        rd.shuffle(choices)

    for choice in choices:
        if distance_matrix[current_result[-1]][choice] == math.inf:
            # Not a valid choice because the distance is infinite.
            continue
        new_choices = copy.deepcopy(choices)
        new_choices.remove(choice)
        current_result.append(choice)
        answer = heuristic_recursive(new_choices, current_result, distance_matrix, greediness)
        if answer is not False:
            # An answer was found, so propagate it back up.
            return answer
        # The last choice did not lead to a solution, so backtrack.
        current_result.pop()
    # None of the choices could lead to any solutions.
    return False


def select_k_tournament(population: list[Candidate],
                        k: int,
                        nr_times: int = 1) -> list[Candidate]:
    """Performs nr_times k-tournaments on the population.
    Each tournament is the best candidate among k random samples without replacement.
    """
    selected = []
    while len(selected) != nr_times:
        tournament = rd.sample(population, k)
        most_fit = Candidate.most_fit(tournament)
        selected.append(most_fit)
    return selected


def elim_lambda_plus_mu(population: list[Candidate],
                        offspring: list[Candidate]) -> list[Candidate]:
    """Performs (lambda+mu)-elimination. Returns the new population."""
    lamda = len(population)
    both = population + offspring
    Candidate.sort(both)
    return both[:lamda]


def elim_lambda_plus_mu_fitness_sharing(population: list[Candidate],
                                        offspring: list[Candidate],
                                        alpha: float,
                                        sigma: float) -> list[Candidate]:
    """Performs (lambda+mu)-elimination with fitness sharing for diversity promotion.
    The sign(f(x)) is always 1 for the Traveling Salesman Problem,
    so the implementation does not explicitly calculate this.
    This implementation has been optimized to update incrementally.
    The fitness values of candidates are only changed temporarily.
    The elitism best candidates do not get penalized.
    """
    lamda = len(population)
    old_population = population + offspring
    Candidate.sort(old_population)
    # We must remember the original fitness.
    for x in old_population:
        x.original_fitness = x.fitness
    new_population = []

    while len(new_population) != lamda:
        choice = old_population.pop(0)
        neighbors = []
        for neighbor in choice.sigma_neighborhood(old_population, sigma):
            if neighbor.fitness != math.inf:
                # Only apply a penalty if its fitness isn't already infinite.
                penalty_term = math.pow((1 - (choice.distance(neighbor) / sigma)), alpha)
                neighbor.fitness += neighbor.original_fitness * penalty_term
            neighbors.append(neighbor)
            # Temporarily remove this neighbor from the population because otherwise it is no longer sorted.
            old_population.remove(neighbor)
        # Now insert the removed neighbors back into the population while maintaining its sortedness.
        for neighbor in neighbors:
            old_population = insert_sorted(old_population, neighbor, key=lambda y: y.fitness)
        new_population.append(choice)
    # Now restore the original fitness.
    for x in new_population:
        x.fitness = x.original_fitness
    return new_population


def insert_sorted(lst: list[Candidate], x: Candidate, key) -> list[Candidate]:
    """Insert an element into a sorted list such that the new list is still sorted. Returns the new list.
    key is a function applied on an element to determine its relative order.
    lst must be sorted in increasing order.
    """
    if len(lst) == 0:
        return [x]
    inserted = False
    key_x = key(x)
    for i, y in enumerate(lst):
        if key_x <= key(y):
            lst.insert(i, x)
            inserted = True
            break
    if not inserted:
        lst.append(x)
    return lst


def is_same_tour(tour1: list[int], tour2: list[int]) -> bool:
    """Returns True if tour1 is the same tour as tour2, False otherwise.
    (They could be shifted copies.)
    """
    start = tour2.index(tour1[0])
    shifted2 = tour2[start:] + tour2[:start]
    return tour1 == shifted2


def is_complete_tour(tour: list[int]) -> bool:
    """Returns True if the tour represents a complete tour, False otherwise.
    A tour is complete iff every city appears in it exactly once. Note that it does
    not matter whether the length of the tour is infinite in this test.
    """
    present = np.zeros(len(tour), dtype=bool)
    for x in tour:
        present[x] = True
    return np.all(present)


def assert_complete_tour(tour: list[int]):
    """Assert that tour is complete."""
    assert is_complete_tour(tour), f'Got incomplete tour: {tour}'


def assert_complete_tours(candidates: list[Candidate]):
    """Assert that all candidates have complete tours."""
    for x in candidates:
        assert_complete_tour(x.tour)


class r0758170:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        distance_matrix = np.loadtxt(file, delimiter=",")
        file.close()

        # Parameters
        k = 3
        lamda = 40
        mu = math.ceil(1.5 * lamda)
        greedy_percentage = 0.20
        alpha = 0.5
        sigma = 15
        mutation_prob = 0.05
        lso_prob = 0.01

        # Check that parameters are valid
        assert k > 1, f'k must be greater than 1, got: {k}'
        assert 0.0 <= mutation_prob <= 1.0, f'Mutation_prob should be a probability, got: {mutation_prob}'
        assert 0.0 <= lso_prob <= 1.0, f'Lso_prob should be a probability, got: {lso_prob}'
        assert lamda > 0, f'Lambda must be positive, got: {lamda}'
        assert mu % 2 == 0, f'Mu must be even, got: {mu}'
        assert alpha >= 0, f'Alpha should be non-negative, got: {alpha}'
        assert sigma > 0.0, f'Sigma must be positive, got: {sigma}'

        # Initialization
        nr_very_greedy = math.ceil(greedy_percentage * lamda)
        nr_more_random = lamda - nr_very_greedy
        print(f'Initializing with {nr_very_greedy} very greedy and {nr_more_random} more random.')
        print('This may take a while...')
        very_greedy = init_heuristic(nr_very_greedy, distance_matrix, fast=True, greediness=1.0)
        more_random = init_heuristic(nr_more_random, distance_matrix, fast=True, greediness=0.0)
        population = very_greedy + more_random
        Candidate.sort(population)
        print('Finished initializing.')

        current_it = 1
        max_it = -1  # Set to -1 for no limit.
        while current_it != max_it + 1:

            # Selection
            selected = select_k_tournament(population, k, mu)

            # Recombination
            offspring = []
            it = iter(selected)
            for p1 in it:
                p2 = next(it)
                new_offspring = p1.recombine(p2)
                for x in new_offspring:
                    x.recalculate_fitness(distance_matrix)
                offspring.extend(new_offspring)

            # Mutation
            for x in offspring:
                if rd.random() < mutation_prob:
                    x.mutate()
                    x.recalculate_fitness(distance_matrix)

            # Local search
            for x in offspring:
                if rd.random() < lso_prob:
                    x.local_search(distance_matrix)

            # Elimination
            population = elim_lambda_plus_mu_fitness_sharing(population, offspring, alpha, sigma)

            # Recalculate statistics
            mean_objective, best_solution = Candidate.stats(population, include_inf=False)

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            time_left = self.reporter.report(mean_objective, best_solution.fitness,
                                             np.array(best_solution.tour, dtype=int))

            print(f'{time_left:5.1f} sec | '
                  f'{current_it:6} | '
                  f'mean: {mean_objective:10.2f} | '
                  f'best: {best_solution.fitness:10.2f}'
                  )

            current_it += 1
            if time_left < 0:
                break
        print('Done.')
        return 0
