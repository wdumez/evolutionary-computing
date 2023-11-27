from __future__ import annotations
import functools
import copy
import itertools
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
        self.nr_mutations = 1
        self.nr_local_search = 1
        self.mutate_func = mutate_swap
        self.recombine_func = recombine_edge_crossover
        self.local_search_func = local_search
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
        self.tour = self.mutate_func(self.tour, self.nr_mutations)

    def recombine(self, other: Candidate) -> list[Candidate]:
        """Recombine with another parent to produce offspring."""
        offspring_tours = [x for x in self.recombine_func(self.tour, other.tour)]
        offspring = [copy.deepcopy(self), copy.deepcopy(other)]
        for x, tour in zip(offspring, offspring_tours):
            x.tour = tour
        return offspring

    def local_search(self, distance_matrix: NDArray[float]) -> None:
        """Perform a local search."""
        self.local_search_func(self, distance_matrix, self.nr_local_search)

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
        # other candidates which have the same tour as self.
        return [x for x in others if self.distance(x) <= sigma and x is not self]


def mutate_inversion(tour: list[int], nr_times: int = 1) -> list[int]:
    """Mutate a tour using inversion mutation."""
    for _ in range(nr_times):
        first_pos = rd.randrange(0, len(tour) - 1)
        second_pos = rd.randrange(first_pos + 1, len(tour))
        tour[first_pos:second_pos + 1] = np.flip(tour[first_pos:second_pos + 1])
    return tour


def mutate_swap(tour: list[int], nr_times: int = 1) -> list[int]:
    """Mutate a tour using swap mutation."""
    for _ in range(nr_times):
        first_pos = rd.randrange(0, len(tour))
        second_pos = first_pos
        while second_pos == first_pos:
            second_pos = rd.randrange(0, len(tour))
        tmp = tour[first_pos]
        tour[first_pos] = tour[second_pos]
        tour[second_pos] = tmp
    return tour


def mutate_scramble(tour: list[int], nr_times: int = 1) -> list[int]:
    """Mutate a tour using scramble mutation."""
    for i in range(nr_times):
        first_pos = rd.randrange(0, len(tour) - 1)
        second_pos = rd.randrange(first_pos + 1, len(tour))
        rd.shuffle(tour[first_pos:second_pos + 1])
    return tour


def mutate_insert(tour: list[int], nr_times: int = 1) -> list[int]:
    """Mutate a tour using insert mutation."""
    for i in range(nr_times):
        first_pos = rd.randrange(0, len(tour) - 1)
        second_pos = rd.randrange(first_pos + 1, len(tour))
        tmp = tour[second_pos]
        tour[first_pos + 2:second_pos + 1] = tour[first_pos + 1:second_pos]
        tour[first_pos + 1] = tmp
    return tour


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
    for i in range(len(tour) - 1):
        edges.append((tour[i], tour[i + 1]))
    edges.append((tour[-1], tour[0]))
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


def recombine_copy(parent1: list[int], parent2: list[int]) -> list[list[int]]:
    """Dummy recombine function which copies the parent tours into the offspring."""
    return [copy.deepcopy(parent1), copy.deepcopy(parent2)]


def recombine_cycle_crossover(parent1: list[int], parent2: list[int]) -> list[list[int]]:
    """Use two parent tours to produce two offspring tours using cycle crossover."""
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


def find_cycles(tour1: list[int], tour2: list[int]) -> list[list[int]]:
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
    assert_valid_tour(offspring)
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
    first_pos = rd.randrange(0, len(parent1) - 1)
    second_pos = rd.randrange(first_pos, len(parent1))
    offspring = []
    for p1, p2 in [(parent1, parent2), (parent2, parent1)]:
        # We must initialize offspring with -1's, to identify whether a spot is not yet filled.
        off = copy.deepcopy(parent1)
        for i in range(len(off)):
            off[i] = -1

        off[first_pos:second_pos + 1] = p1[first_pos:second_pos + 1]
        for elem in p2[first_pos:second_pos + 1]:
            if elem in p1[first_pos:second_pos + 1]:
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
    first_pos = rd.randrange(0, len(parent1) - 1)
    second_pos = rd.randrange(first_pos, len(parent1))
    offspring = []
    for p1, p2 in [(parent1, parent2), (parent2, parent1)]:
        off = copy.deepcopy(parent1)
        off[first_pos:second_pos + 1] = p1[first_pos:second_pos + 1]
        # Now copy the remaining values of p2 into off, starting from second_pos.
        idx_p2 = second_pos + 1 if second_pos < len(parent1) - 1 else 0
        idx_off = idx_p2
        while idx_off != first_pos:
            if p2[idx_p2] not in off[first_pos:second_pos + 1]:
                off[idx_off] = p2[idx_p2]
                idx_off = 0 if idx_off + 1 >= len(parent1) else idx_off + 1
            idx_p2 = 0 if idx_p2 + 1 >= len(parent1) else idx_p2 + 1
        offspring.append(off)
    return offspring


def local_search(candidate: Candidate, distance_matrix: NDArray[float], nr_times: int = 2) -> None:
    """Perform a local search on a candidate. It gets updated in-place if a better fitness was found.
    The worst edge (x,y) is replaced by another edge(z,y),
    found by trying to insert y in each other position.
    """
    changed = False
    for i in range(nr_times):
        if i > 0 and not changed:
            break  # The first time could not improve, so repeating does nothing.
        best_fit = candidate.fitness
        tmp = copy.deepcopy(candidate)
        changed = False
        worst_idx = get_worst_element_idx(candidate.tour, distance_matrix)
        worst_element = tmp.tour.pop(worst_idx)

        for new_idx in range(0, len(candidate)):
            if new_idx == worst_idx:
                continue
            # Make change.
            tmp.tour.insert(new_idx, worst_element)
            # Test to see if it's better.
            tmp.recalculate_fitness(distance_matrix)
            if tmp.fitness < best_fit:
                best_fit = tmp.fitness
                candidate.tour = copy.deepcopy(tmp.tour)
                changed = True
            # Undo change.
            tmp.tour.pop(new_idx)
        if changed:
            candidate.recalculate_fitness(distance_matrix)


def get_worst_element_idx(tour: list[int], distance_matrix: NDArray[float]) -> int:
    """Returns the index of element y of the worst edge (x,y) in the tour."""
    worst_fit = 0.0
    worst_idx = None
    for i in range(-1, len(tour) - 1):
        fit = distance_matrix[i][i + 1]
        if fit > worst_fit:
            worst_fit = fit
            worst_idx = i + 1
    assert worst_idx is not None, 'Got None for worst in local search'
    return worst_idx


def init_monte_carlo(size: int, distance_matrix: NDArray[float]) -> list[Candidate]:
    """Initializes the population at random."""
    sample = list(range(len(distance_matrix)))
    population = []
    for i in range(size):
        rd.shuffle(sample)
        population.append(Candidate(copy.deepcopy(sample)))
    return population


def init_heuristic(size: int, distance_matrix: NDArray[float],
                   fast: bool = True, greedy: float = 0.5) -> list[Candidate]:
    """Initializes the population with a heuristic."""
    population = []
    for i in range(size):
        candidate = Candidate(heuristic_solution(distance_matrix, fast, greedy))
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


def select_k_tournament(population: list[Candidate], k: int, nr_times: int = 1) -> list[Candidate]:
    """Performs nr_times k-tournaments on the population. Returns the best candidate among k random samples."""
    assert k <= len(population), f'Cannot perform a k-tournament with k = {k} on a population of size {len(population)}'
    selected = []
    for _ in range(nr_times):
        tournament = rd.sample(population, k)
        selected.append(Candidate.most_fit(tournament))
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
                                        sigma: float,
                                        elitism: int = 0) -> list[Candidate]:
    """Performs (lambda+mu)-elimination with fitness sharing for diversity promotion.
    The sign(f(x)) is always 1 for the Traveling Salesman Problem,
    so the implementation does not explicitly calculate this.
    This implementation has been optimized to update incrementally.
    The fitness values of candidates are only changed temporarily.
    The elitism parameter signifies how many of the best solutions are kept
    without
    """
    lamda = len(population)
    old_population = population + offspring
    Candidate.sort(old_population)
    # We must remember the original fitness.
    for x in old_population:
        x.original_fitness = x.fitness
    new_population = []

    elites = [old_population[i] for i in range(elitism)]

    while len(new_population) != lamda:
        choice = old_population.pop(0)
        neighbors = []
        for neighbor in choice.sigma_neighborhood(old_population, sigma):
            if neighbor not in elites and neighbor.fitness != math.inf:
                # Only apply a penalty if this neighbor is not an elite.
                penalty_term = math.pow((1 - (choice.distance(neighbor) / sigma)), alpha)
                neighbor.fitness += neighbor.original_fitness * penalty_term
            neighbors.append(neighbor)
            # Temporarily remove this neighbor from the population because otherwise it is no longer sorted.
            old_population.remove(neighbor)
        # Now insert the removed neighbors back into the population while maintaining its sortedness.
        for neighbor in neighbors:
            old_population = insert_sorted(old_population, neighbor, key=lambda y: y.fitness)
        new_population.append(choice)
    # We can restore the fitness here.
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


def elim_lambda_comma_mu(population: list[Candidate],
                         offspring: list[Candidate]) -> list[Candidate]:
    """Performs (lambda,mu)-elimination. Returns the new population."""
    lamda = len(population)
    mu = len(offspring)
    assert lamda <= mu, \
        f'(lambda,mu)-elimination requires lambda <= mu, got: {lamda} > {mu}'
    Candidate.sort(offspring)
    return offspring[:lamda]


def is_valid_tour(tour: list[int]) -> bool:
    """Returns True if the tour represents a valid tour, False otherwise.
    A tour is valid iff every city appears in it exactly once. Note that it does
    not matter whether the length of the tour is infinite in this test.
    """
    present = np.zeros(len(tour), dtype=bool)
    for x in tour:
        present[x] = True
    return np.all(present)


def assert_valid_tour(tour: list[int]):
    assert is_valid_tour(tour), f'Got invalid tour: {tour}'


def assert_valid_tours(candidates: list[Candidate]):
    for x in candidates:
        assert_valid_tour(x.tour)


class r0758170:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        distance_matrix = np.loadtxt(file, delimiter=",")
        file.close()

        # Parameters
        k = 5
        mutation_prob = 0.10
        lso_prob = 0.05
        lamda = 50
        mu = math.ceil(1.0 * lamda)
        elitism = 1
        greedy = 0.65
        alpha = 0.75
        sigma = math.ceil(len(distance_matrix) / 4)

        assert mu % 2 == 0, f'Mu must be even, got: {mu}'

        # Initialization
        population = []
        population.extend(init_heuristic(0, distance_matrix, fast=True, greedy=1.0))
        population.extend(init_heuristic(lamda - len(population), distance_matrix, fast=True, greedy=greedy))
        Candidate.sort(population)

        assert len(population) == lamda, f'Expected pop size == lamda, got: {len(population)} != {lamda}'

        assert_valid_tours(population)

        current_it = 1
        max_it = -1  # Set to -1 for no limit.
        while current_it != max_it + 1:
            offspring = []

            # Selection
            selected = select_k_tournament(population, k, mu)

            # Recombination
            it = iter(selected)
            for p1 in it:
                p2 = next(it)
                new_offspring = p1.recombine(p2)
                for x in new_offspring:
                    x.recalculate_fitness(distance_matrix)
                offspring.extend(new_offspring)

            assert len(offspring) == mu, f'Number of offspring ({len(offspring)}) does not match mu ({mu})'

            assert_valid_tours(offspring)

            # Mutation & Local search
            for x in itertools.chain(population[elitism:], offspring):
                if rd.random() < mutation_prob:
                    x.mutate()
                    x.recalculate_fitness(distance_matrix)
                if rd.random() < lso_prob:
                    x.local_search(distance_matrix)

            assert_valid_tours(population)
            assert_valid_tours(offspring)

            # Elimination
            population = elim_lambda_plus_mu_fitness_sharing(population, offspring, alpha, sigma, elitism)
            # population = elim_lambda_comma_mu(population, offspring)

            assert len(population) == lamda, f'Expected pop size == lamda, got: {len(population)} != {lamda}'

            assert_valid_tours(population)

            # Recalculate mean and best
            mean_objective, best_solution = Candidate.stats(population, include_inf=False)
            min_dist, avg_dist, max_dist = Candidate.distance_stats(population)

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            timeLeft = self.reporter.report(mean_objective, best_solution.fitness,
                                            np.array(best_solution.tour, dtype=int))
            print(f'{timeLeft:5.1f} sec | '
                  f'{current_it:6} | '
                  f'mean: {mean_objective:10.2f} | '
                  f'best: {best_solution.fitness:10.2f} | '
                  f'min dist: {min_dist:8.2f} | '
                  f'avg dist: {avg_dist:8.2f} | '
                  f'max dist: {max_dist:8.2f} | '
                  f'mutation prob: {mutation_prob:4.2f} | '
                  f'lso prob: {lso_prob:4.2f}'
                  )

            current_it += 1
            if timeLeft < 0:
                break
        print('Done')
        return 0
