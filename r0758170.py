from __future__ import annotations

import copy
import random as rd
import sys

import math
import numpy as np
from numpy.typing import NDArray

import Reporter


class Candidate:

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
        self.mutate_func = mutate_inversion
        self.recombine_func = recombine_edge_crossover
        self.local_search_func = None
        self.fitness_func = path_length
        self.distance_func = distance_hamming
        self.recombine_operators = [
            recombine_PMX,
            recombine_cycle_crossover,
            recombine_edge_crossover,
            recombine_order_crossover
        ]
        self.mutate_operators = [
            mutate_swap,
            mutate_inversion,
            mutate_scramble,
            mutate_insert
        ]

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
        """Mutate in-place self.nr_mutations times."""
        for _ in range(self.nr_mutations):
            self.mutate_func(self.tour)
        self.mutate_func = rd.choice(self.mutate_operators)
        self.recombine_func = rd.choice(self.recombine_operators)

    def recombine(self, other: Candidate) -> list[Candidate]:
        """Recombine with another parent to produce offspring."""
        offspring = [Candidate(x) for x in self.recombine_func(self.tour, other.tour)]
        for x in offspring:
            x.mutate_func = rd.choice([self.mutate_func, other.mutate_func])
            x.recombine_func = rd.choice([self.recombine_func, other.recombine_func])
        return offspring

    def local_search(self, distance_matrix: NDArray[float]) -> None:
        """Perform a local search."""
        print('No local search implemented!')
        # self.local_search_func(self, distance_matrix)

    def recalculate_fitness(self, distance_matrix: NDArray[float]) -> None:
        """Recalculate the fitness."""
        self.fitness = self.fitness_func(self.tour, distance_matrix)

    def distance(self, other: Candidate) -> float:
        """Return the distance to another candidate."""
        return self.distance_func(self.tour, other.tour)

    def closest_to(self, others: list[Candidate]) -> Candidate:
        """Returns the closest from others using the distance metric from self."""
        return min(others, key=lambda x: self.distance(x))

    def sigma_neighborhood(self, others: list[Candidate], sigma: float) -> list[Candidate]:
        """Returns all candidates from others that are within sigma-distance from self."""
        return [x for x in others if self.distance(x) <= sigma and x is not self]


def mutate_inversion(tour: list[int]) -> None:
    """Mutate in-place using inversion mutation."""
    first_pos = rd.randrange(0, len(tour) - 1)
    second_pos = rd.randrange(first_pos + 1, len(tour))
    tour[first_pos:second_pos + 1] = np.flip(tour[first_pos:second_pos + 1])


def mutate_swap(tour: list[int]) -> None:
    """Mutate in-place using swap mutation."""
    first_pos = rd.randrange(0, len(tour))
    second_pos = first_pos
    while second_pos == first_pos:
        second_pos = rd.randrange(0, len(tour))
    tmp = tour[first_pos]
    tour[first_pos] = tour[second_pos]
    tour[second_pos] = tmp


def mutate_scramble(tour: list[int]) -> None:
    """Mutate in-place using scramble mutation."""
    first_pos = rd.randrange(0, len(tour) - 1)
    second_pos = rd.randrange(first_pos + 1, len(tour))
    np.random.shuffle(tour[first_pos:second_pos + 1])


def mutate_insert(tour: list[int]) -> None:
    """Mutate in-place using insert mutation."""
    first_pos = rd.randrange(0, len(tour) - 1)
    second_pos = rd.randrange(first_pos + 1, len(tour))
    tmp = tour[second_pos]
    tour[first_pos + 2:second_pos + 1] = tour[first_pos + 1:second_pos]
    tour[first_pos + 1] = tmp


def path_length(tour: list[int], distance_matrix: NDArray[float]) -> float:
    """Return the length of a tour."""
    result = 0.0
    for i in range(len(tour) - 1):
        # Order is important for the distance matrix.
        result += distance_matrix[tour[i]][tour[i + 1]]
    result += distance_matrix[tour[len(tour) - 1]][tour[0]]
    return result


def distance_hamming(tour1: list[int], tour2: list[int]) -> float:
    """Return the distance between two tours.
    The distance is the nr. of elements that are different.
    """
    # We must first align the tours so that they start with the same element.
    offset = tour2.index(tour1[0])
    tour2_aligned = tour2[offset:] + tour2[:offset]
    dist = 0
    for x, y in zip(tour1, tour2_aligned):
        if x != y:
            dist += 1
    return float(dist)


def recombine_copy(parent1: list[int], parent2: list[int]) -> list[list[int]]:
    """Dummy recombine function which copies the parents into the offspring."""
    return [copy.deepcopy(parent1), copy.deepcopy(parent2)]


def recombine_cycle_crossover(parent1: list[int], parent2: list[int]) -> list[list[int]]:
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


def find_cycles(parent1: list[int], parent2: list[int]) -> list[list[int]]:
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


def recombine_edge_crossover(parent1: list[int], parent2: list[int]) -> list[list[int]]:
    """Use two parent candidates to produce two offspring using edge crossover.
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
    """Create an edge table for tour1 and tour2."""
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


def get_adj(x: int, candidate: list[int]) -> list[int]:
    """Returns the adjacent values of x in candidate as a list."""
    x_idx = candidate.index(x)
    prev_idx = x_idx - 1
    next_idx = x_idx + 1 if x_idx < len(candidate) - 1 else 0
    return [int(candidate[prev_idx]), int(candidate[next_idx])]


def recombine_PMX(parent1: list[int], parent2: list[int]) -> list[list[int]]:
    """Use two parent candidates to produce two offspring using partially mapped crossover."""
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
    """Use two parent candidates to produce two offspring using order crossover."""
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


def init_monte_carlo(size: int, distance_matrix: NDArray[float]) -> list[Candidate]:
    """Initializes the population at random."""
    sample = list(range(len(distance_matrix)))
    population = []
    for i in range(size):
        rd.shuffle(sample)
        population.append(Candidate(copy.deepcopy(sample)))
    return population


def init_heuristic(size: int, distance_matrix: NDArray[float],
                   fast: bool = True, greedy: bool = True) -> list[Candidate]:
    """Initializes the population with a heuristic."""
    population = []
    for i in range(size):
        candidate = heuristic_solution(distance_matrix, fast, greedy)
        population.append(Candidate(candidate))
    return population


def heuristic_solution(distance_matrix: NDArray[float], fast: bool = True, greedy: bool = True) -> list[int]:
    """Uses a greedy heuristic to find a solution.
    If fast is True, then it returns the first found solution using a random starting position.
    If fast is False, then it tries all starting positions and returns the best found solution.
    If greedy is True, then it takes the greedy position each step.
    If greedy is False, then it takes a random step that does not lead to infinite length.
    """
    # We need to temporarily up the recursion limit because the nr. of recursions is just
    # slightly higher than the problem size (e.g. tour200 recurses 200-210 times).
    # The default limit is 1000, so for tour1000 this *just* becomes a problem.
    # Therefore, we increase the recursion limit in this function only.
    recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(10 * len(distance_matrix))

    choices = list(range(len(distance_matrix)))
    rd.shuffle(choices)
    results = []
    for start in choices:
        starting_choices = copy.deepcopy(choices)
        starting_choices.remove(start)
        result = heuristic_recursive(starting_choices, [start], distance_matrix, greedy)
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
                        greedy: bool = True) -> list[int] | bool:
    """Recursive function used in heuristic_solution."""
    if len(choices) == 0:
        if distance_matrix[current_result[-1]][current_result[0]] == math.inf:
            # The last choice cannot connect to the first choice, so backtrack.
            return False
        # All edges are valid, so the answer has been found.
        return current_result

    if greedy:
        choices.sort(key=lambda x: distance_matrix[current_result[-1]][x])

    for choice in choices:
        if distance_matrix[current_result[-1]][choice] == math.inf:
            # Not a valid choice because the distance is infinite.
            continue
        new_choices = copy.deepcopy(choices)
        new_choices.remove(choice)
        current_result.append(choice)
        answer = heuristic_recursive(new_choices, current_result, distance_matrix, greedy)
        if answer is not False:
            # An answer was found, so propagate it back up.
            return answer
        # The last choice did not lead to a solution, so backtrack.
        current_result.pop()
    # None of the choices could lead to any solutions.
    return False


def select_k_tournament(population: list[Candidate], k: int, nr_times: int = 1) -> list[Candidate]:
    """Performs a k-tournament on the population. Returns the best candidate among k random samples."""
    selected = []
    for _ in range(nr_times):
        tournament = rd.sample(population, k)
        selected.append(Candidate.most_fit(tournament))
    return selected


def select_k_tournament_fitness_sharing(population: list[Candidate],
                                        k: int,
                                        nr_times: int = 1,
                                        sigma: float = 5.0,
                                        alpha: float = 1.0) -> list[Candidate]:
    """Performs a k-tournament on the population with fitness sharing."""
    selected = []
    for _ in range(nr_times):
        tournament = []
        for _ in range(k):
            sample = rd.choice(population)
            while sample in tournament:  # Sample without replacement.
                sample = rd.choice(population)
            if sample.fitness != math.inf:
                sample.original_fitness = sample.fitness
                neighbors = sample.sigma_neighborhood(population, sigma)
                for neighbor in neighbors:
                    penalty_term = math.pow((1 - (sample.distance(neighbor) / sigma)), alpha)
                    sample.fitness += sample.original_fitness * penalty_term
            tournament.append(sample)
        selected.append(Candidate.most_fit(tournament))
        for x in tournament:
            x.fitness = x.original_fitness
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


def elim_lambda_plus_mu_fitness_sharing(population: list[Candidate],
                                        offspring: list[Candidate],
                                        alpha: float = 1.0,
                                        sigma: float = 5.0) -> list[Candidate]:
    """Performs (lambda+mu)-elimination with fitness sharing for diversity promotion.
    The sign(f(x)) is always 1 for the Traveling Salesman Problem.
    This implementation has been optimized to update incrementally.
    The fitness values of candidates are only changed temporarily.
    """
    lamda = len(population)
    population.extend(offspring)
    Candidate.sort(population)
    for x in population:
        x.original_fitness = x.fitness
    new_population = []
    while len(new_population) != lamda:
        choice = population.pop(0)
        neighbors = []
        for neighbor in choice.sigma_neighborhood(population, sigma):
            if neighbor.fitness != math.inf:
                penalty_term = math.pow((1 - (choice.distance(neighbor) / sigma)), alpha)
                neighbor.fitness += neighbor.original_fitness * penalty_term
            neighbors.append(neighbor)
            population.remove(neighbor)  # Temporarily remove this neighbor from the population.
        # Now insert the neighbors back into the population to maintain its sortedness.
        for neighbor in neighbors:
            population = insert_sorted(population, neighbor)
        new_population.append(choice)
    # We can restore the fitness here.
    for x in new_population:
        x.fitness = x.original_fitness
    return new_population


def insert_sorted(population: list[Candidate], candidate: Candidate) -> list[Candidate]:
    """Insert candidate into a sorted population such that the new population is still sorted.
    Returns the new population.
    """
    for i, x in enumerate(population):
        if candidate.fitness <= x.fitness:
            population.insert(i, candidate)
            break
    return population


def elim_lambda_comma_mu(population: list[Candidate],
                         offspring: list[Candidate]) -> list[Candidate]:
    """Performs (lambda,mu)-elimination. Returns the new population."""
    lamda = len(population)
    mu = len(offspring)
    assert lamda <= mu, \
        f'(lambda,mu)-elimination requires lambda <= mu, got: {lamda} > {mu}'
    Candidate.sort(offspring)
    return offspring[:lamda]


def elim_age_based(population: list[Candidate],
                   offspring: list[Candidate]) -> list[Candidate]:
    """Performs age-based elimination. Returns the new population.
    This is really just a specific case of (lamda,mu)-elimination.
    """
    assert len(population) == len(offspring), \
        f'Age based elimination requires lambda == mu, got: {len(population)} != {len(offspring)}'
    return elim_lambda_comma_mu(population, offspring)


def elim_k_tournament(population: list[Candidate],
                      offspring: list[Candidate],
                      k: int) -> list[Candidate]:
    """Performs k-tournament elimination. Returns the new population."""
    lamda = len(population)
    population.extend(offspring)
    return select_k_tournament(population, k, lamda)


def is_valid_tour(tour: list[int]) -> bool:
    """Returns True if the candidate represents a valid tour, False otherwise.
    A tour is valid if every city appears in it exactly once. Note that it does
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

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        distance_matrix = np.loadtxt(file, delimiter=",")
        file.close()

        # seed = 3
        # rd.seed(seed)
        # np.random.seed(seed)

        # Parameters
        # TODO These should eventually be moved into the Candidate class,
        #      so they can be used for self-adaptivity.
        k = 3
        mutation_prob = 0.10
        lamda = 60
        mu = int(1.5 * lamda)
        seed_fraction = 0.05
        nr_seeds = math.ceil(lamda * seed_fraction)
        alpha = 1.0
        sigma = len(distance_matrix) // 10

        assert mu % 2 == 0, f'Mu must be even, got: {mu}'

        # Initialization

        # Seeding:
        population = []
        population.extend(init_heuristic(nr_seeds, distance_matrix, fast=True, greedy=True))
        population.extend(init_heuristic(lamda - len(population), distance_matrix, fast=True, greedy=False))
        # population.extend(init_monte_carlo(lamda - len(population), distance_matrix))

        for x in population:
            x.recalculate_fitness(distance_matrix)

        assert_valid_tours(population)

        current_it = 1
        max_it = 300
        while current_it <= max_it:
            offspring = []

            # Selection
            selected = select_k_tournament(population, k, mu)
            # selected = select_k_tournament_fitness_sharing(population, k, mu)

            # Recombination
            it = iter(selected)
            for p1 in it:
                p2 = next(it)
                new_offspring = p1.recombine(p2)
                for x in new_offspring:
                    x.recalculate_fitness(distance_matrix)
                offspring.extend(new_offspring)

            assert len(offspring) == mu, f'Nr. offspring ({len(offspring)}) does not match mu ({mu})'

            assert_valid_tours(population)
            assert_valid_tours(offspring)

            # Mutation & Local search
            for x in offspring:
                if rd.random() < mutation_prob:
                    x.mutate()
                    x.recalculate_fitness(distance_matrix)
                    # x.local_search(distance_matrix, 2)
                    # x.recalculate_fitness(distance_matrix)

            assert_valid_tours(population)
            assert_valid_tours(offspring)

            # for x in population:
            #     x.recalculate_fitness(distance_matrix)

            # Elimination
            # population = elim_lambda_plus_mu(population, offspring)
            population = elim_lambda_plus_mu_fitness_sharing(population, offspring, alpha, sigma)
            # population = elim_lambda_comma_mu(population, offspring)
            # population = elim_k_tournament(population, offspring, k)

            assert_valid_tours(population)

            # Recalculate mean and best
            mean_objective, best_solution = Candidate.stats(population, include_inf=False)

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
                  f'best: {best_solution.fitness:10.2f}')
            current_it += 1
            if timeLeft < 0:
                break
        print('Done')
        return 0
