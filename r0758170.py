import random as rd
import numpy as np
import Reporter


def mutate_inversion(candidate):
    """Mutate a candidate solution in place using inversion mutation."""
    size = candidate.size
    if size <= 1:
        return
    first_pos = rd.randint(0, size - 2)
    second_pos = rd.randint(first_pos, size - 1)
    candidate[first_pos:second_pos + 1] = np.flip(candidate[first_pos:second_pos + 1])


def recombine_PMX(parent1, parent2):
    """Use two parent candidates to produce offspring using partially mapped crossover.
    See p.70-71 in Eiben & Smith.
    """
    size = parent1.size
    offspring = np.zeros_like(parent1)
    # We must initialize offspring with -1's, to identify whether a spot is not yet filled.
    for i in range(size):
        offspring[i] = -1
    # 1.
    first_pos = rd.randint(0, size-2)
    second_pos = rd.randint(first_pos, size-1)
    offspring[first_pos:second_pos+1] = parent1[first_pos:second_pos+1]
    # 2. -  5.
    for elem in parent2[first_pos:second_pos+1]:
        if elem in parent1[first_pos:second_pos+1]:
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
    tmp = np.where(array == value)
    return tmp[0][0]
    # return np.where(array == value)[0][0]


# Modify the class name to match your student number.
class r0758170:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        rd.seed(2023)  # During testing, set the seed for reproducible results.
        self.mutation_function = mutate_inversion
        self.recombine_function = recombine_PMX

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        # Your code here.
        yourConvergenceTestsHere = True
        while yourConvergenceTestsHere:
            meanObjective = 0.0
            bestObjective = 0.0
            bestSolution = np.array([1, 2, 3, 4, 5])

            # Your code here.

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break

        # Your code here.
        return 0
