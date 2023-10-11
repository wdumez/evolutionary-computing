import Reporter
import numpy as np
import random as rd


def swap_mutate(candidate):
    """Mutate a candidate solution in place."""
    if candidate.size <= 1:
        return candidate
    first_pos = rd.randint(0, candidate.size - 2)
    second_pos = rd.randint(first_pos, candidate.size - 1)
    candidate[first_pos:second_pos + 1] = np.flip(candidate[first_pos:second_pos + 1])


def PMX_recombine(parent1, parent2):
    """Use two parent candidates to produce offspring using PMX recombination."""
    raise NotImplementedError


# Modify the class name to match your student number.
class r0758170:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        rd.seed(2023)  # During testing, set the seed for reproducible results.
        self.mutation_function = swap_mutate
        self.recombine_function = PMX_recombine

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
