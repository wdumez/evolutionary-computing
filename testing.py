import sys

import numpy as np
from r0758170 import *


def create_random_candidate(size: int) -> Candidate:
    array = np.array(list(range(size)), dtype=int)
    np.random.shuffle(array)
    return Candidate(array)


with open('./tour1000.csv') as file:
    distance_matrix = np.loadtxt(file, delimiter=",")

n = 50
k = 5
lamda = 10
mu = 2

sys.setrecursionlimit(10 * len(distance_matrix))
heur = greedy_heuristic(distance_matrix, True)
print(heur)
print(heur.fitness)
