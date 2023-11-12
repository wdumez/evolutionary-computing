import sys

import numpy as np
from r0758170 import *


def create_random_candidate(size: int) -> Candidate:
    array = np.array(list(range(size)), dtype=int)
    np.random.shuffle(array)
    return Candidate(array)


with open('./tour100.csv') as file:
    distance_matrix = np.loadtxt(file, delimiter=",")

n = 50
k = 5
lamda = 10
mu = 2

a = [recombine_probabilities(0.01, 0.10, 1.0) for i in range(1000000)]
sum = sum(a)
print(sum / len(a))
# sys.setrecursionlimit(10 * len(distance_matrix))
# heur = heuristic_solution(distance_matrix, True, False)
# heur.recalculate_fitness(distance_matrix)
# # print(heur)
# print(heur.fitness)
