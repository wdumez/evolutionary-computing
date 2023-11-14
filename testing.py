import sys

import numpy as np
from r0758170 import *


def create_random_candidate(size: int) -> Candidate:
    array = np.array(list(range(size)), dtype=int)
    np.random.shuffle(array)
    return Candidate(array)


with open('./tour50.csv') as file:
    distance_matrix = np.loadtxt(file, delimiter=",")

n = 50
k = 5
lamda = 10
mu = 2

pop = init_heuristic(20, distance_matrix, fast=True, greedy=True)
a = pop[0]
a.recalculate_fitness(distance_matrix)
print(a.fitness)
a.local_search(distance_matrix, 100)
a.recalculate_fitness(distance_matrix)
print(a.fitness)
