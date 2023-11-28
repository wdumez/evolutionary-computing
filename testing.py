import sys
import time

import numpy as np
from r0758170 import *

with open('./tour50.csv') as file:
    distance_matrix = np.loadtxt(file, delimiter=",")

pop = init_heuristic(1, distance_matrix, fast=False, greedy=0.0)
print(pop[0].fitness)
pop[0].local_search(distance_matrix, 10)
pop[0].recalculate_fitness(distance_matrix)
print(pop[0].fitness)
