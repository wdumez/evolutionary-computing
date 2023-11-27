import sys
import time

import numpy as np
from r0758170 import *

with open('./tour50.csv') as file:
    distance_matrix = np.loadtxt(file, delimiter=",")

pop = init_heuristic(1, distance_matrix, fast=False, greedy=1.0)
print(pop[0].fitness)
pop[0].mutate()
pop[0].recalculate_fitness(distance_matrix)
print(pop[0].fitness)
