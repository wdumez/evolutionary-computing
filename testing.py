import sys
import time

import numpy as np
from r0758170 import *

with open('./tour1000.csv') as file:
    distance_matrix = np.loadtxt(file, delimiter=",")

pop = init_heuristic(1, distance_matrix, fast=True, greediness=0.5)
c = pop[0]
print(c.fitness)
c.local_search(distance_matrix)
c.recalculate_fitness(distance_matrix)
print(c.fitness)
