import sys
import time

import numpy as np
from r0758170 import *

with open('./tour1000.csv') as file:
    distance_matrix = np.loadtxt(file, delimiter=",")

pop = init_heuristic(10, distance_matrix, fast=True, greedy=0.80)
for x in pop:
    x.recalculate_fitness(distance_matrix)
    print(x.fitness)
