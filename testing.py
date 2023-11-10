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

pop = init_avoid_inf_heuristic(lamda, distance_matrix)
off = init_monte_carlo(mu, distance_matrix)
for x in itertools.chain(pop, off):
    x.recalculate_fitness(distance_matrix)

c = Candidate(np.array(list(range(n)), dtype=int))
print(c)
for x in neighborhood_inversion(c, 4):
    print(x)
