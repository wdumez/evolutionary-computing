import numpy as np
from r0758170 import *
from numpy.typing import NDArray


def create_random_candidate(size: int) -> Candidate:
    array = np.array(list(range(size)), dtype=int)
    np.random.shuffle(array)
    return Candidate(array)


with open('./tour50.csv') as file:
    distance_matrix = np.loadtxt(file, delimiter=",")

n = 50
k = 5
population = init_avoid_inf_heuristic(100, distance_matrix)
offspring = init_monte_carlo(20, distance_matrix)

sel = select_k_tournament(population, 5)
