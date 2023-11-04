import numpy as np
from r0758170 import *


def create_random_candidate(size: int):
    array = np.array(list(range(size)))
    np.random.shuffle(array)
    return array


with open('./tour50.csv') as file:
    distance_matrix = np.loadtxt(file, delimiter=",")
