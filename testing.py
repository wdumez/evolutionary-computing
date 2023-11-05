import numpy as np
from r0758170 import *


def create_random_candidate(size: int):
    array = np.array(list(range(size)))
    np.random.shuffle(array)
    return array


with open('./tour50.csv') as file:
    distance_matrix = np.loadtxt(file, delimiter=",")

a = create_random_candidate(3)
b = create_random_candidate(3)
print(a)
print(b)
dist = distance(a, b)
print(dist)
