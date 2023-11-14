import sys
import time

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

arr = np.array(range(n), dtype=int)
lst = list(range(n))

start_time = time.time()
for x in arr:
    pass
stop_time = time.time()
print(f'Array: {stop_time - start_time}')

start_time = time.time()
for x in lst:
    pass
stop_time = time.time()
print(f'List: {stop_time - start_time}')

start_time = time.time()
for _ in range(100000):
    arr = np.array(range(n), dtype=int)
    path_length(arr, distance_matrix)
stop_time = time.time()
print(f'Array: {stop_time - start_time}')

start_time = time.time()
for _ in range(100000):
    lst = list(range(n))
    path_length(lst, distance_matrix)
stop_time = time.time()
print(f'List: {stop_time - start_time}')
