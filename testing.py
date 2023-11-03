import numpy as np

from r0758170 import recombine_cycle_crossover, Candidate, mutate_insert


def create_random_candidate(size: int):
    array = np.array(list(range(size)))
    np.random.shuffle(array)
    return array


a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
b = np.array([9, 3, 7, 8, 2, 6, 5, 1, 4])

a = Candidate(a)
b = Candidate(b)

print(a)
mutate_insert(a)
print(a)
