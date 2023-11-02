import numpy as np

from r0758170 import Candidate


def create_random_array(size: int):
    array = np.array(list(range(size)))
    np.random.shuffle(array)
    return array


a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
b = np.array([9, 3, 7, 8, 2, 6, 5, 1, 4])

a_c = Candidate(a)
print(a_c)
a_c.mutate_inversion()
print(a_c)
