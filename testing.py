import numpy as np
import random as rd
from r0758170 import recombine_cycle_crossover, recombine_edge_crossover, create_adj_table, remove_references, \
    pick_next_element


def create_random_array(size: int):
    array = np.array(list(range(size)))
    np.random.shuffle(array)
    return array


a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
b = np.array([9, 3, 7, 8, 2, 6, 5, 1, 4])

adj_table = create_adj_table(a, b)
print(adj_table)

c_yes = np.array([1, 5, 6, 2, 8, 7, 3, 9, 4])
while True:
    a = create_random_array(20)
    b = create_random_array(20)
    c = recombine_edge_crossover(a, b)
    print(c)
    # if np.array_equal(c, c_yes):
    #     break
