from r0758170 import recombine_PMX, recombine_edge_crossover
import numpy as np

a = np.array([1,2,3,4,5,6,7,8,9])
b = np.array([9,3,7,8,2,6,5,1,4])

c = np.zeros_like(a)
while not np.array_equal(np.array([1,5,6,2,8,7,3,9,4]), c):
    c = recombine_edge_crossover(a, b)
    print(c)
