import sys
import time

import numpy as np
from r0758170 import *

with open('./tour50.csv') as file:
    distance_matrix = np.loadtxt(file, delimiter=",")

n = 50
k = 5
lamda = 10
mu = 2


def insert_sorted(lst, x):
    """Insert candidate into a sorted population such that the new population is still sorted.
    Returns the new population.
    """
    if len(lst) == 0:
        return [x]
    b = False
    for i, y in enumerate(lst):
        if x <= y:
            lst.insert(i, x)
            b = True
            break
    if not b:
        lst.append(x)
    return lst


a = [1, 3, 5, 7, 9]
b = insert_sorted(a, 10)
print(b)
