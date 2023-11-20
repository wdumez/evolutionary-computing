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

a = [0, 1, 2, 3, 4]
b = [1, 2, 0, 4, 3]
dist = distance_hamming(a, b)
print(dist)
