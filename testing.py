import sys
import time

import numpy as np
from r0758170 import *

with open('./tour50.csv') as file:
    distance_matrix = np.loadtxt(file, delimiter=",")

t = list(range(10))
print(t)
t = mutate_inversion(t)
print(t)
