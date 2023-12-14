import sys
import time

import numpy as np
from r0758170 import *

with open('./tour50.csv') as file:
    distance_matrix = np.loadtxt(file, delimiter=",")

t = [0, 1, 2, 3, 4]
t = insert(t, 3, 0)
print(t)
