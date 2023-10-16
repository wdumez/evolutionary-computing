from r0758170 import recombine_PMX
import numpy as np

a = np.array([1,2,3,4,5,6,7,8,9])
b = np.array([9,3,7,8,2,6,5,1,4])

c = recombine_PMX(a, b)
print(c)