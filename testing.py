from r0758170 import recombine_PMX, length, monte_carlo, avoid_inf
import numpy as np

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
b = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
# b = np.array([9,3,7,8,2,6,5,1,4])

c = recombine_PMX(a, b)
print(c)

# file = open("./tour200.csv")
# distanceMatrix = np.loadtxt(file, delimiter=",")
# file.close()
#
# candidate = avoid_inf(distanceMatrix, 1)[0]
# print(candidate)
# print(length(candidate, distanceMatrix))
