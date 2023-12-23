import itertools

import numpy as np
import random as rd

import r0758170

import math


def exact(distance_matrix):
    best_tour = list(range(len(distance_matrix)))
    best_fit = math.inf
    for tour in itertools.permutations(range(len(distance_matrix))):
        fit = r0758170.path_length(list(tour), distance_matrix)
        if fit < best_fit:
            best_tour = tour
            best_fit = fit
    return best_tour, best_fit


def main():
    with open('./tour50.csv') as file:
        distance_matrix = np.loadtxt(file, delimiter=",")
    tour, fit = exact(distance_matrix)
    print(tour)
    print(fit)


if __name__ == '__main__':
    main()
