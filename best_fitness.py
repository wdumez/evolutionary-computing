import pandas as pd
import math

import histogram


def main():
    problem = 'tour1000'
    start = 1
    nr_benchmarks = 20
    best_fit = math.inf
    best_nr = 1
    for nr in range(start, nr_benchmarks + start):
        filename = f'./benchmark/benchmark_{problem}_{nr:03}.csv'
        fit, _ = histogram.get_fitness(filename)
        print(f'{nr} has fitness: {fit}')
        if fit < best_fit:
            best_fit = fit
            best_nr = nr
    print(f'{best_nr} was the best benchmark, with fitness: {best_fit}')


if __name__ == '__main__':
    main()
