import pandas as pd
import math

import histogram


def main():
    problem = 'tour50'
    start = 1
    nr_benchmarks = 100
    mean_fits = []
    best_fits = []
    for nr in range(start, nr_benchmarks + start):
        filename = f'./benchmark/benchmark_{problem}_{nr:03}.csv'
        best, mean = histogram.get_fitness(filename)
        best_fits.append(best)
        mean_fits.append(mean)
    data = {'best': best_fits, 'mean': mean_fits}
    df = pd.DataFrame(data)
    print(df.describe())


if __name__ == '__main__':
    main()
