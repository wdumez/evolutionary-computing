import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from visualize import preprocess


def get_best_fitness(filename: str) -> float:
    df = preprocess(filename)
    return float(df['Best value'].iloc[-1])


def main():
    problem = 'tour50'
    nr_benchmarks = 20
    data = {}
    for benchmark_nr in range(1, nr_benchmarks + 1):
        benchmark_filename = f'benchmark_{problem}_{benchmark_nr:03}.csv'
        filename = f'./benchmark/{benchmark_filename}'
        fit = get_best_fitness(filename)
        data[benchmark_nr] = fit
    g = sns.histplot(data)
    g.set_title('tour50 Histogram (500 benchmarks, 1000 iterations each)')
    plt.show()


if __name__ == '__main__':
    main()
