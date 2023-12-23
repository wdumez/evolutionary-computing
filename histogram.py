import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from visualize import preprocess


def get_fitness(filename: str) -> tuple[float, float]:
    df = preprocess(filename)
    return float(df['Best value'].iloc[-1]), float(df['Mean value'].iloc[-1])


def create_df(nr_benchmarks, problem) -> pd.DataFrame:
    data = {'best': [], 'mean': []}
    for benchmark_nr in range(1, nr_benchmarks + 1):
        benchmark_filename = f'benchmark_{problem}_{benchmark_nr:03}.csv'
        filename = f'./benchmark/{benchmark_filename}'
        best_fit, mean_fit = get_fitness(filename)
        data['best'].append(best_fit)
        data['mean'].append(mean_fit)
    return pd.DataFrame(data)


def main():
    problem = 'tour50'
    nr_benchmarks = 100
    df = create_df(nr_benchmarks, problem)
    # df = df.melt()
    g = sns.histplot(df, element='step', binwidth=100)
    g.set_title(f'{problem} Histogram ({nr_benchmarks} benchmarks)')
    g.set_xlabel('Fitness')
    plt.show()


if __name__ == '__main__':
    main()
