from io import StringIO

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_heuristic(problem_size: int) -> float:
    """Get the heuristic-to-beat for the problem size."""
    heuristics = {
        50: 27723,
        100: 90851,
        200: 39745,
        500: 157034,
        750: 197541,
        1000: 195848
    }
    return float(heuristics[problem_size])


def get_problem_size(filename) -> int:
    """Get the problem size of a solution file."""
    df = pd.read_csv(filename, skiprows=2, header=0, skipinitialspace=True)
    problem_size = df.shape[1] - 5
    return problem_size


def preprocess(filename: str) -> pd.DataFrame:
    """Reads in the data with preprocessing applied."""
    new_file = ''
    with open(filename, 'r') as file:
        for i, line in enumerate(file.readlines()):
            if i == 0:
                new_file += line
                continue
            words = line.split(',')
            words = words[0:4]
            new_file += ','.join(words) + '\n'
    return pd.read_csv(StringIO(new_file),
                       skiprows=1, header=0, skipinitialspace=True)


def plot(filename: str = './r0758170.csv'):
    df = preprocess(filename)
    problem_size = get_problem_size(filename)
    heuristic = get_heuristic(problem_size)
    df['Heuristic'] = [heuristic for _ in df['Elapsed time']]
    top = max([float(x) for x in df['Mean value'].iloc[5:]])
    bottom = 0.975 * min(heuristic, min([float(x) for x in df['Best value']]))
    nr_iterations = len(df)
    final_best_fit = float(df['Best value'].iloc[-1])
    df = pd.melt(df, ['Elapsed time', '# Iteration'], var_name='Objective', value_name='Fitness')
    g = sns.lineplot(df, x='Elapsed time', y='Fitness', hue='Objective')
    g.set_title(f'Size: {problem_size} | Iterations: {nr_iterations} | Best fitness: {final_best_fit:.2f}')
    g.set_xlabel('Elapsed time (sec)')
    g.set_ylabel('Fitness (tour length)')
    g.set_ylim(top=top, bottom=bottom, auto=True)
    return g


if __name__ == '__main__':
    _ = plot()
    plt.show()
