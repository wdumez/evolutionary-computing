from io import StringIO

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def preprocess(filename):
    """Reads in the data with preprocessing applied."""
    new_file = ''
    with open(filename, 'r') as file:
        for i, line in enumerate(file.readlines()):
            if i == 0:
                new_file += line
                continue
            words = line.split(',')
            words = words[1:4]
            new_file += ','.join(words) + '\n'
    return pd.read_csv(StringIO(new_file),
                       skiprows=1, header=0, skipinitialspace=True)


df = preprocess('./r0758170.csv')
df = pd.melt(df, ['Elapsed time'], var_name='Objective', value_name='Fitness')
g = sns.lineplot(df, x='Elapsed time', y='Fitness', hue='Objective')
g.set_title('TSP')
g.set_xlabel('Elapsed time (sec)')
g.set_ylabel('Fitness (tour length)')
plt.show()
