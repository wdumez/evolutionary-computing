import os

import r0758170
import matplotlib.pyplot as plt
import visualize

start_nr = 51
nr_benchmarks = 50

for benchmark_nr in range(start_nr, start_nr + nr_benchmarks):
    problem = 'tour50'
    filename = f'./{problem}.csv'
    benchmark_filename = f'benchmark_{problem}_{benchmark_nr:03}'

    a = r0758170.r0758170()
    a.optimize(filename)

    os.system(f'cp ./r0758170.csv ./benchmark/{benchmark_filename}.csv')

    g = visualize.plot(f'./benchmark/{benchmark_filename}.csv')
    plt.savefig(f'./benchmark/{benchmark_filename}.png')
    plt.close()
