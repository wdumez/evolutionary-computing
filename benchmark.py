import os

import r0758170

for benchmark_nr in range(1, 5 + 1):
    filename = "./tour200.csv"
    a = r0758170.r0758170()
    a.optimize(filename)
    os.system(f'cp ./r0758170.csv ./benchmark/benchmark_{benchmark_nr:03}.csv')
