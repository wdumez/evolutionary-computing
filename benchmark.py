import os

import r0758170

for benchmark_nr in range(1, 500 + 1):
    filename = "./tour50.csv"
    a = r0758170.r0758170()
    a.optimize(filename)
    os.system(f'cp ./r0758170.csv ./tour50_benchmark/tour50_{benchmark_nr:03}.csv')
