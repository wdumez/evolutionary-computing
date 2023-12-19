import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Helvetica'


def alpha(t: float, t_max: float, alpha_min: float, alpha_max: float, power: float) -> float:
    return alpha_min + (alpha_max - alpha_min) * math.pow((t_max - t) / t_max, power)


def k(t: float, t_max: float, k_min: int, k_max: int, power: float) -> int:
    return round(k_min + (k_max - k_min) * math.pow((t / t_max), power))


def sigma(problem_size: int, power) -> int:
    percentage = 0.025 + (0.20 - 0.025) * math.pow((problem_size - 50.0) / (1000.0 - 50.0), power)
    return round(percentage * problem_size)


def main():
    t_max = 300.0
    alpha_min = 0.001
    alpha_max = 3.0
    k_min = 3
    k_max = 7
    power = 1.0
    t_range = np.linspace(0.0, 300.0, 1000)
    pr_size_range = [50, 100, 200, 500, 750, 1000]
    data = {t: alpha(t, t_max, alpha_min, alpha_max, power) for t in t_range}
    # data = {t: k(t, t_max, k_min, k_max, power) for t in t_range}
    # data = {problem_size: sigma(problem_size, 1.0) for problem_size in pr_size_range}
    print(data)
    g = sns.lineplot(data)
    g.set_title(r'$\alpha$ in function of the time', fontsize=16)
    g.set_xlabel(r'$t$ (seconds)', fontsize=14)
    g.set_ylabel(r'$\alpha(t)$', rotation=0, fontsize=14)
    plt.show()


if __name__ == '__main__':
    main()
