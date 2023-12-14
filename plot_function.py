import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Helvetica'


def func(t: float, t_max: float, f_min: float, f_max: float, power: float) -> float:
    return f_min + (f_max - f_min) * math.pow((t_max - t) / t_max, power)


def main():
    t_max = 300.0
    alpha_min = 0.5
    alpha_max = 3.0
    power = 2.5
    t_range = np.linspace(0.0, 300.0, 1000)
    data = {t: func(t, t_max, alpha_min, alpha_max, power) for t in t_range}
    g = sns.lineplot(data)
    g.set_title(r'$\alpha$ in function of the time', fontsize=16)
    g.set_xlabel(r'$t$ (seconds)', fontsize=14)
    g.set_ylabel(r'$\alpha(t)$', rotation=0, fontsize=14)
    plt.show()


if __name__ == '__main__':
    main()
