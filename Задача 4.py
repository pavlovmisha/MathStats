import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, cauchy, laplace, poisson, uniform, gaussian_kde

# Распределения и параметры
distributions = {
    'Normal': {'scipy': norm(0, 1), 'gen': lambda n: np.random.normal(0, 1, n)},
    'Cauchy': {'scipy': cauchy(0, 1), 'gen': lambda n: np.random.standard_cauchy(n)},
    'Laplace': {'scipy': laplace(0, 1 / np.sqrt(2)), 'gen': lambda n: np.random.laplace(0, 1 / np.sqrt(2), n)},
    'Poisson': {'scipy': poisson(5), 'gen': lambda n: np.random.poisson(5, n)},
    'Uniform': {'scipy': uniform(-np.sqrt(3), 2 * np.sqrt(3)),
                'gen': lambda n: np.random.uniform(-np.sqrt(3), np.sqrt(3), n)},
}

sample_sizes = [20, 60, 100]

for dist_name, dist_obj in distributions.items():
    for n in sample_sizes:
        sample = dist_obj['gen'](n)

        # --- 1. Эмпирическая и теоретическая функции распределения ---
        if dist_name == 'Poisson':
            x = np.arange(6, 15)
            # ЭФР строим по гистограмме или вручную
            ecdf = np.array([np.mean(sample <= xi) for xi in x])
            cdf = dist_obj['scipy'].cdf(x)
        else:
            x = np.linspace(-4, 4, 200)
            ecdf = np.array([np.mean(sample <= xi) for xi in x])
            cdf = dist_obj['scipy'].cdf(x)

        plt.figure(figsize=(7, 4))
        plt.step(x, ecdf, where='post', label='Эмпирическая функция', color='blue')
        plt.plot(x, cdf, label='Теоретическая функция', color='red')
        plt.title(f'ЭФР и теоретическая CDF\n{dist_name}, n={n}')
        plt.xlabel('x')
        plt.ylabel('F(x)')
        plt.legend()
        plt.grid()
        plt.show()

        # --- 2. Ядерная оценка плотности + теоретическая плотность ---
        plt.figure(figsize=(7, 4))
        if dist_name != 'Poisson':
            x = np.linspace(-4, 4, 200)
            kde = gaussian_kde(sample)  #bw_method
            plt.plot(x, kde(x), label='Ядерная оценка', color='blue')
            plt.plot(x, dist_obj['scipy'].pdf(x), label='Теоретическая плотность', color='red')
            plt.xlabel('x')
            plt.ylabel('Плотность')
            plt.title(f'KDE/Гистограмма и теоретическая плотность\n{dist_name}, n={n}')
            plt.legend()
            plt.grid()
            plt.show()