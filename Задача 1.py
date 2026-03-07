import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, cauchy, laplace, poisson, uniform

# Размер выборки
sample_sizes = [10, 100, 1000]

# Параметры распределений
params = {
    'Normal': {'loc': 0, 'scale': 1},
    'Cauchy': {'loc': 0, 'scale': 1},
    'Laplace': {'loc': 0, 'scale': 1 / np.sqrt(2)},
    'Poisson': {'mu': 5},
    'Uniform': {'loc': -np.sqrt(3), 'scale': 2 * np.sqrt(3)}  # от -√3 до √3
}

# Генерируем выборки
np.random.seed(0)
data_samples = {}

for dist_name in ['Normal', 'Cauchy', 'Laplace', 'Poisson', 'Uniform']:
    data_samples[dist_name] = {}
    for size in sample_sizes:
        if dist_name == 'Normal':
            sample = np.random.normal(params['Normal']['loc'], params['Normal']['scale'], size)
        elif dist_name == 'Cauchy':
            sample = np.random.standard_cauchy(size) * params['Cauchy']['scale'] + params['Cauchy']['loc']
        elif dist_name == 'Laplace':
            sample = np.random.laplace(params['Laplace']['loc'], params['Laplace']['scale'], size)
        elif dist_name == 'Poisson':
            sample = np.random.poisson(params['Poisson']['mu'], size)
        elif dist_name == 'Uniform':
            sample = np.random.uniform(params['Uniform']['loc'],
                                       params['Uniform']['loc'] + params['Uniform']['scale'], size)
        data_samples[dist_name][size] = sample

# Создаём отдельный график для каждого распределения и размера выборки
for dist_name in ['Normal', 'Cauchy', 'Laplace', 'Poisson', 'Uniform']:
    for size in sample_sizes:
        plt.figure(figsize=(8, 6))
        sample = data_samples[dist_name][size]
        # Определяем диапазон для графика
        if dist_name == 'Poisson':
            x = np.arange(0, max(sample) + 5)
        elif dist_name == 'Cauchy':
            x = np.linspace(-10, 10, 300)
        elif dist_name == 'Normal':
            x = np.linspace(-5, 5, 300)
        elif dist_name == 'Laplace':
            x = np.linspace(-5, 5, 300)
        elif dist_name == 'Uniform':
            x = np.linspace(-np.sqrt(3) - 0.5, np.sqrt(3) + 0.5, 300)

        # Построение гистограммы
        if (dist_name == 'Cauchy') & (size == 1000):
            plt.hist(sample, bins=300, density=True, alpha=0.7, label='Выборка')
        elif (dist_name == 'Cauchy') & (size == 100):
            plt.hist(sample, bins=50, density=True, alpha=0.7, label='Выборка')
        elif (dist_name != 'Cauchy') & (size == 100) & (dist_name != 'Poisson'):
            plt.hist(sample, bins=20, density=True, alpha=0.7, label='Выборка')
        elif (dist_name != 'Cauchy') & (size == 1000) & (dist_name != 'Poisson'):
            plt.hist(sample, bins=50, density=True, alpha=0.7, label='Выборка')
        elif (dist_name == 'Poisson') & (size == 1000):
            plt.hist(sample, bins=14, density=True, alpha=0.7, label='Выборка')
        elif (dist_name == 'Poisson') & (size == 100):
            plt.hist(sample, bins=10, density=True, alpha=0.7, label='Выборка')
        elif (dist_name == 'Poisson') & (size == 10):
            plt.hist(sample, bins=9, density=True, alpha=0.7, label='Выборка')
        else:
            plt.hist(sample, bins=5, density=True, alpha=0.7, label='Выборка')

        # Теоретическая плотность
        if dist_name == 'Normal':
            y = norm.pdf(x, **params['Normal'])
        elif dist_name == 'Cauchy':
            y = cauchy.pdf(x, **params['Cauchy'])
        elif dist_name == 'Laplace':
            y = laplace.pdf(x, **params['Laplace'])
        elif dist_name == 'Poisson':
            y = poisson.pmf(np.round(x).astype(int), **params['Poisson'])
        elif dist_name == 'Uniform':
            y = uniform.pdf(x, **params['Uniform'])
        plt.plot(x, y, 'k-', linewidth=2, label='Теоретическая плотность')

        plt.title(f'{dist_name} Distribution\nSize={size}')
        plt.xlabel('x')
        plt.ylabel('Плотность')
        plt.legend()
        plt.show()