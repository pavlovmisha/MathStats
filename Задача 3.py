import numpy as np
import matplotlib.pyplot as plt

# Распределения и параметры
distributions = {
    'Normal': {'func': lambda size: np.random.normal(0, 1, size)},
    'Cauchy': {'func': lambda size: np.random.standard_cauchy(size)},
    'Laplace': {'func': lambda size: np.random.laplace(0, 1/np.sqrt(2), size)},
    'Poisson': {'func': lambda size: np.random.poisson(5, size)},
    'Uniform': {'func': lambda size: np.random.uniform(-np.sqrt(3), np.sqrt(3), size)}
}

sample_sizes = [20, 100]
repeats_for_outliers = 1000

for n in sample_sizes:
    data = []
    labels = []
    for dist_name, dist_info in distributions.items():
        sample = dist_info['func'](n)
        data.append(sample)
        labels.append(dist_name)
    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.title(f'Боксплоты Тьюки для распределений (n={n})')
    plt.ylabel('Значения')
    plt.show()

for dist_name, dist_info in distributions.items():
    plt.figure(figsize=(8, 5))
    data = []
    for n in sample_sizes:
        sample = dist_info['func'](n)
        data.append(sample)
    plt.boxplot(data, labels=[f'n={n}' for n in sample_sizes], showfliers=True)
    plt.title(f'Боксплоты Тьюки для {dist_name}')
    plt.ylabel('Значения')
    plt.legend([f'{dist_name} (n={n})' for n in sample_sizes])
    plt.show()


# 2. Определение доли выбросов по тесту Тьюки
# Правила Тьюки: за границы берутся интерквантильные границы (Q1 - 1.5*IQR, Q3 + 1.5*IQR),
# или по аналогии - для нормального распределения: границы (Q1 - 3*IQR, Q3 + 3*IQR),
# но здесь лучше использовать стандартные межквартильные границы.

def compute_outlier_fraction(sample):
    q1, q3 = np.percentile(sample, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = np.sum((sample < lower_bound) | (sample > upper_bound))
    return outliers / len(sample)

# Для каждого распределения и каждого n считаем среднюю долю выбросов
results_outliers = {}

for dist_name, dist_info in distributions.items():
    results_outliers[dist_name] = {}
    for n in sample_sizes:
        outlier_fractions = []
        for _ in range(repeats_for_outliers):
            sample = dist_info['func'](n)
            fraction = compute_outlier_fraction(sample)
            outlier_fractions.append(fraction)
        mean_fraction = np.mean(outlier_fractions)
        results_outliers[dist_name][n] = mean_fraction

# Вывод результатов
for dist_name in results_outliers:
    print(f"Распределение: {dist_name}")
    for n in sample_sizes:
        print(f"  Размер выборки = {n}: средняя доля выбросов = {results_outliers[dist_name][n]:.4f}")
