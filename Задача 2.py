import numpy as np

# Параметры
n_values = [10, 100, 1000]
repeats = 1000

# Распределения и их параметры
distributions = {
    'Normal': {'func': lambda size: np.random.normal(0, 1, size)},
    'Cauchy': {'func': lambda size: np.random.standard_cauchy(size)},
    'Laplace': {'func': lambda size: np.random.laplace(0, 1 / np.sqrt(2), size)},
    'Poisson': {'func': lambda size: np.random.poisson(5, size)},
    'Uniform': {'func': lambda size: np.random.uniform(-np.sqrt(3), np.sqrt(3), size)}
}

results = {}

for dist_name, dist_info in distributions.items():
    results[dist_name] = {}
    for n in n_values:
        # Создаем массив для вывода характеристик для каждого повторения
        mean_vals = np.empty(repeats)
        med_vals = np.empty(repeats)
        zR_vals = np.empty(repeats)
        zQ_vals = np.empty(repeats)
        ztr_vals = np.empty(repeats)

        for i in range(repeats):
            sample = dist_info['func'](n)
            sorted_sample = np.sort(sample)
            xmin, xmax = sorted_sample[0], sorted_sample[-1]
            med = np.median(sample)
            zR = (xmin + xmax) / 2

            # квартильные позиции для 1/4 и 3/4
            q1 = np.percentile(sample, 25)
            q3 = np.percentile(sample, 75)
            zQ = (q1 + q3) / 2

            # Усекаем 10% снизу и сверху для Z_tr
            lower_idx = int(0.1 * n)
            upper_idx = int(0.9 * n)
            trimmed_sample = sample[lower_idx:upper_idx]
            ztr = np.mean(trimmed_sample)

            # Запоминаем
            mean_vals[i] = np.mean(sample)
            med_vals[i] = med
            zR_vals[i] = zR
            zQ_vals[i] = zQ
            ztr_vals[i] = ztr

        # Сохраняем результаты: среднее и дисперсию по 1000 повторениям
        results[dist_name][n] = {
            'mean': (np.mean(mean_vals), np.var(mean_vals)),
            'med': (np.mean(med_vals), np.var(med_vals)),
            'zR': (np.mean(zR_vals), np.var(zR_vals)),
            'zQ': (np.mean(zQ_vals), np.var(zQ_vals)),
            'ztr': (np.mean(ztr_vals), np.var(ztr_vals))
        }

# Теперь выводим результаты
for dist_name in results:
    print(f"\nРаспределение: {dist_name}")
    for n in results[dist_name]:
        print(f"Объем выборки: {n}")
        for char_name in ['mean', 'med', 'zR', 'zQ', 'ztr']:
            mean_, var_ = results[dist_name][n][char_name]
            print(f"  {char_name}: Среднее = {mean_:.4f}, Дисперсия = {var_:.6f}")