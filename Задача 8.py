import numpy as np
from scipy import stats

# Для воспроизводимости результатов
np.random.seed(2024)

# 1. Генерация выборок
n1, n2 = 20, 100
sample1 = np.random.normal(loc=0.0, scale=1.0, size=n1)
sample2 = np.random.normal(loc=0.0, scale=1.0, size=n2)

# Точечные оценки
mean1, std1, var1 = np.mean(sample1), np.std(sample1, ddof=1), np.var(sample1, ddof=1)
mean2, std2, var2 = np.mean(sample2), np.std(sample2, ddof=1), np.var(sample2, ddof=1)

print("Выборочные характеристики:")
print(f"Выборка 1 (n={n1}): среднее = {mean1:.4f}, станд. откл. = {std1:.4f}, дисперсия = {var1:.4f}")
print(f"Выборка 2 (n={n2}): среднее = {mean2:.4f}, станд. откл. = {std2:.4f}, дисперсия = {var2:.4f}")
print()

# Уровень доверия
alpha = 0.05
conf_level = 1 - alpha

# 2. Доверительный интервал для математического ожидания
# Используем t-распределение Стьюдента
def mean_ci(sample, alpha=0.05):
    n = len(sample)
    mean = np.mean(sample)
    sem = stats.sem(sample)  # стандартная ошибка среднего: std / sqrt(n)
    df = n - 1
    t_crit = stats.t.ppf(1 - alpha/2, df)
    margin = t_crit * sem
    return (mean - margin, mean + margin, mean, sem, t_crit, df)

ci1 = mean_ci(sample1, alpha)
ci2 = mean_ci(sample2, alpha)

print("Доверительные интервалы для математического ожидания:")
print(f"Выборка 1: [{ci1[0]:.4f}, {ci1[1]:.4f}]")
print(f"  (среднее={ci1[2]:.4f}, SE={ci1[3]:.4f}, t({ci1[5]})={ci1[4]:.4f})")
print(f"Выборка 2: [{ci2[0]:.4f}, {ci2[1]:.4f}]")
print(f"  (среднее={ci2[2]:.4f}, SE={ci2[3]:.4f}, t({ci2[5]})={ci2[4]:.4f})")
print()

# 3. Доверительный интервал для дисперсии
# Используем хи-квадрат распределение
def var_ci(sample, alpha=0.05):
    n = len(sample)
    var = np.var(sample, ddof=1)
    df = n - 1
    chi2_low = stats.chi2.ppf(alpha/2, df)
    chi2_high = stats.chi2.ppf(1 - alpha/2, df)
    low = (df * var) / chi2_high
    high = (df * var) / chi2_low
    return (low, high, var, df, chi2_low, chi2_high)

var_ci1 = var_ci(sample1, alpha)
var_ci2 = var_ci(sample2, alpha)

print("Доверительные интервалы для дисперсии:")
print(f"Выборка 1: [{var_ci1[0]:.4f}, {var_ci1[1]:.4f}]")
print(f"  (дисперсия={var_ci1[2]:.4f}, df={var_ci1[3]}, χ²(ниж)={var_ci1[4]:.4f}, χ²(верх)={var_ci1[5]:.4f})")
print(f"  => для станд. откл.: [{np.sqrt(var_ci1[0]):.4f}, {np.sqrt(var_ci1[1]):.4f}]")
print(f"Выборка 2: [{var_ci2[0]:.4f}, {var_ci2[1]:.4f}]")
print(f"  (дисперсия={var_ci2[2]:.4f}, df={var_ci2[3]}, χ²(ниж)={var_ci2[4]:.4f}, χ²(верх)={var_ci2[5]:.4f})")
print(f"  => для станд. откл.: [{np.sqrt(var_ci2[0]):.4f}, {np.sqrt(var_ci2[1]):.4f}]")
print()

# 4. Проверка гипотезы о равенстве дисперсий (F-тест Фишера)
# Двусторонний критерий
f_stat = var1 / var2  # отношение дисперсий
df1, df2 = n1 - 1, n2 - 1
# Критические границы
f_crit_low = stats.f.ppf(alpha/2, df1, df2)
f_crit_high = stats.f.ppf(1 - alpha/2, df1, df2)
p_value = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))

print("F-тест на равенство дисперсий:")
print(f"F-статистика = {f_stat:.4f}")
print(f"Критические границы: [{f_crit_low:.4f}, {f_crit_high:.4f}]")
print(f"p-value = {p_value:.4f}")
if f_crit_low <= f_stat <= f_crit_high:
    print("Нулевая гипотеза о равенстве дисперсий НЕ отвергается (на уровне 0.05)")
else:
    print("Нулевая гипотеза отвергается")
print()

# Дополнительно: доверительный интервал для отношения дисперсий
ratio_low = f_stat / f_crit_high
ratio_high = f_stat / f_crit_low
print(f"95% доверительный интервал для отношения дисперсий: [{ratio_low:.4f}, {ratio_high:.4f}]")