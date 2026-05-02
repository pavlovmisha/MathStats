import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

#np.random.seed(42)
alpha = 0.05

# 1. Нормальная выборка n=100
n = 100
sample_norm = np.random.normal(loc=0, scale=1, size=n)

# 2. Оценка параметров ММП
mu_hat = np.mean(sample_norm)
sigma_hat = np.std(sample_norm, ddof=0)   # MLE для σ
print(f"Оценки параметров: μ̂ = {mu_hat:.4f}, σ̂ = {sigma_hat:.4f}")

# 3. Критерий χ² для нормальности с оценёнными параметрами
k = 6
quantiles = np.linspace(0, 1, k+1)
bins = stats.norm.ppf(quantiles, loc=mu_hat, scale=sigma_hat)
bins[0] = -np.inf
bins[-1] = np.inf

observed, _ = np.histogram(sample_norm, bins=bins)
expected = np.ones(k) * n / k

chi2_stat, p_value = stats.chisquare(f_obs=observed, f_exp=expected, ddof=0)
df = k - 1 - 2   # два оценённых параметра
critical_value = stats.chi2.ppf(1 - alpha, df)
p_value_adjusted = 1 - stats.chi2.cdf(chi2_stat, df)

print(f"\nКритерий χ² для нормальности (своя выборка):")
print(f"Число интервалов: {k}")
print(f"Статистика χ² = {chi2_stat:.4f}")
print(f"Степени свободы = {df}")
print(f"Критическое значение (α=0.05) = {critical_value:.4f}")
print(f"Скорректированное p-value = {p_value_adjusted:.4f}")
if chi2_stat > critical_value:
    print("Результат: H0 отвергается на уровне 0.05")
else:
    print("Результат: H0 не отвергается")

# 4. Исследование чувствительности (n=20) для равномерного и Лапласа
n_small = 20
sample_uniform = np.random.uniform(low=0, high=1, size=n_small)
sample_laplace = np.random.laplace(loc=0, scale=1, size=n_small)

def chi2_normality_test(data, alpha=0.05, k=4):
    n = len(data)
    mu = np.mean(data)
    sigma = np.std(data, ddof=0)
    bins = np.linspace(0, 1, k+1)
    bins = stats.norm.ppf(bins, loc=mu, scale=sigma)
    bins[0] = -np.inf
    bins[-1] = np.inf
    obs, _ = np.histogram(data, bins=bins)
    exp = np.ones(k) * n / k
    chi2 = np.sum((obs - exp)**2 / exp)
    df = k - 1 - 2
    if df <= 0:
        raise ValueError("Слишком мало интервалов")
    p_val = 1 - stats.chi2.cdf(chi2, df)
    crit = stats.chi2.ppf(1 - alpha, df)
    reject = chi2 > crit
    return chi2, p_val, crit, df, reject

k_small = 4  # ожидаемые частоты = 5

# Тест для равномерной выборки
chi2_u, p_u, crit_u, df_u, rej_u = chi2_normality_test(sample_uniform, alpha, k_small)
print("\n--- Равномерное распределение (n=20) ---")
print(f"Статистика χ² = {chi2_u:.4f}, df = {df_u}, p = {p_u:.4f}")
print(f"Критическое значение: {crit_u:.4f}, H0 отвергается: {rej_u}")

# Тест для выборки Лапласа
chi2_l, p_l, crit_l, df_l, rej_l = chi2_normality_test(sample_laplace, alpha, k_small)
print("\n--- Распределение Лапласа (n=20) ---")
print(f"Статистика χ² = {chi2_l:.4f}, df = {df_l}, p = {p_l:.4f}")
print(f"Критическое значение: {crit_l:.4f}, H0 отвергается: {rej_l}")

# Графики на отдельных картинках
# 1) Нормальная выборка
plt.figure(figsize=(6,4))
plt.hist(sample_norm, bins='auto', density=True, alpha=0.6, label='N(0,1) n=100')
x = np.linspace(min(sample_norm), max(sample_norm), 100)
plt.plot(x, stats.norm.pdf(x, mu_hat, sigma_hat), 'r-', label='Теоретическая зависимость')
plt.title('Нормальная выборка, n=100')
plt.legend()
plt.tight_layout()
plt.show()

# 2) Равномерная выборка
plt.figure(figsize=(6,4))
plt.hist(sample_uniform, bins='auto', density=True, alpha=0.6, label='U(0,1) n=20')
plt.title('Равномерная выборка, n=20')
plt.legend()
plt.tight_layout()
plt.show()

# 3) Выборка Лапласа
plt.figure(figsize=(6,4))
plt.hist(sample_laplace, bins='auto', density=True, alpha=0.6, label='Laplace(0,1) n=20')
plt.title('Лаплас, n=20')
plt.legend()
plt.tight_layout()
plt.show()