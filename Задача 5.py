import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, spearmanr, pearsonr
from matplotlib.patches import Ellipse

# ---------------------------
# Вспомогательные функции
# ---------------------------

def generate_normal(n, rho, mean=(0,0), var=(1,1)):
    """Генерация выборки из двумерного нормального распределения."""
    cov = [[var[0], rho * np.sqrt(var[0]*var[1])],
           [rho * np.sqrt(var[0]*var[1]), var[1]]]
    return np.random.multivariate_normal(mean, cov, size=n)

def generate_mixture(n, p=0.1):
    """
    Смесь: 0.9 * N(0,0,1,1,0.9) + 0.1 * N(0,0,10,10,-0.9).
    """
    n1 = np.random.binomial(n, 0.9)
    n2 = n - n1
    sample1 = generate_normal(n1, 0.9, mean=(0,0), var=(1,1))
    sample2 = generate_normal(n2, -0.9, mean=(0,0), var=(10,10))
    sample = np.vstack([sample1, sample2])
    np.random.shuffle(sample)
    return sample[:,0], sample[:,1]

def quadrant_corr(x, y):
    """Квадрантный коэффициент корреляции."""
    med_x = np.median(x)
    med_y = np.median(y)
    signs = np.sign((x - med_x) * (y - med_y))
    return np.mean(signs)

def compute_correlations(x, y):
    """Возвращает словарь с тремя коэффициентами корреляции."""
    r_pearson, _ = pearsonr(x, y)
    r_spearman, _ = spearmanr(x, y)
    r_quad = quadrant_corr(x, y)
    return {'Pearson': r_pearson, 'Spearman': r_spearman, 'Quadrant': r_quad}

def plot_ellipse(x, y, ax, n_std=2.0, **kwargs):
    """
    Построение эллипса рассеяния по ковариационной матрице.
    n_std — радиус в стандартных отклонениях (2.0 ≈ 95% для нормального).
    """
    cov = np.cov(x, y)
    mean = np.mean(x), np.mean(y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)

# ---------------------------
# Параметры эксперимента
# ---------------------------
n_values = [20, 60, 100]
rhos_normal = [0.0, 0.5, 0.9]
n_sim = 1000

# Истинные корреляции
true_rho = {
    'normal_0.0': 0.0,
    'normal_0.5': 0.5,
    'normal_0.9': 0.9,
    'mixture': 0.0
}

# Сохранение результатов
results = []

# ---------------------------
# Цикл моделирования
# ---------------------------
np.random.seed(42)  # для воспроизводимости

# 1. Нормальное распределение с разными ρ
for rho in rhos_normal:
    for n in n_values:
        print(f"Processing Normal, ρ={rho}, n={n}")
        pearson_vals = []
        spearman_vals = []
        quadrant_vals = []
        for _ in range(n_sim):
            x, y = generate_normal(n, rho).T
            corrs = compute_correlations(x, y)
            pearson_vals.append(corrs['Pearson'])
            spearman_vals.append(corrs['Spearman'])
            quadrant_vals.append(corrs['Quadrant'])
        results.append({
            'Distribution': f'Normal ρ={rho}',
            'n': n,
            'True ρ': true_rho[f'normal_{rho}'],
            'Pearson mean': np.mean(pearson_vals),
            'Pearson var': np.var(pearson_vals),
            'Spearman mean': np.mean(spearman_vals),
            'Spearman var': np.var(spearman_vals),
            'Quadrant mean': np.mean(quadrant_vals),
            'Quadrant var': np.var(quadrant_vals)
        })

# 2. Смесь
for n in n_values:
    print(f"Processing Mixture, n={n}")
    pearson_vals = []
    spearman_vals = []
    quadrant_vals = []
    for _ in range(n_sim):
        x, y = generate_mixture(n)
        corrs = compute_correlations(x, y)
        pearson_vals.append(corrs['Pearson'])
        spearman_vals.append(corrs['Spearman'])
        quadrant_vals.append(corrs['Quadrant'])
    results.append({
        'Distribution': 'Mixture',
        'n': n,
        'True ρ': true_rho['mixture'],
        'Pearson mean': np.mean(pearson_vals),
        'Pearson var': np.var(pearson_vals),
        'Spearman mean': np.mean(spearman_vals),
        'Spearman var': np.var(spearman_vals),
        'Quadrant mean': np.mean(quadrant_vals),
        'Quadrant var': np.var(quadrant_vals)
    })

# ---------------------------
# Таблицы результатов
# ---------------------------
df = pd.DataFrame(results)
print("\nСредние значения коэффициентов корреляции (1000 повторений):")
print(df[['Distribution', 'n', 'True ρ', 'Pearson mean', 'Spearman mean', 'Quadrant mean']].round(4))

print("\nДисперсии коэффициентов корреляции (1000 повторений):")
print(df[['Distribution', 'n', 'Pearson var', 'Spearman var', 'Quadrant var']].round(6))

# Сохраним для наглядности в виде LaTeX или CSV (при необходимости)
# df.to_csv('correlation_results.csv', index=False)

# ---------------------------
# Диаграммы рассеяния с эллипсами (по 4 на фигуру)
# ---------------------------
# Для каждого n создаём отдельную фигуру 2x2
for n in n_values:
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f'Диаграммы рассеяния для n = {n} с 95% эллипсами', fontsize=14)
    axes = axes.flatten()

    # 1. Normal ρ = 0
    x, y = generate_normal(n, 0.0).T
    ax = axes[0]
    ax.scatter(x, y, alpha=0.7, edgecolors='k', linewidth=0.5)
    plot_ellipse(x, y, ax, edgecolor='red', fc='none', lw=2)
    ax.set_title('Normal ρ = 0')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)

    # 2. Normal ρ = 0.5
    x, y = generate_normal(n, 0.5).T
    ax = axes[1]
    ax.scatter(x, y, alpha=0.7, edgecolors='k', linewidth=0.5)
    plot_ellipse(x, y, ax, edgecolor='red', fc='none', lw=2)
    ax.set_title('Normal ρ = 0.5')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)

    # 3. Normal ρ = 0.9
    x, y = generate_normal(n, 0.9).T
    ax = axes[2]
    ax.scatter(x, y, alpha=0.7, edgecolors='k', linewidth=0.5)
    plot_ellipse(x, y, ax, edgecolor='red', fc='none', lw=2)
    ax.set_title('Normal ρ = 0.9')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)

    # 4. Mixture
    x, y = generate_mixture(n)
    ax = axes[3]
    ax.scatter(x, y, alpha=0.7, edgecolors='k', linewidth=0.5)
    plot_ellipse(x, y, ax, edgecolor='red', fc='none', lw=2)
    ax.set_title('Mixture')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()