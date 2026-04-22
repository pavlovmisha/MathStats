import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

np.random.seed(42)

# 1. Генерация x
x = np.arange(-1.8, 2.01, 0.2)
n = len(x)

# Истинные параметры
a_true, b_true = 2.0, 2.0

# 2. Генерация y без выбросов
eps = np.random.normal(0, 1, n)
y_clean = a_true + b_true * x + eps

# 3. Функция МНМ (через statsmodels QuantReg с tau=0.5)
def lad_fit(x, y):
    X = sm.add_constant(x)
    model = sm.QuantReg(y, X)
    res = model.fit(q=0.5)  # медианная регрессия = LAD
    return res.params[0], res.params[1]

# 4. Оценка на чистых данных
# МНК
a_mnk_clean, b_mnk_clean = np.polyfit(x, y_clean, 1)
# МНМ
a_mnm_clean, b_mnm_clean = lad_fit(x, y_clean)

# 5. Добавление выбросов
y_out = y_clean.copy()
y_out[0] += 10
y_out[-1] -= 10

# 6. Оценка на данных с выбросами
a_mnk_out, b_mnk_out = np.polyfit(x, y_out, 1)
a_mnm_out, b_mnm_out = lad_fit(x, y_out)

# 7. Расчёт относительных погрешностей
def rel_err(true, est):
    return np.abs(true - est) / np.abs(true) * 100

print("Без выбросов:")
print(f"МНК: a={a_mnk_clean:.3f}, b={b_mnk_clean:.3f}, err_a={rel_err(a_true, a_mnk_clean):.2f}%, err_b={rel_err(b_true, b_mnk_clean):.2f}%")
print(f"МНМ: a={a_mnm_clean:.3f}, b={b_mnm_clean:.3f}, err_a={rel_err(a_true, a_mnm_clean):.2f}%, err_b={rel_err(b_true, b_mnm_clean):.2f}%")

print("\nС выбросами:")
print(f"МНК: a={a_mnk_out:.3f}, b={b_mnk_out:.3f}, err_a={rel_err(a_true, a_mnk_out):.2f}%, err_b={rel_err(b_true, b_mnk_out):.2f}%")
print(f"МНМ: a={a_mnm_out:.3f}, b={b_mnm_out:.3f}, err_a={rel_err(a_true, a_mnm_out):.2f}%, err_b={rel_err(b_true, b_mnm_out):.2f}%")

# 8. Построение графиков (пример)
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(x, y_clean, label='Данные')
plt.plot(x, a_true + b_true*x, 'k--', label='Истинная')
plt.plot(x, a_mnk_clean + b_mnk_clean*x, 'r-', label='МНК')
plt.plot(x, a_mnm_clean + b_mnm_clean*x, 'b-', label='МНМ')
plt.title('Без выбросов')
plt.legend()

plt.subplot(1,2,2)
plt.scatter(x, y_out, label='Данные с выбросами')
plt.plot(x, a_true + b_true*x, 'k--', label='Истинная')
plt.plot(x, a_mnk_out + b_mnk_out*x, 'r-', label='МНК')
plt.plot(x, a_mnm_out + b_mnm_out*x, 'b-', label='МНМ')
plt.title('С выбросами')
plt.legend()

plt.tight_layout()
plt.show()