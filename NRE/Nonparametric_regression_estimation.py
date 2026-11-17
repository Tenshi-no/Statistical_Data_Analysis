import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- ЧАСТЬ 1 ----------
df1 = pd.read_csv("part1_data.csv")
df1.columns = df1.columns.str.strip()
err1 = pd.read_csv("part1_error.csv")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1.1 Данные + истинная функция + регрессия
ax = axes[0, 0]
ax.scatter(df1["x"], df1["y_noisy"], s=4, alpha=0.3, label="Данные (с шумом)", color="gray")
ax.plot(df1["x"], df1["y_true"], "g-", linewidth=1.5, label="Истинная функция")
ax.plot(df1["x"], df1["y_pred"], "r-", linewidth=2, label="Непараметрическая регрессия")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("Часть 1: восстановление регрессии")
ax.legend(); ax.grid(alpha=0.3)

# 1.2 Увеличенный фрагмент
ax = axes[0, 1]
mask = (df1["x"] > 20) & (df1["x"] < 30)
ax.scatter(df1.loc[mask, "x"], df1.loc[mask, "y_noisy"], s=8, alpha=0.5, color="gray", label="Данные")
ax.plot(df1.loc[mask, "x"], df1.loc[mask, "y_true"], "g-", linewidth=1.5, label="Истинная")
ax.plot(df1.loc[mask, "x"], df1.loc[mask, "y_pred"], "r-", linewidth=2, label="Регрессия")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("Часть 1: фрагмент x ∈ (20, 30)")
ax.legend(); ax.grid(alpha=0.3)

# 1.3 Зависимость MSE от c
ax = axes[1, 0]
ax.plot(err1["c"], err1["error"], "b-o", markersize=3)
best_idx = err1["error"].idxmin()
best_c = err1.loc[best_idx, "c"]
best_err = err1.loc[best_idx, "error"]
ax.axvline(best_c, color="r", linestyle="--", label=f"c* = {best_c:.2f}")
ax.scatter([best_c], [best_err], color="r", s=80, zorder=5)
ax.set_xlabel("Параметр размытости c")
ax.set_ylabel("MSE (скользящий экзамен)")
ax.set_title("Часть 1: выбор параметра c")
ax.legend(); ax.grid(alpha=0.3)
ax.set_yscale("log")  

# 1.4 Остатки / ошибка по точкам
ax = axes[1, 1]
residuals = df1["y_noisy"] - df1["y_pred"]
ax.scatter(df1["x"], residuals, s=4, alpha=0.4, color="purple")
ax.axhline(0, color="k", linestyle="--")
ax.set_xlabel("x"); ax.set_ylabel("y - ŷ")
ax.set_title("Часть 1: остатки")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("part1_plots.png", dpi=150)
plt.show()


# ---------- ЧАСТЬ 2 ----------
df2 = pd.read_csv("part2_data.csv")
err2 = pd.read_csv("part2_error.csv")

# Сортируем по x для корректного построения линии
df2_sorted = df2.sort_values("x").reset_index(drop=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 2.1 Данные + регрессия
ax = axes[0]
ax.scatter(df2["x"], df2["y_noisy"], s=40, color="steelblue", label="Данные (n=50)")
ax.plot(df2_sorted["x"], df2_sorted["y_pred"], "r-", linewidth=2, label="Непараметрическая регрессия")
ax.set_xlabel("Площадь, м²")
ax.set_ylabel("Цена, руб.")
ax.set_title("Часть 2: зависимость цены от площади")
ax.legend(); ax.grid(alpha=0.3)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

# 2.2 MSE от c
ax = axes[1]
ax.plot(err2["c"], err2["error"], "b-o", markersize=3)
best_idx2 = err2["error"].idxmin()
best_c2 = err2.loc[best_idx2, "c"]
best_err2 = err2.loc[best_idx2, "error"]
ax.axvline(best_c2, color="r", linestyle="--", label=f"c* = {best_c2:.2f}")
ax.scatter([best_c2], [best_err2], color="r", s=80, zorder=5)
ax.set_xlabel("Параметр размытости c")
ax.set_ylabel("MSE (скользящий экзамен)")
ax.set_title("Часть 2: выбор параметра c")
ax.legend(); ax.grid(alpha=0.3)
ax.set_yscale("log")

plt.tight_layout()
plt.savefig("part2_plots.png", dpi=150)
plt.show()
