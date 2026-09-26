"""Bias and variance, explained with one dataset.

Article: https://chekh.dev/writing/bias-and-variance-drawn-from-one-dataset/
Run:     uv run python bias_variance.py   (about a minute; downloads California housing once)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, learning_curve, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from _common import ACCENT, INK, INK_2, MUTED, plain_numbers, save, versions

SLUG = "bias-and-variance-drawn-from-one-dataset"
X, y = fetch_california_housing(return_X_y=True)  # target: median house value, $100,000s
results: dict = {"versions": versions(), "rows": len(y)}


# 0 · The classic picture: four archers. Bias is where the shots centre, variance is
#     how far they scatter. Drawn, not measured — everything after this is measured.
fig, axes = plt.subplots(2, 2, figsize=(5.2, 5.4))
shots_rng = np.random.default_rng(1)
for ax, (row, col) in zip(axes.flat, [(0, 0), (0, 1), (1, 0), (1, 1)]):
    for radius in (1.0, 0.66, 0.33):
        ax.add_patch(plt.Circle((0, 0), radius, fill=False, color=MUTED, linewidth=1))
    ax.add_patch(plt.Circle((0, 0), 0.07, color=MUTED, linewidth=0))
    centre = (0.5, 0.4) if col else (0, 0)  # high bias: centred away from the bullseye
    spread = 0.3 if row else 0.07  # high variance: scattered
    shots = shots_rng.normal(centre, spread, size=(12, 2))
    ax.plot(shots[:, 0], shots[:, 1], "o", color=ACCENT, markersize=5.5, markeredgecolor="white", markeredgewidth=1)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{'high' if col else 'low'} bias, {'high' if row else 'low'} variance", fontsize=10, loc="center")
fig.tight_layout()
save(fig, SLUG, "targets")


# 1 · Decomposition: 30 trees per depth, each on a bootstrap sample of 5,000 districts,
#     all scored on the same 2,000 held-out districts.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2_000, random_state=0)
rng = np.random.default_rng(0)
depths = list(range(1, 21))
decomposition = []
for depth in depths:
    predictions = np.array(
        [
            DecisionTreeRegressor(max_depth=depth, random_state=b)
            .fit(*(a[idx] for a in (X_train, y_train)))
            .predict(X_test)
            for b, idx in enumerate(rng.choice(len(X_train), (30, 5_000), replace=True))
        ]
    )
    mse = float(((predictions - y_test) ** 2).mean())
    variance = float(predictions.var(axis=0).mean())  # how much the 30 trees disagree
    bias_noise = float(((predictions.mean(axis=0) - y_test) ** 2).mean())  # the rest
    decomposition.append({"depth": depth, "mse": mse, "variance": variance, "bias2_noise": bias_noise})
results["decomposition"] = decomposition
best = min(decomposition, key=lambda r: r["mse"])
print(f"best depth {best['depth']}: MSE {best['mse']:.3f} (RMSE ${np.sqrt(best['mse']) * 100_000:,.0f})")

fig, ax = plt.subplots()
series = [("mse", "total error", ACCENT, 2.2), ("bias2_noise", "bias² + noise", INK, 1.5), ("variance", "variance", INK_2, 1.5)]
for key, label, color, width in series:
    values = [r[key] for r in decomposition]
    ax.plot(depths, values, color=color, linewidth=width)
    ax.annotate(label, (depths[-1], values[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=color, fontsize=10)
ax.plot(best["depth"], best["mse"], "o", color=ACCENT, markersize=7, markeredgecolor="white", markeredgewidth=1.5)
ax.annotate(f"best: depth {best['depth']}", (best["depth"], best["mse"]), xytext=(6, -20), textcoords="offset points", ha="left", color=INK, fontsize=10)
ax.set_xlim(0.5, 20.5)
ax.set_ylim(0, 1.0)
ax.set_xticks([1, 4, 8, 12, 16, 20])
ax.set_xlabel("tree depth")
ax.set_ylabel("squared error")
ax.set_title("Error on 2,000 held-out districts, by tree depth")
save(fig, SLUG, "decomposition")


# 2 · Learning curves for a tree that is too simple, about right, and unlimited.
models = [(2, "depth 2: too simple"), (8, "depth 8: about right"), (None, "no limit: too flexible")]
fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.0), sharey=True)
results["learning_curves"] = []
for ax, (depth, title) in zip(axes, models):
    sizes, train, valid = learning_curve(
        DecisionTreeRegressor(max_depth=depth, random_state=0),
        X, y, train_sizes=np.linspace(0.05, 1.0, 8), cv=KFold(5, shuffle=True, random_state=0),
        # The rows are stored in geographic order; without shuffling, every small
        # training set would come from one corner of California.
        shuffle=True, random_state=0, scoring="r2", n_jobs=-1,
    )
    train, valid = train.mean(axis=1), valid.mean(axis=1)
    results["learning_curves"].append({"depth": depth, "sizes": sizes.tolist(), "train_r2": train.tolist(), "valid_r2": valid.tolist()})
    ax.plot(sizes, train, color=INK_2, linewidth=1.5)
    ax.plot(sizes, valid, color=ACCENT, linewidth=2.2)
    ax.set_title(title, fontsize=10.5)
    ax.set_xticks([0, 8_000, 16_000], ["0", "8k", "16k"])
    if depth == 2:
        ax.annotate("training", (sizes[-1], train[-1]), xytext=(-4, 8), textcoords="offset points", ha="right", color=INK_2, fontsize=9.5)
        ax.annotate("validation", (sizes[-1], valid[-1]), xytext=(-4, -14), textcoords="offset points", ha="right", color=ACCENT, fontsize=9.5)
    print(f"learning curve depth={depth}: train R² {train[-1]:.3f}, validation R² {valid[-1]:.3f} at {sizes[-1]:,} districts")
axes[0].set_ylim(0, 1.05)
axes[0].set_ylabel("R²")
axes[1].set_xlabel("districts in the training set")
save(fig, SLUG, "learning-curves")


# 3 · Double descent: least squares on random ReLU features of 200 districts. Past as many
#     features as points, lstsq returns the minimum-norm fit.
Xs = StandardScaler().fit_transform(X)
n = 200
widths = [10, 25, 50, 100, 150, 180, 195, 200, 205, 220, 250, 300, 400, 600, 1_000, 2_000, 5_000, 10_000]
curve = {p: [] for p in widths}
for rep in range(20):
    r = np.random.default_rng(rep)
    idx = r.permutation(len(Xs))
    train_idx, test_idx = idx[:n], idx[n : n + 2_000]
    for p in widths:
        W = r.normal(size=(Xs.shape[1], p)) / np.sqrt(Xs.shape[1])
        F_train, F_test = np.maximum(Xs[train_idx] @ W, 0), np.maximum(Xs[test_idx] @ W, 0)
        coef, *_ = np.linalg.lstsq(F_train, y[train_idx], rcond=None)
        curve[p].append(float(((F_test @ coef - y[test_idx]) ** 2).mean()))
median = {p: float(np.median(v)) for p, v in curve.items()}
results["double_descent"] = {"train_points": n, "median_test_mse": median}
first_min = min((p for p in widths if p < n), key=median.get)
print(f"double descent: best under n at {first_min} features (MSE {median[first_min]:.2f}); "
      f"peak {median[n]:.0f} at {n}; {median[widths[-1]]:.2f} at {widths[-1]:,}")

fig, ax = plt.subplots()
ax.plot(widths, [median[p] for p in widths], color=ACCENT, linewidth=2.2, marker="o", markersize=4.5, markeredgecolor="white", markeredgewidth=1)
ax.axvline(n, color=MUTED, linewidth=1)
ax.annotate("features = training districts", (n, median[n]), xytext=(8, -2), textcoords="offset points", color=INK_2, fontsize=9.5, va="top")
ax.annotate(f"best: {first_min} features, {median[first_min]:.2f}", (first_min, median[first_min]), xytext=(0, -16), textcoords="offset points", ha="left", color=INK, fontsize=9.5)
ax.annotate(f"{widths[-1]:,} features: {median[widths[-1]]:.2f}", (widths[-1], median[widths[-1]]), xytext=(0, 12), textcoords="offset points", ha="right", color=INK, fontsize=9.5)
ax.set_xscale("log")
ax.set_yscale("log")
plain_numbers(ax.xaxis)
plain_numbers(ax.yaxis)
ax.set_xlabel("random features (log scale)")
ax.set_ylabel("test error (log scale)")
ax.set_title("Test error of a least-squares fit on 200 districts")
save(fig, SLUG, "double-descent")

out = Path(__file__).parent / "results" / f"{SLUG}.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(results, indent=2))
print(f"  wrote {out.name}")
