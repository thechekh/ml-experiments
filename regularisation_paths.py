"""L1, L2 and elastic net, watched on the weights.

Article: https://chekh.dev/writing/l1-l2-and-elastic-net-watched-on-the-weights/
Run:     uv run python regularisation_paths.py                 (under a minute)
         uv run python regularisation_paths.py --charts-only   (redraw from results/)

The diabetes dataset — 442 patients, 10 measurements, and a score for how far the
disease progressed a year later — plus 20 extra features of pure noise. Ridge (L2),
lasso (L1) and elastic net are fitted along a range of penalty strengths, to watch
which weights survive and how well each predicts held-out patients.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import (
    ElasticNet,
    ElasticNetCV,
    Lasso,
    LassoCV,
    LinearRegression,
    Ridge,
    RidgeCV,
    enet_path,
    lasso_path,
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from _common import ACCENT, INK, INK_2, MUTED, plain_numbers, save, versions

SLUG = "l1-l2-and-elastic-net-watched-on-the-weights"
NOISE = 20
L1_RATIOS = [0.2, 0.5, 0.8, 0.95]  # elastic net's mix, from mostly L2 to mostly L1
LASSO_ALPHAS = np.logspace(-1.5, 2, 60)
RIDGE_ALPHAS = np.logspace(-1, 5, 60)
CV = KFold(5, shuffle=True, random_state=0)


def data():
    """Diabetes features plus noise, standardised, split 75/25."""
    bunch = load_diabetes()
    rng = np.random.default_rng(0)
    X = np.hstack([bunch.data, rng.normal(size=(len(bunch.data), NOISE))])
    names = list(bunch.feature_names) + [f"noise{i + 1}" for i in range(NOISE)]
    X_train, X_test, y_train, y_test = train_test_split(X, bunch.target, test_size=0.25, random_state=0)
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test, names


def compute() -> dict:
    X_train, X_test, y_train, y_test = data()[:4]
    names = data()[4]
    real = [i for i, n in enumerate(names) if not n.startswith("noise")]
    noise = [i for i, n in enumerate(names) if n.startswith("noise")]
    y_centred = y_train - y_train.mean()  # the path functions fit no intercept

    # The penalty each method would choose by cross-validation on the training rows
    # (elastic net chooses its L1/L2 mix the same way), then how each does on the
    # held-out patients.
    # Elastic net twice: at an even 50/50 mix (the paths and error charts use this),
    # and with the mix chosen by cross-validation like everything else.
    enet_cv = ElasticNetCV(l1_ratio=L1_RATIOS, alphas=LASSO_ALPHAS, cv=CV, random_state=0).fit(X_train, y_train)
    l1_ratio = float(enet_cv.l1_ratio_)
    chosen = {
        "ridge": RidgeCV(alphas=RIDGE_ALPHAS, cv=CV).fit(X_train, y_train).alpha_,
        "lasso": LassoCV(alphas=LASSO_ALPHAS, cv=CV, random_state=0).fit(X_train, y_train).alpha_,
        "elastic net": ElasticNetCV(l1_ratio=0.5, alphas=LASSO_ALPHAS, cv=CV, random_state=0).fit(X_train, y_train).alpha_,
        "elastic net, CV mix": enet_cv.alpha_,
    }
    print(f"elastic net mix chosen by CV: {l1_ratio:g} L1, {1 - l1_ratio:g} L2")

    # Coefficient paths: every weight at every penalty strength.
    paths = {
        "ridge": np.array([Ridge(alpha=a).fit(X_train, y_train).coef_ for a in RIDGE_ALPHAS]).T,
        "lasso": lasso_path(X_train, y_centred, alphas=LASSO_ALPHAS)[1],
        "elastic net": enet_path(X_train, y_centred, l1_ratio=0.5, alphas=LASSO_ALPHAS)[1],
    }
    alphas = {"ridge": RIDGE_ALPHAS, "lasso": LASSO_ALPHAS, "elastic net": LASSO_ALPHAS}

    models = {
        "no penalty": LinearRegression(),
        "ridge": Ridge(alpha=chosen["ridge"]),
        "lasso": Lasso(alpha=chosen["lasso"]),
        "elastic net": ElasticNet(alpha=chosen["elastic net"], l1_ratio=0.5),
        "elastic net, CV mix": ElasticNet(alpha=chosen["elastic net, CV mix"], l1_ratio=l1_ratio),
    }
    summary = {}
    for name, model in models.items():
        coef = model.fit(X_train, y_train).coef_
        kept = np.abs(coef) > 1e-6
        pred = model.predict(X_test)
        summary[name] = {
            "alpha": float(chosen.get(name, 0)),
            "real_kept": int(kept[real].sum()),
            "noise_kept": int(kept[noise].sum()),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "r2": float(r2_score(y_test, pred)),
            "coef": {n: float(c) for n, c in zip(names, coef)},
        }
        print(f"{name:20s} alpha {summary[name]['alpha']:8.3f}  real kept {summary[name]['real_kept']:2d}/10  "
              f"noise kept {summary[name]['noise_kept']:2d}/20  test RMSE {summary[name]['rmse']:.1f}  R2 {summary[name]['r2']:.3f}")

    # Cross-validated error along each path, for the error chart.
    cv_rmse = {}
    for name, alpha_grid in alphas.items():
        make = {
            "ridge": lambda a: Ridge(alpha=a),
            "lasso": lambda a: Lasso(alpha=a),
            "elastic net": lambda a: ElasticNet(alpha=a, l1_ratio=0.5),
        }[name]
        scores = []
        for a in alpha_grid:
            fold = [np.sqrt(mean_squared_error(y_train[te], make(a).fit(X_train[tr], y_train[tr]).predict(X_train[te]))) for tr, te in CV.split(X_train)]
            scores.append(float(np.mean(fold)))
        cv_rmse[name] = scores

    # The correlated pair: total cholesterol (s1) and LDL (s2), r ≈ 0.9 in this data.
    s1, s2 = names.index("s1"), names.index("s2")
    corr = float(np.corrcoef(X_train[:, s1], X_train[:, s2])[0, 1])
    print(f"corr(s1, s2) = {corr:.3f}; weights at chosen alpha: "
          + "; ".join(f"{m}: s1 {summary[m]['coef']['s1']:+.1f}, s2 {summary[m]['coef']['s2']:+.1f}" for m in summary))

    return {
        "versions": versions(),
        "names": names,
        "l1_ratio": l1_ratio,
        "alphas": {k: v.tolist() for k, v in alphas.items()},
        "paths": {k: v.tolist() for k, v in paths.items()},
        "cv_rmse": cv_rmse,
        "summary": summary,
        "y_train_std": float(y_train.std()),
        "corr_s1_s2": corr,
        "rows": {"train": int(len(y_train)), "test": int(len(y_test))},
    }


def draw(results: dict) -> None:
    names = results["names"]
    real = [i for i, n in enumerate(names) if not n.startswith("noise")]
    noise = [i for i, n in enumerate(names) if n.startswith("noise")]
    methods = ["ridge", "lasso", "elastic net"]
    labels = {"ridge": "Ridge (L2)", "lasso": "Lasso (L1)", "elastic net": "Elastic net (50% L1, 50% L2)"}

    # Chart 1 · why L1 makes zeros: the loss contours meet the diamond at a corner.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7))
    w_star = np.array([1.4, 0.35])  # the unpenalised best weights
    A = np.array([[1.0, 0.25], [0.25, 1.3]])  # the shape of the loss bowl
    loss = lambda w: float((w - w_star) @ A @ (w - w_star))  # noqa: E731
    g1, g2 = np.meshgrid(np.linspace(-0.6, 2.0, 300), np.linspace(-0.8, 1.6, 300))
    L = np.array([[loss(np.array([a, b])) for a in g1[0]] for b in g2[:, 0]])
    t = 0.75
    angles = np.linspace(0, 2 * np.pi, 4000)
    circle = np.column_stack([t * np.cos(angles), t * np.sin(angles)])
    diamond = np.array([[c, s] for c, s in zip(np.cos(angles), np.sin(angles))])
    diamond = t * diamond / np.abs(diamond).sum(axis=1, keepdims=True)
    for ax, region, title in zip(axes, [circle, diamond], ["L2: the circle", "L1: the diamond"]):
        ax.contour(g1, g2, L, levels=[0.05, 0.2, 0.45, 0.8, 1.25], colors=[MUTED], linewidths=0.9)
        ax.plot(region[:, 0], region[:, 1], color=INK, linewidth=1.4)
        best = region[np.argmin([loss(w) for w in region])]
        ax.plot(*w_star, "o", color=INK_2, markersize=6, markeredgecolor="white", markeredgewidth=1)
        ax.annotate("no penalty", w_star, xytext=(8, 6), textcoords="offset points", fontsize=9, color=INK_2)
        ax.plot(*best, "o", color=ACCENT, markersize=8, markeredgecolor="white", markeredgewidth=1.5)
        ax.annotate(rf"$w_1$ = {best[0]:.2f}, $w_2$ = {best[1]:.2f}", best, xytext=(10, -14), textcoords="offset points", fontsize=9.5, color=ACCENT)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_xlim(-0.6, 2.0)
        ax.set_ylim(-0.8, 1.6)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.spines[["left", "bottom"]].set_visible(False)
        ax.set_xlabel(r"weight $w_1$", color=INK_2, fontsize=9.5)
        ax.set_title(title, fontsize=10.5)
    axes[0].set_ylabel(r"weight $w_2$", color=INK_2, fontsize=9.5)
    fig.tight_layout()
    save(fig, SLUG, "constraints")

    # Chart 2 · the paths: every weight against the penalty strength.
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.4), sharey=True)
    for ax, method in zip(axes, methods):
        alphas, path = np.array(results["alphas"][method]), np.array(results["paths"][method])
        for i in noise:
            ax.plot(alphas, path[i], color=MUTED, linewidth=1)
        for i in real:
            ax.plot(alphas, path[i], color=INK, linewidth=1.2)
        chosen = results["summary"][method]["alpha"]
        ax.axvline(chosen, color=ACCENT, linewidth=1.2)
        ax.set_xscale("log")
        plain_numbers(ax.xaxis)
        ax.set_title(labels[method], fontsize=10.5)
        ax.set_xlabel(r"penalty strength $\alpha$")
    axes[0].set_ylabel("weight")
    axes[1].annotate("real features", (0.03, 0.96), xycoords="axes fraction", fontsize=9, color=INK, va="top")
    axes[1].annotate("noise features", (0.03, 0.88), xycoords="axes fraction", fontsize=9, color=INK_2, va="top")
    axes[2].annotate("chosen by CV", (results["summary"]["elastic net"]["alpha"], 0.97), xycoords=("data", "axes fraction"), xytext=(4, 0), textcoords="offset points", fontsize=9, color=ACCENT, va="top")
    fig.tight_layout()
    save(fig, SLUG, "paths")

    # Chart 3 · held-out error along each path: too little penalty, then too much.
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0), sharey=True)
    for ax, method in zip(axes, methods):
        alphas, rmse = np.array(results["alphas"][method]), np.array(results["cv_rmse"][method])
        ax.plot(alphas, rmse, color=ACCENT, linewidth=2)
        best = int(np.argmin(rmse))
        ax.plot(alphas[best], rmse[best], "o", color=ACCENT, markersize=7, markeredgecolor="white", markeredgewidth=1.5)
        ax.annotate(rf"{rmse[best]:.1f} at $\alpha$ = {alphas[best]:.3g}", (alphas[best], rmse[best]), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9, color=INK)
        ax.set_xscale("log")
        plain_numbers(ax.xaxis)
        ax.set_title(labels[method], fontsize=10.5)
        ax.set_xlabel(r"penalty strength $\alpha$")
    axes[0].set_ylabel("cross-validated RMSE")
    axes[0].set_ylim(50, 80)
    fig.tight_layout()
    save(fig, SLUG, "error")


if __name__ == "__main__":
    path = Path(__file__).parent / "results" / f"{SLUG}.json"
    if "--charts-only" in sys.argv:
        results = json.loads(path.read_text())
    else:
        results = compute()
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(results, indent=2))
        print(f"  wrote {path.name}")
    draw(results)
