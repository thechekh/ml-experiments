"""Six fixes for overfitting, tested side by side.

Article: https://chekh.dev/writing/six-fixes-for-overfitting-measured-side-by-side/
Run:     uv run python overfitting_fixes.py                 (about fifteen minutes on 12 cores)
         uv run python overfitting_fixes.py --charts-only   (redraw from results/, in seconds)

One overfit model — a 256-unit MLP on 300 handwritten digits — then each fix, with its
one setting chosen by 5-fold cross-validation on the training labels only. Twice: with
clean labels, and with 20% of training labels replaced by a wrong digit. The 600-image
test set always keeps its true labels.
"""

import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from _common import ACCENT, INK, INK_2, MUTED, plain_numbers, save, versions

SLUG = "six-fixes-for-overfitting-measured-side-by-side"
X, y = load_digits(return_X_y=True)
X = X / 16.0  # pixel intensities 0–16 → 0–1
TRAIN, TEST, NOISE, REPEATS = 300, 600, 0.20, 20
BASE = dict(hidden_layer_sizes=(256,), alpha=1e-6, max_iter=1_000)
ALPHAS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1, 3, 10, 30]


def data(seed: int, noise: float):
    """A clean test set, and a training pool whose labels are `noise` wrong."""
    X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=TEST, stratify=y, random_state=seed)
    rng = np.random.default_rng(seed)
    flip = rng.random(len(y_pool)) < noise
    y_seen = y_pool.copy()
    y_seen[flip] = (y_pool[flip] + rng.integers(1, 10, flip.sum())) % 10  # always a wrong digit
    order = rng.permutation(len(y_pool))
    return X_pool[order], y_seen[order], X_test, y_test


def search(estimator, grid: dict, seed: int):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return GridSearchCV(estimator, grid, cv=cv, n_jobs=1)


def one_repeat(seed: int, noise: float) -> dict:
    warnings.simplefilter("ignore", ConvergenceWarning)
    X_pool, y_pool, X_test, y_test = data(seed, noise)
    X_small, y_small = X_pool[:TRAIN], y_pool[:TRAIN]
    mlp = lambda **kw: MLPClassifier(**{**BASE, "random_state": seed, **kw})  # noqa: E731
    conditions = {
        "no fix": (mlp(), X_small, y_small),
        "more data (4×)": (mlp(), X_pool, y_pool),
        "L2 regularisation": (search(mlp(), {"alpha": [1e-4, 1e-3, 1e-2, 0.1, 0.3, 1, 3, 10]}, seed), X_small, y_small),
        "simpler model": (search(mlp(), {"hidden_layer_sizes": [(4,), (8,), (16,), (32,), (64,), (128,), (256,)]}, seed), X_small, y_small),
        "PCA": (search(Pipeline([("pca", PCA(random_state=seed)), ("mlp", mlp())]), {"pca__n_components": [5, 10, 15, 20, 30, 40]}, seed), X_small, y_small),
        "early stopping": (mlp(early_stopping=True, validation_fraction=0.2, n_iter_no_change=20), X_small, y_small),
    }
    out = {}
    for name, (model, X_fit, y_fit) in conditions.items():
        start = time.perf_counter()
        model.fit(X_fit, y_fit)
        seconds = time.perf_counter() - start
        out[name] = {
            "train": model.score(X_fit, y_fit),  # against the labels it was given
            "test": model.score(X_test, y_test),  # against the true labels
            "seconds": seconds,
            "chosen": {k: str(v) for k, v in getattr(model, "best_params_", {}).items()},
        }
    # Cross-validation's own job: see the gap without touching the test set.
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    out["no fix"]["cv"] = float(np.mean([mlp().fit(X_small[a], y_small[a]).score(X_small[b], y_small[b]) for a, b in cv.split(X_small, y_small)]))
    return out


def summarise(runs: list[dict]) -> dict:
    return {
        name: {
            "train": float(np.mean([r[name]["train"] for r in runs])),
            "test": float(np.mean([r[name]["test"] for r in runs])),
            "test_sd": float(np.std([r[name]["test"] for r in runs])),
            "seconds": float(np.mean([r[name]["seconds"] for r in runs])),
            "chosen": [r[name]["chosen"] for r in runs],
            **({"cv": float(np.mean([r[name]["cv"] for r in runs]))} if "cv" in runs[0][name] else {}),
        }
        for name in runs[0]
    }


def epochs(seed: int, n_epochs: int = 200):
    """Accuracy after every epoch, on the noisy training labels and on the true test labels."""
    warnings.simplefilter("ignore", ConvergenceWarning)
    X_pool, y_pool, X_test, y_test = data(seed, NOISE)
    model = MLPClassifier(**{**BASE, "random_state": seed})
    train, test = [], []
    for _ in range(n_epochs):
        model.partial_fit(X_pool[:TRAIN], y_pool[:TRAIN], classes=np.arange(10))
        train.append(model.score(X_pool[:TRAIN], y_pool[:TRAIN]))
        test.append(model.score(X_test, y_test))
    return train, test


def by_alpha(seed: int):
    """For each L2 penalty: what 5-fold CV on the noisy labels sees, and the true test accuracy."""
    warnings.simplefilter("ignore", ConvergenceWarning)
    X_pool, y_pool, X_test, y_test = data(seed, NOISE)
    Xs, ys = X_pool[:TRAIN], y_pool[:TRAIN]
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    cv_scores, test_scores = [], []
    for alpha in ALPHAS:
        make = lambda: MLPClassifier(**{**BASE, "alpha": alpha, "random_state": seed})  # noqa: E731
        cv_scores.append(np.mean([make().fit(Xs[a], ys[a]).score(Xs[b], ys[b]) for a, b in cv.split(Xs, ys)]))
        test_scores.append(make().fit(Xs, ys).score(X_test, y_test))
    return cv_scores, test_scores


def compute() -> dict:
    results: dict = {"versions": versions(), "train": TRAIN, "test": TEST, "noise": NOISE, "repeats": REPEATS}
    for label, noise in [("clean", 0.0), ("noisy", NOISE)]:
        runs = Parallel(n_jobs=-1)(delayed(one_repeat)(seed, noise) for seed in range(REPEATS))
        results[label] = summarise(runs)
        print(f"--- {label} labels ({REPEATS} repeats)")
        for name, r in results[label].items():
            extra = f"  cv {r['cv']:.3f}" if "cv" in r else ""
            print(f"{name:18s} train {r['train']:.3f}  test {r['test']:.3f} ± {r['test_sd']:.3f}  {r['seconds']:6.1f}s{extra}")

    curves = np.array(Parallel(n_jobs=-1)(delayed(epochs)(seed) for seed in range(10))) * 100  # (seed, 2, epoch)
    train_curve, test_curve = curves[:, 0].mean(axis=0), curves[:, 1].mean(axis=0)
    peak = int(np.argmax(test_curve))
    results["epochs"] = {"train": train_curve.tolist(), "test": test_curve.tolist(), "peak_epoch": peak + 1}
    print(f"early stopping: test peaks at epoch {peak + 1} ({test_curve[peak]:.1f}%), "
          f"ends at {test_curve[-1]:.1f}% while training reaches {train_curve[-1]:.1f}%")

    sweep = np.array(Parallel(n_jobs=-1)(delayed(by_alpha)(seed) for seed in range(10))) * 100
    cv_curve, truth_curve = sweep[:, 0].mean(axis=0), sweep[:, 1].mean(axis=0)
    results["alpha_sweep"] = {
        "alphas": ALPHAS, "cv": cv_curve.tolist(), "test": truth_curve.tolist(),
        "cv_pick": ALPHAS[int(np.argmax(cv_curve))], "test_best": ALPHAS[int(np.argmax(truth_curve))],
    }
    print(f"alpha sweep: CV picks {results['alpha_sweep']['cv_pick']}; best on truth {results['alpha_sweep']['test_best']}")
    return results


def draw(results: dict) -> None:
    # Chart 1 · each fix with noisy labels. A dot plot: the values sit in a narrow range,
    # and a bar would have to start at zero to be honest.
    noisy = results["noisy"]
    names = sorted(noisy, key=lambda n: noisy[n]["test"])
    top = max(r["test"] for r in noisy.values())
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for i, name in enumerate(names):
        r = noisy[name]
        mean, sd = r["test"] * 100, r["test_sd"] * 100
        color = ACCENT if top - r["test"] < 0.005 else INK_2  # within half a point of the best
        ax.plot([mean - sd, mean + sd], [i, i], color=MUTED, linewidth=2, solid_capstyle="round", zorder=1)
        ax.plot(mean, i, "o", color=color, markersize=8, markeredgecolor="white", markeredgewidth=1.5, zorder=2)
        ax.annotate(f"{mean:.1f}%", (mean + sd, i), xytext=(6, 0), textcoords="offset points", va="center", fontsize=10, color=INK)
    ax.axvline(noisy["no fix"]["test"] * 100, color=MUTED, linewidth=1, zorder=0)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlim(70, 95)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("test accuracy on true labels (%), mean ± 1 SD of 20 repeats")
    ax.set_title("With 20% wrong labels: accuracy after each fix")
    save(fig, SLUG, "fixes")

    # Chart 2 · early stopping, epoch by epoch
    e = results["epochs"]
    train_curve, test_curve, peak = np.array(e["train"]), np.array(e["test"]), e["peak_epoch"]
    x = np.arange(1, len(test_curve) + 1)
    fig, ax = plt.subplots()
    ax.plot(x, train_curve, color=INK_2, linewidth=1.5)
    ax.plot(x, test_curve, color=ACCENT, linewidth=2.2)
    ax.plot(peak, test_curve[peak - 1], "o", color=ACCENT, markersize=7, markeredgecolor="white", markeredgewidth=1.5)
    ax.annotate(f"best: epoch {peak}", (peak, test_curve[peak - 1]), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=10, color=INK)
    ax.annotate("training labels (20% wrong)", (x[-1], train_curve[-1]), xytext=(0, 8), textcoords="offset points", ha="right", fontsize=9.5, color=INK_2)
    ax.annotate("true test labels", (x[-1], test_curve[-1]), xytext=(0, -14), textcoords="offset points", ha="right", fontsize=9.5, color=ACCENT)
    ax.set_ylim(40, 102)
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("The model learns the digits, then memorises the noise")
    save(fig, SLUG, "early-stopping")

    # Chart 3 · the L2 penalty: what cross-validation sees, against the truth
    s = results["alpha_sweep"]
    alphas, cv_curve, truth_curve = s["alphas"], s["cv"], s["test"]
    best = alphas.index(s["test_best"])
    fig, ax = plt.subplots()
    ax.plot(alphas, cv_curve, color=INK_2, linewidth=1.5, marker="o", markersize=4, markeredgecolor="white", markeredgewidth=1)
    ax.plot(alphas, truth_curve, color=ACCENT, linewidth=2.2, marker="o", markersize=4.5, markeredgecolor="white", markeredgewidth=1)
    ax.axvline(s["cv_pick"], color=MUTED, linewidth=1)
    ax.annotate(f"cross-validation picks {s['cv_pick']:g}", (s["cv_pick"], 97), xytext=(-6, 0), textcoords="offset points", ha="right", va="top", fontsize=9.5, color=INK)
    ax.annotate(f"best on true labels: {s['test_best']:g}", (s["test_best"], truth_curve[best]), xytext=(8, 4), textcoords="offset points", fontsize=9.5, color=ACCENT)
    ax.annotate("5-fold CV on the noisy labels", (alphas[0], cv_curve[0]), xytext=(0, -14), textcoords="offset points", fontsize=9.5, color=INK_2)
    ax.annotate("true test labels", (alphas[0], truth_curve[0]), xytext=(0, 8), textcoords="offset points", fontsize=9.5, color=ACCENT)
    ax.set_xscale("log")
    plain_numbers(ax.xaxis)
    ax.set_ylim(0, 100)
    ax.set_xlabel("L2 penalty, alpha (log scale)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("More penalty helps, until the model underfits")
    save(fig, SLUG, "regularisation")


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
