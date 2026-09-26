"""Machine learning in one map: what each branch is for.

Article: https://chekh.dev/writing/machine-learning-in-one-map-what-each-branch-is-for/
Run:     uv run python ml_map.py                 (about a minute)
         uv run python ml_map.py --charts-only   (redraw from results/)

Four branches on one dataset — the 1,797 handwritten digits that ship with
scikit-learn. Supervised classification (labels given), clustering (no labels),
dimensionality reduction (64 pixels to 2 numbers) and self-supervised learning (fill in
the hidden half of a digit; the image is its own answer). Plus the map itself.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from _common import ACCENT, DANGER, INK, INK_2, MUTED, save, versions

SLUG = "machine-learning-in-one-map-what-each-branch-is-for"


def compute() -> dict:
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    results: dict = {"versions": versions(), "rows": {"train": int(len(y_train)), "test": int(len(y_test))}}

    # Supervised classification: inputs and the right answers.
    classifier = LogisticRegression(max_iter=3_000).fit(X_train, y_train)
    predicted = classifier.predict(X_test)
    results["classification"] = {
        "accuracy": float((predicted == y_test).mean()),
        "examples": [{"pixels": X_test[i].tolist(), "true": int(y_test[i]), "predicted": int(predicted[i])} for i in range(12)],
    }
    print(f"classification: {results['classification']['accuracy']:.1%} of {len(y_test)} test digits right")

    # Clustering: inputs only. Ten groups, and no idea what a digit is.
    kmeans = KMeans(n_clusters=10, n_init=10, random_state=0).fit(X_train)
    results["clustering"] = {
        "centres": kmeans.cluster_centers_.tolist(),
        "agreement": float(adjusted_rand_score(y_train, kmeans.labels_)),
    }
    print(f"clustering: adjusted Rand index against the true labels {results['clustering']['agreement']:.2f}")

    # Dimensionality reduction: 64 numbers per digit to 2, keeping what varies most.
    pca = PCA(n_components=2).fit(X_train)
    points = pca.transform(X_test)
    results["pca"] = {
        "explained": [float(v) for v in pca.explained_variance_ratio_],
        "points": [{"x": float(p[0]), "y": float(p[1]), "digit": int(d)} for p, d in zip(points[:400], y_test[:400])],
    }
    print(f"PCA: two components keep {sum(results['pca']['explained']):.0%} of the variation")

    # Self-supervised: hide the bottom half of each digit; the top half must predict it.
    # Nothing here uses a label — the image itself is the answer.
    def mask(images):
        hidden = images.copy().reshape(-1, 8, 8)
        hidden[:, 4:, :] = 0
        return hidden.reshape(-1, 64)

    inpainter = MLPRegressor(hidden_layer_sizes=(256,), max_iter=500, random_state=0).fit(mask(X_train), X_train)
    filled = np.clip(inpainter.predict(mask(X_test)), 0, 1)
    bottom = slice(32, 64)  # the hidden pixels
    baseline = np.tile(X_train.mean(axis=0), (len(X_test), 1))
    results["inpainting"] = {
        "error": float(np.abs(filled[:, bottom] - X_test[:, bottom]).mean()),
        "baseline_error": float(np.abs(baseline[:, bottom] - X_test[:, bottom]).mean()),
        "examples": [{"masked": mask(X_test[i : i + 1])[0].tolist(), "filled": filled[i].tolist(), "original": X_test[i].tolist()} for i in range(8)],
    }
    print(f"inpainting: mean pixel error {results['inpainting']['error']:.3f} against {results['inpainting']['baseline_error']:.3f} for the average digit")
    return results


def box(ax, x, y, w, h, title, body, strong=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.02", facecolor="#faf9f6", edgecolor=ACCENT if strong else MUTED, linewidth=1.4 if strong else 1))
    ax.text(x + w / 2, y + h - 0.07, title, ha="center", va="top", fontsize=9 if len(title) > 14 else 9.5, color=INK, fontweight="semibold")
    ax.text(x + w / 2, y + 0.06, body, ha="center", va="bottom", fontsize=7.8, color=INK_2, linespacing=1.3)


def draw_map() -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.set_xlim(0, 8.4)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    box(ax, 2.7, 4.3, 3.0, 0.75, "Machine learning", "a program that learns a rule from examples\ninstead of being given one", strong=True)
    branches = [
        (0.15, "Supervised", "you have inputs and\nthe right answers"),
        (2.2, "Unsupervised", "you have inputs\nand nothing else"),
        (4.25, "Reinforcement", "you act, and the\nreward comes later"),
        (6.3, "Self-supervised", "the input is its\nown answer"),
    ]
    for x, title, body in branches:
        box(ax, x, 2.75, 1.95, 0.95, title, body)
        ax.plot([4.2, x + 0.975], [4.3, 3.7], color=MUTED, linewidth=1)
    # Leaves hang off a spine on the left of their branch, so no line crosses a box.
    leaves = [
        (0.15, 1.6, "Classification", "which category?\nspam, a digit, fraud"),
        (0.15, 0.55, "Regression", "how much?\na price, a temperature"),
        (2.2, 1.6, "Clustering", "which things\ngo together?"),
        (2.2, 0.55, "Dimensionality reduction", "the same data\nin fewer numbers"),
    ]
    for x, y, title, body in leaves:
        box(ax, x + 0.35, y, 1.6, 0.75, title, body)
        ax.plot([x + 0.18, x + 0.35], [y + 0.375, y + 0.375], color=MUTED, linewidth=1)
    for x in (0.15, 2.2):
        ax.plot([x + 0.18, x + 0.18], [2.75, 0.925], color=MUTED, linewidth=1)
    ax.text(5.225, 2.35, "bandits, games, robots,\nRLHF for language models", ha="center", va="top", fontsize=7.8, color=INK_2, linespacing=1.3)
    ax.text(7.275, 2.35, "predict the hidden part:\nthe next word, a masked\npatch — how LLMs are\npretrained", ha="center", va="top", fontsize=7.8, color=INK_2, linespacing=1.3)
    save(fig, SLUG, "map")


def digit(ax, pixels, title="", color=INK):
    ax.imshow(np.array(pixels).reshape(8, 8), cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_edgecolor("#e4e4da")
    if title:
        ax.set_title(title, fontsize=9, color=color, loc="center", pad=4)


def draw_branches(results: dict) -> None:
    # Classification: twelve test digits, the model's answer above each.
    examples = results["classification"]["examples"]
    fig, axes = plt.subplots(1, 12, figsize=(8.4, 1.25))
    for ax, e in zip(axes, examples):
        right = e["predicted"] == e["true"]
        digit(ax, e["pixels"], f"{e['predicted']}" if right else f"{e['predicted']} (is {e['true']})", INK if right else DANGER)
    fig.suptitle(f"Supervised classification — {results['classification']['accuracy']:.1%} of {results['rows']['test']} held-out digits right", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    save(fig, SLUG, "classification")

    # Clustering: the ten cluster centres, never shown a label.
    centres = results["clustering"]["centres"]
    fig, axes = plt.subplots(1, 10, figsize=(8.4, 1.25))
    for ax, c in zip(axes, centres):
        digit(ax, c)
    fig.suptitle("Clustering — the centres of ten groups found without a single label", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    save(fig, SLUG, "clustering")

    # Dimensionality reduction: 400 test digits as two numbers each.
    pca = results["pca"]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for p in pca["points"]:
        ax.text(p["x"], p["y"], str(p["digit"]), ha="center", va="center", fontsize=8, color=ACCENT if p["digit"] in (0, 1) else INK_2)
    xs, ys = [p["x"] for p in pca["points"]], [p["y"] for p in pca["points"]]
    ax.set_xlim(min(xs) - 0.2, max(xs) + 0.2)
    ax.set_ylim(min(ys) - 0.2, max(ys) + 0.2)
    ax.set_xlabel("first component")
    ax.set_ylabel("second component")
    ax.set_title(f"Dimensionality reduction — 64 pixels to 2 numbers, keeping {sum(pca['explained']):.0%} of the variation")
    save(fig, SLUG, "pca")

    # Self-supervised: masked input, the model's completion, the original.
    examples = results["inpainting"]["examples"]
    fig, axes = plt.subplots(3, len(examples), figsize=(8.4, 3.4))
    for col, e in enumerate(examples):
        digit(axes[0, col], e["masked"])
        digit(axes[1, col], e["filled"])
        digit(axes[2, col], e["original"])
    for row, label in enumerate(["shown", "completed", "original"]):
        axes[row, 0].set_ylabel(label, fontsize=9, color=INK_2, rotation=0, ha="right", va="center", labelpad=8)
    fig.suptitle("Self-supervised — predict the hidden half; the image is its own answer", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, SLUG, "inpainting")


if __name__ == "__main__":
    path = Path(__file__).parent / "results" / f"{SLUG}.json"
    if "--charts-only" in sys.argv:
        results = json.loads(path.read_text())
    else:
        results = compute()
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(results))
        print(f"  wrote {path.name}")
    draw_map()
    draw_branches(results)
