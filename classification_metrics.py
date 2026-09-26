"""Classification metrics that change the decision.

Article: https://chekh.dev/writing/classification-metrics-that-change-the-decision/
Run:     uv run python classification_metrics.py                 (about a minute; downloads one OpenML dataset)
         uv run python classification_metrics.py --charts-only   (redraw from results/)

The mammography dataset: 11,183 candidate regions from screening mammograms, six
measurements each, 2.3% of them real microcalcifications that need a radiologist's
attention. A gradient-boosted classifier scored out of fold, then every metric at
every threshold.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from _common import ACCENT, DANGER, INK, INK_2, MUTED, save, versions

SLUG = "classification-metrics-that-change-the-decision"
TARGET_RECALL = 0.80


def counts(y, p, threshold):
    flagged = p >= threshold
    tp, fp = int((flagged & (y == 1)).sum()), int((flagged & (y == 0)).sum())
    fn, tn = int((~flagged & (y == 1)).sum()), int((~flagged & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    return {
        "threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": (tp + tn) / len(y), "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "flagged_per_1000": 1000 * (tp + fp) / len(y),
    }


def compute() -> dict:
    d = fetch_openml(data_id=310, as_frame=True)  # mammography
    X = d.data.to_numpy(float)
    labels = d.target.astype(str)
    y = (labels == labels.value_counts().idxmin()).astype(int).to_numpy()  # 1 = the rare class
    print(f"{len(y):,} regions, {int(y.sum())} positive ({y.mean():.1%})")

    # Every row scored by a model that never saw it.
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    p = cross_val_predict(HistGradientBoostingClassifier(random_state=0), X, y, cv=cv, method="predict_proba")[:, 1]

    thresholds = np.round(np.arange(0.01, 0.99, 0.01), 2)
    sweep = [counts(y, p, t) for t in thresholds]
    at_half = counts(y, p, 0.5)
    best_f1 = max(sweep, key=lambda c: c["f1"])

    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, curve_thresholds = precision_recall_curve(y, p)
    # The highest threshold that still catches at least 90% of the real ones, read off
    # the curve, which visits every distinct score the model produced.
    for_recall = counts(y, p, float(curve_thresholds[recall[:-1] >= TARGET_RECALL].max()))
    results = {
        "versions": versions(),
        "rows": int(len(y)), "positives": int(y.sum()), "base_rate": float(y.mean()),
        "always_negative_accuracy": float(1 - y.mean()),
        "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
        "at_half": at_half, "best_f1": best_f1, "for_recall": for_recall,
        "sweep": sweep,
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr": {"precision": precision.tolist(), "recall": recall.tolist()},
    }
    for label, c in [("threshold 0.5", at_half), ("best F1", best_f1), (f"recall >= {TARGET_RECALL:.0%}", for_recall)]:
        print(f"{label:16s} t={c['threshold']:.2f}  acc {c['accuracy']:.3f}  precision {c['precision']:.2f}  "
              f"recall {c['recall']:.2f}  F1 {c['f1']:.2f}  flagged/1000 {c['flagged_per_1000']:.0f}  "
              f"TP {c['tp']} FP {c['fp']} FN {c['fn']} TN {c['tn']}")
    print(f"always negative: accuracy {results['always_negative_accuracy']:.3f}; ROC AUC {results['roc_auc']:.3f}; PR AUC {results['pr_auc']:.3f}")
    return results


def matrix(ax, c, title):
    """A 2×2 confusion matrix: correct cells in accent, mistakes in danger."""
    cells = [("caught", c["tp"], ACCENT, 0, 0), ("false alarm", c["fp"], DANGER, 1, 0),
             ("missed", c["fn"], DANGER, 0, 1), ("correctly cleared", c["tn"], ACCENT, 1, 1)]
    for label, n, color, col, row in cells:
        ax.add_patch(plt.Rectangle((col, 1 - row), 1, 1, facecolor=color, alpha=0.16, edgecolor="white", linewidth=3))
        ax.text(col + 0.5, 1 - row + 0.58, f"{n:,}", ha="center", va="center", fontsize=15, color=INK, fontweight="semibold")
        ax.text(col + 0.5, 1 - row + 0.26, label, ha="center", va="center", fontsize=8.5, color=INK_2)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5], ["real", "not real"], fontsize=9)
    ax.set_yticks([1.5, 0.5], ["flagged", "cleared"], fontsize=9)
    ax.tick_params(length=0)
    ax.grid(False)
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10.5, loc="center")


def draw(results: dict) -> None:
    half, low = results["at_half"], results["for_recall"]

    # Chart 1 · the same model at two thresholds
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6))
    matrix(axes[0], half, f"threshold 0.5: precision {half['precision']:.0%}, recall {half['recall']:.0%}")
    matrix(axes[1], low, f"threshold {low['threshold']:.2f}: precision {low['precision']:.0%}, recall {low['recall']:.0%}")
    fig.text(0.5, 0.01, "columns: what the region really was · rows: what the model said", ha="center", fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, SLUG, "confusion")

    # Chart 2 · ROC and precision–recall, side by side
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5))
    roc, pr = results["roc"], results["pr"]
    axes[0].plot(roc["fpr"], roc["tpr"], color=ACCENT, linewidth=2)
    axes[0].plot([0, 1], [0, 1], color=MUTED, linewidth=1)
    axes[0].set_xlabel("false-alarm rate (of the 97.7% that are not real)")
    axes[0].set_ylabel("recall")
    axes[0].set_title(f"ROC curve — area {results['roc_auc']:.2f}", fontsize=10.5)
    axes[1].plot(pr["recall"], pr["precision"], color=ACCENT, linewidth=2)
    axes[1].axhline(results["base_rate"], color=MUTED, linewidth=1)
    axes[1].annotate(f"guessing: {results['base_rate']:.1%}", (0.02, results["base_rate"]), xytext=(0, 5), textcoords="offset points", fontsize=8.5, color=INK_2)
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title(f"precision–recall curve — area {results['pr_auc']:.2f}", fontsize=10.5)
    for ax in axes:
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        for c, label in [(half, "0.5"), (low, f"{low['threshold']:.2f}")]:
            x = c["fp"] / (c["fp"] + c["tn"]) if ax is axes[0] else c["recall"]
            yv = c["recall"] if ax is axes[0] else c["precision"]
            ax.plot(x, yv, "o", color=INK, markersize=6, markeredgecolor="white", markeredgewidth=1.2)
            ax.annotate(f"t = {label}", (x, yv), xytext=(7, -3), textcoords="offset points", fontsize=8.5, color=INK)
    fig.tight_layout()
    save(fig, SLUG, "curves")

    # Chart 3 · precision and recall against the threshold: pick your operating point
    sweep = results["sweep"]
    t = [c["threshold"] for c in sweep]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(t, [c["recall"] for c in sweep], color=ACCENT, linewidth=2)
    ax.plot(t, [c["precision"] for c in sweep], color=INK, linewidth=1.6)
    ax.plot(t, [c["f1"] for c in sweep], color=INK_2, linewidth=1.2, linestyle=(0, (3, 2)))
    ax.annotate("recall", (t[-1], sweep[-1]["recall"]), xytext=(5, 0), textcoords="offset points", fontsize=9.5, color=ACCENT, va="center")
    ax.annotate("precision", (t[-1], sweep[-1]["precision"]), xytext=(5, 0), textcoords="offset points", fontsize=9.5, color=INK, va="center")
    ax.annotate("F1", (t[-1], sweep[-1]["f1"]), xytext=(5, 0), textcoords="offset points", fontsize=9.5, color=INK_2, va="center")
    for c, label in [(low, f"catch {TARGET_RECALL:.0%}: t = {low['threshold']:.2f}"), (half, "default: t = 0.5")]:
        ax.axvline(c["threshold"], color=MUTED, linewidth=1)
        ax.annotate(label, (c["threshold"], 0.03), xytext=(4, 0), textcoords="offset points", fontsize=8.5, color=INK_2)
    ax.set_xlim(0, 1.12)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("threshold: flag a region when its score is at least this")
    ax.set_ylabel("share")
    ax.set_title("Moving the threshold trades precision for recall")
    save(fig, SLUG, "threshold")


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
