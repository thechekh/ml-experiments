"""Cross-validation mistakes that fake a good score.

Article: https://chekh.dev/writing/cross-validation-mistakes-that-fake-a-good-score/
Run:     uv run python cv_mistakes.py   (about two minutes; downloads three OpenML datasets once)

Each case scores the same model twice: once with the mistake, once without it.
"""

import json
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from _common import ACCENT, DANGER, INK, INK_2, MUTED, save, versions

SLUG = "cross-validation-mistakes-that-fake-a-good-score"
SEEDS = range(5)
results: dict = {"versions": versions(), "cases": []}


def record(key, title, data, metric, leaky, honest, **extra):
    results["cases"].append({"key": key, "title": title, "data": data, "metric": metric, "leaky": leaky, "honest": honest, **extra})
    print(f"{title:40s} {metric:9s} leaky {leaky:.3f}  honest {honest:.3f}")


# 1 · Selecting features before the split — on pure noise.
leaky, honest = [], []
for seed in range(20):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(100, 10_000))
    y = rng.integers(0, 2, 100)  # labels unrelated to X: the honest answer is 50%
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    X_selected = SelectKBest(f_classif, k=20).fit_transform(X, y)
    leaky.append(cross_val_score(LogisticRegression(max_iter=1_000), X_selected, y, cv=cv).mean())
    pipeline = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1_000))
    honest.append(cross_val_score(pipeline, X, y, cv=cv).mean())
record("selection", "Feature selection before the split", "100 rows of pure noise, 10,000 features", "accuracy", float(np.mean(leaky)), float(np.mean(honest)))


# 2 · Oversampling the rare class before the split.
ozone = fetch_openml(data_id=1487, as_frame=True)  # ozone-level-8hr: 2,534 days, 6.3% ozone days
X = ozone.data.to_numpy(float)
labels = ozone.target.astype(str)
y = (labels == labels.value_counts().idxmin()).astype(int).to_numpy()  # 1 = the rare ozone day


def oversample(X, y, rng):
    """Duplicate positives at random until the classes are even."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    idx = np.concatenate([neg, pos, rng.choice(pos, len(neg) - len(pos))])
    return X[idx], y[idx]


def forest(seed):
    return RandomForestClassifier(300, random_state=seed, n_jobs=-1)


leaky, honest = [], []
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    Xo, yo = oversample(X, y, rng)  # the mistake: copies land in both train and test folds
    leaky.append(np.mean([f1_score(yo[b], forest(seed).fit(Xo[a], yo[a]).predict(Xo[b])) for a, b in cv.split(Xo, yo)]))
    scores = []
    for a, b in cv.split(X, y):
        Xa, ya = oversample(X[a], y[a], rng)  # oversample the training fold only
        scores.append(f1_score(y[b], forest(seed).fit(Xa, ya).predict(X[b])))
    honest.append(np.mean(scores))
record("oversampling", "Oversampling before the split", f"ozone days, {y.mean():.1%} positive", "F1", float(np.mean(leaky)), float(np.mean(honest)))


# 3 · Shuffled folds on a time series.
bikes = fetch_openml(data_id=42712, as_frame=True).frame  # hourly rentals, 2011–2012, in time order
y = bikes["count"].to_numpy(float)
features = bikes.drop(columns=["count", "casual", "registered"], errors="ignore")
X = np.column_stack([features[c].cat.codes if features[c].dtype.name == "category" else features[c] for c in features])
model = HistGradientBoostingRegressor(random_state=0)
shuffled = float(np.mean([cross_val_score(model, X, y, cv=KFold(5, shuffle=True, random_state=s), scoring="r2").mean() for s in SEEDS]))
blocked = float(cross_val_score(model, X, y, cv=KFold(5), scoring="r2").mean())
ordered = float(cross_val_score(model, X, y, cv=TimeSeriesSplit(5), scoring="r2").mean())
mae = {
    "shuffled": float(-cross_val_score(model, X, y, cv=KFold(5, shuffle=True, random_state=0), scoring="neg_mean_absolute_error").mean()),
    "ordered": float(-cross_val_score(model, X, y, cv=TimeSeriesSplit(5), scoring="neg_mean_absolute_error").mean()),
}
record("time", "Shuffled folds on a time series", f"{len(y):,} hourly bike rentals", "R²", shuffled, ordered, blocked=blocked, mae=mae, mean_count=float(y.mean()))


# 4 · The same subject on both sides of the split.
meta = json.load(urllib.request.urlopen("https://api.openml.org/api/v1/json/data/40966"))
arff = urllib.request.urlopen(meta["data_set_description"]["url"]).read().decode()
rows = [line for line in arff.splitlines() if line and line[0] not in "@%"]
mice = fetch_openml(data_id=40966, as_frame=True)  # MiceProtein: 72 mice, 15 measurements each
# fetch_openml drops MouseID, so take it from the raw file — after checking the rows line up.
assert [r.rsplit(",", 1)[1].strip("'\"") for r in rows] == mice.target.astype(str).tolist()
groups = np.array([r.split(",")[0].strip("'\"").split("_")[0] for r in rows])
X, y = mice.data.to_numpy(float), mice.target.astype(str).to_numpy()
model = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=5_000))
mixed = float(np.mean([cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=s)).mean() for s in SEEDS]))
grouped = float(cross_val_score(model, X, y, cv=GroupKFold(5), groups=groups).mean())
record("groups", "The same mouse on both sides", f"{len(set(groups))} mice × 15 measurements, 8 classes", "accuracy", mixed, grouped)


# 5 · Scaling before the split — the control that does not leak much.
X, y = load_breast_cancer(return_X_y=True)
leaky, honest = [], []
for seed in SEEDS:
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    leaky.append(cross_val_score(LogisticRegression(max_iter=5_000), StandardScaler().fit_transform(X), y, cv=cv).mean())
    honest.append(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5_000)), X, y, cv=cv).mean())
record("scaling", "Scaling before the split", "569 tumours, 30 features", "accuracy", float(np.mean(leaky)), float(np.mean(honest)))


# Chart 1 · every case, leaky against honest
cases = sorted(results["cases"], key=lambda c: c["leaky"] - c["honest"])
fig, ax = plt.subplots(figsize=(6.4, 3.3))
for i, c in enumerate(cases):
    ax.plot([c["honest"], c["leaky"]], [i, i], color=MUTED, linewidth=2, solid_capstyle="round", zorder=1)
    ax.plot(c["leaky"], i, "o", color=DANGER, markersize=8, markeredgecolor="white", markeredgewidth=1.5, zorder=2)
    ax.plot(c["honest"], i, "o", color=ACCENT, markersize=8, markeredgecolor="white", markeredgewidth=1.5, zorder=3)
    gap = c["leaky"] - c["honest"]
    if gap > 0.02:
        ax.annotate(f"{c['honest']:.2f}", (c["honest"], i), xytext=(-8, 0), textcoords="offset points", ha="right", va="center", fontsize=9.5, color=ACCENT)
        ax.annotate(f"{c['leaky']:.2f}", (c["leaky"], i), xytext=(8, 0), textcoords="offset points", ha="left", va="center", fontsize=9.5, color=DANGER)
    else:
        ax.annotate(f"{c['honest']:.2f} both ways", (c["honest"], i), xytext=(-10, 0), textcoords="offset points", ha="right", va="center", fontsize=9.5, color=INK)
ax.set_yticks(range(len(cases)), [f"{c['title']} ({c['metric']})" for c in cases], fontsize=9.5)
top = len(cases) - 1
ax.annotate("honest", (cases[top]["honest"], top), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9.5, color=ACCENT)
ax.annotate("with the mistake", (cases[top]["leaky"], top), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9.5, color=DANGER)
ax.set_xlim(0, 1.08)
ax.set_ylim(-0.6, len(cases) - 0.3)
ax.grid(axis="y", visible=False)
ax.grid(axis="x", visible=True)
ax.set_xlabel("cross-validated score (1 is perfect)")
ax.set_title("The same model, scored with and without each mistake")
save(fig, SLUG, "leaky-vs-honest")


# Chart 2 · where each split puts the test rows
def draw(ax, test_masks, title, blocks=None):
    """One row per fold: test rows in accent, training rows in grey, unused rows blank.

    Consecutive cells of the same kind are drawn as one rectangle, so no seams show.
    """
    for fold, (train, test) in enumerate(test_masks):
        kinds = np.where(test, 2, np.where(train, 1, 0))
        start = 0
        for i in range(1, len(kinds) + 1):
            if i == len(kinds) or kinds[i] != kinds[start]:
                color = {2: ACCENT, 1: MUTED}.get(int(kinds[start]))
                if color:
                    ax.add_patch(Rectangle((start, fold), i - start, 0.8, color=color, linewidth=0))
                start = i
    for edge in blocks or []:
        ax.axvline(edge, color=INK_2, linewidth=0.8)
    ax.set_xlim(0, len(test_masks[0][0]))
    ax.set_ylim(len(test_masks), -0.2)
    ax.set_yticks([f + 0.4 for f in range(len(test_masks))], [f"fold {f + 1}" for f in range(len(test_masks))], fontsize=8.5)
    ax.set_xticks([])
    ax.grid(False)
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.set_title(title, fontsize=10, loc="left")


def masks(cv, n, groups=None):
    out = []
    for train, test in cv.split(np.zeros((n, 1)), np.zeros(n), groups):
        t, s = np.zeros(n, bool), np.zeros(n, bool)
        t[train], s[test] = True, True
        out.append((t, s))
    return out


n_time, per_mouse, n_mice = 60, 6, 10
toy_groups = np.repeat(np.arange(n_mice), per_mouse)
fig, axes = plt.subplots(2, 2, figsize=(8.2, 4.2))
draw(axes[0, 0], masks(KFold(5, shuffle=True, random_state=0), n_time), "Shuffled folds: tests scattered through time")
draw(axes[0, 1], masks(TimeSeriesSplit(5), n_time), "Time-ordered: always test on the future")
draw(axes[1, 0], masks(KFold(5, shuffle=True, random_state=3), n_mice * per_mouse), "Random folds: each mouse on both sides",
     blocks=[per_mouse * k for k in range(1, n_mice)])
draw(axes[1, 1], masks(GroupKFold(5), n_mice * per_mouse, toy_groups), "Grouped folds: each mouse on one side",
     blocks=[per_mouse * k for k in range(1, n_mice)])
for ax in axes[0]:
    ax.set_xlabel("time, earlier to later", fontsize=9, color=INK_2)
for ax in axes[1]:
    ax.set_xlabel("measurements, one block per mouse", fontsize=9, color=INK_2)
fig.legend(handles=[Patch(color=ACCENT, label="test rows"), Patch(color=MUTED, label="training rows")],
           loc="lower left", ncol=2, frameon=False, fontsize=9, handlelength=1, handleheight=1)
fig.tight_layout(h_pad=1.6, rect=(0, 0.05, 1, 1))
save(fig, SLUG, "splits")

out = Path(__file__).parent / "results" / f"{SLUG}.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(results, indent=2))
print(f"  wrote {out.name}")
