"""Regression: beat the baseline before you trust R².

Article: https://chekh.dev/writing/regression-beat-the-baseline-before-you-trust-r-squared/
Run:     uv run python regression_baselines.py                 (under a minute)
         uv run python regression_baselines.py --charts-only   (redraw from results/)

California housing: 20,640 census districts, eight features, the median house value
as the target. Two do-nothing baselines, a location-only baseline, a linear model and
gradient boosting, scored with MAE, RMSE and R² on the same held-out districts.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.datasets import fetch_california_housing
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from _common import ACCENT, INK, INK_2, MUTED, save, versions

SLUG = "regression-beat-the-baseline-before-you-trust-r-squared"
CAP = 5.00001  # the dataset's ceiling: values above $500,000 were recorded as this
DOLLARS = 100_000


class LocationOnly(KNeighborsRegressor):
    """The average price of the ten nearest districts, using latitude and longitude only."""

    def fit(self, X, y):
        return super().fit(X[:, -2:], y)

    def predict(self, X):
        return super().predict(X[:, -2:])


def compute() -> dict:
    X, y = fetch_california_housing(return_X_y=True)  # features end with latitude, longitude
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    models = {
        "always the mean": DummyRegressor(strategy="mean"),
        "always the median": DummyRegressor(strategy="median"),
        "ten nearest districts": LocationOnly(n_neighbors=10),
        "linear regression": make_pipeline(StandardScaler(), LinearRegression()),
        "gradient boosting": HistGradientBoostingRegressor(random_state=0),
    }
    summary, predictions = {}, {}
    for name, model in models.items():
        pred = model.fit(X_train, y_train).predict(X_test)
        predictions[name] = pred
        summary[name] = {
            "mae": float(mean_absolute_error(y_test, pred) * DOLLARS),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred)) * DOLLARS),
            "median_ae": float(median_absolute_error(y_test, pred) * DOLLARS),
            "r2": float(r2_score(y_test, pred)),
        }
        s = summary[name]
        print(f"{name:22s} MAE ${s['mae']:>8,.0f}  RMSE ${s['rmse']:>8,.0f}  median AE ${s['median_ae']:>8,.0f}  R² {s['r2']:6.3f}")

    # The toy that explains MAE against RMSE: same average miss, one big miss.
    toy = {}
    for label, errors in [("five misses of $10,000", [10, 10, 10, 10, 10]), ("four of $2,000 and one of $42,000", [2, 2, 2, 2, 42])]:
        e = np.array(errors, float)
        toy[label] = {"errors": errors, "mae": float(e.mean()), "rmse": float(np.sqrt((e**2).mean()))}
        print(f"{label:36s} MAE {toy[label]['mae']:.1f}  RMSE {toy[label]['rmse']:.1f}  (thousands)")

    best = predictions["gradient boosting"]
    residual = y_test - best
    capped = y_test >= CAP - 1e-6
    abs_err = np.abs(residual) * DOLLARS
    print(f"capped districts in test: {int(capped.sum())} of {len(y_test):,} ({capped.mean():.1%}); "
          f"boosting misses them by ${np.abs(residual[capped]).mean() * DOLLARS:,.0f} on average, "
          f"others by ${np.abs(residual[~capped]).mean() * DOLLARS:,.0f}")
    print(f"boosting absolute error: median ${np.median(abs_err):,.0f}, 90th percentile ${np.percentile(abs_err, 90):,.0f}, "
          f"largest ${abs_err.max():,.0f}")

    return {
        "versions": versions(),
        "rows": {"train": int(len(y_train)), "test": int(len(y_test))},
        "target_mean": float(y_train.mean() * DOLLARS), "target_std": float(y_train.std() * DOLLARS),
        "summary": summary,
        "toy": toy,
        "capped_share": float(capped.mean()),
        "residuals": {"predicted": (best * DOLLARS).tolist(), "residual": (residual * DOLLARS).tolist(), "capped": capped.tolist()},
        "abs_error_percentiles": {str(q): float(np.percentile(abs_err, q)) for q in (50, 75, 90, 95, 99)},
    }


def draw(results: dict) -> None:
    summary = results["summary"]
    order = ["always the mean", "always the median", "ten nearest districts", "linear regression", "gradient boosting"]

    # Chart 1 · RMSE and MAE per model, dollars
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for i, name in enumerate(order):
        s = summary[name]
        color = ACCENT if name == "gradient boosting" else INK_2
        ax.plot([s["mae"], s["rmse"]], [i, i], color=MUTED, linewidth=2, solid_capstyle="round", zorder=1)
        ax.plot(s["mae"], i, "o", color=color, markersize=7, markeredgecolor="white", markeredgewidth=1.2, zorder=2)
        ax.plot(s["rmse"], i, "s", color=color, markersize=7, markeredgecolor="white", markeredgewidth=1.2, zorder=2)
        ax.annotate(f"${s['rmse'] / 1000:,.0f}k", (s["rmse"], i), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9.5, color=INK)
        ax.annotate(f"${s['mae'] / 1000:,.0f}k", (s["mae"], i), xytext=(-8, 0), textcoords="offset points", va="center", ha="right", fontsize=9.5, color=INK_2)
    ax.set_yticks(range(len(order)), order)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(0, 140_000)
    ax.set_xticks([0, 40_000, 80_000, 120_000], ["$0", "$40k", "$80k", "$120k"])
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.legend(
        handles=[Line2D([], [], marker="o", color=INK_2, linestyle="none", markersize=7, label="MAE"),
                 Line2D([], [], marker="s", color=INK_2, linestyle="none", markersize=7, label="RMSE")],
        loc="upper right", frameon=False, fontsize=9.5,
    )
    ax.set_xlabel("typical miss on 4,128 held-out districts")
    ax.set_title("Every model against the do-nothing baselines")
    save(fig, SLUG, "models")

    # Chart 2 · residuals of the best model, with the capped districts picked out
    r = results["residuals"]
    predicted, residual, capped = np.array(r["predicted"]), np.array(r["residual"]), np.array(r["capped"])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.scatter(predicted[~capped], residual[~capped], s=5, color=MUTED, alpha=0.5, linewidths=0, rasterized=True)
    ax.scatter(predicted[capped], residual[capped], s=6, color=ACCENT, alpha=0.7, linewidths=0, rasterized=True)
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.annotate("districts recorded at the $500,001 cap", (330_000, 175_000), fontsize=9.5, color=ACCENT)
    ax.set_xlim(0, 550_000)
    ax.set_ylim(-350_000, 350_000)
    ax.set_xticks([0, 100_000, 200_000, 300_000, 400_000, 500_000], ["$0", "$100k", "$200k", "$300k", "$400k", "$500k"])
    ax.set_yticks([-300_000, -150_000, 0, 150_000, 300_000], ["−$300k", "−$150k", "0", "+$150k", "+$300k"])
    ax.set_xlabel("predicted price")
    ax.set_ylabel("actual − predicted")
    ax.set_title("Gradient boosting: where the misses are")
    save(fig, SLUG, "residuals")

    # Chart 3 · the spread of misses, and why RMSE sits to the right of MAE
    abs_err = np.clip(np.abs(residual), 0, 250_000)  # the last bin collects everything above
    s = summary["gradient boosting"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.hist(abs_err, bins=np.arange(0, 250_001, 5_000), color=MUTED)
    top = ax.get_ylim()[1]
    for value, label, color, height in [(s["median_ae"], "median", INK_2, 0.92), (s["mae"], "MAE", INK, 0.78), (s["rmse"], "RMSE", ACCENT, 0.64)]:
        ax.axvline(value, color=color, linewidth=1.4)
        ax.annotate(f"{label} ${value / 1000:,.0f}k", (value, top * height), xytext=(4, 0), textcoords="offset points", fontsize=9.5, color=color)
    ax.set_xlim(0, 250_000)
    ax.set_xticks([0, 50_000, 100_000, 150_000, 200_000, 250_000], ["$0", "$50k", "$100k", "$150k", "$200k", "$250k+"])
    ax.set_xlabel("size of the miss on a held-out district")
    ax.set_ylabel("districts")
    ax.set_title("Most misses are small; a few huge ones pull RMSE up")
    save(fig, SLUG, "errors")


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
