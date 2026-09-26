# ml-experiments

The runnable experiments behind the machine-learning articles on
[chekh.dev](https://chekh.dev). Every number and chart those articles quote comes from a
script here, so you can rerun it, change it, and watch the article's results change with
it.

| Script | Article | Runtime |
|---|---|---|
| `bias_variance.py` | [Bias and variance, explained with one dataset](https://chekh.dev/writing/bias-and-variance-drawn-from-one-dataset/) | about 1 min |
| `overfitting_fixes.py` | [Six fixes for overfitting, tested side by side](https://chekh.dev/writing/six-fixes-for-overfitting-measured-side-by-side/) | about 15 min on 12 cores; `--charts-only` redraws from `results/` in seconds |
| `cv_mistakes.py` | [Cross-validation mistakes that fake a good score](https://chekh.dev/writing/cross-validation-mistakes-that-fake-a-good-score/) | about 2 min |

## Run it

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/). No API keys, no accounts.

```sh
git clone https://github.com/thechekh/ml-experiments
cd ml-experiments
uv sync                          # creates .venv with the pinned versions
uv run python bias_variance.py   # or any script in the table
```

Each script prints its headline numbers as it goes, writes every number the article
quotes to `results/<slug>.json`, and draws its charts to `charts/<slug>/` as SVG.

Datasets come from scikit-learn (bundled, or downloaded once to `~/scikit_learn_data`) and
from OpenML. The first run downloads about 10 MB.

## What is in here

- `_common.py` — shared setup: fonts, chart style, where charts are saved
- `paper.mplstyle` — the site's chart style for matplotlib: colours, type, hairline axes
- `fonts/` — Lora, the site's typeface, under the SIL Open Font License (`fonts/OFL.txt`)
- `results/` — the numbers behind each article, from the last run
- `charts/` — the charts, from the last run

## Change something

The scripts are meant to be edited. Some places to start:

- `bias_variance.py` — widen the depth range, or swap `DecisionTreeRegressor` for
  `KNeighborsRegressor` and watch what bias and variance do
- `overfitting_fixes.py` — change `NOISE` (the share of wrong labels) or `TRAIN` (the
  training-set size); each fix's knob is chosen by cross-validation, so the comparison
  stays fair
- `cv_mistakes.py` — put your own dataset through any of the five cases

Versions are pinned in `uv.lock`. The numbers in the articles were produced with Python
3.12, scikit-learn 1.9 and NumPy 2.5; other versions may differ in the last digit.

## Licence

MIT, see `LICENSE`. The Lora font files in `fonts/` are under the SIL Open Font License.
