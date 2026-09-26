"""Shared setup for the experiments.

Registers Lora so charts match chekh.dev, loads the Paper chart style, and saves each
chart under charts/<article-slug>/. Set CHEKH_DEV_SITE to a checkout of the site to copy
charts straight into its public/images/articles/ as well.
"""

import os
import platform
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent
CHARTS = ROOT / "charts"
SITE = Path(os.environ["CHEKH_DEV_SITE"]) if os.environ.get("CHEKH_DEV_SITE") else None

for font in (ROOT / "fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(font)
plt.style.use(ROOT / "paper.mplstyle")

# Paper tokens (chekh.dev DESIGN.md §2.1)
ACCENT = "#2f5d3f"
MUTED = "#b8b8ac"
INK = "#1c1e1a"
INK_2 = "#6f7268"
DANGER = "#8a2f2f"


def save(fig: plt.Figure, slug: str, name: str) -> None:
    """Write charts/<slug>/<name>.svg, with no timestamp so reruns diff cleanly."""
    folder = CHARTS / slug
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.svg"
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT).as_posix()}")
    if SITE:
        target = SITE / "apps" / "web" / "public" / "images" / "articles" / slug / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


def plain_numbers(axis) -> None:
    """Log-scale ticks as 1, 10, 1,000 rather than 10⁰, 10¹, 10³."""
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))


def versions() -> str:
    """The line each article quotes, so readers know what produced its numbers."""
    return (
        f"Python {platform.python_version()}, scikit-learn {sklearn.__version__}, "
        f"NumPy {np.__version__}"
    )
