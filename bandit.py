"""Reinforcement learning, starting from a bandit.

Article: https://chekh.dev/writing/reinforcement-learning-starting-from-a-bandit/
Run:     uv run python bandit.py                 (about ten seconds)
         uv run python bandit.py --charts-only   (redraw from results/)

Five slot machines with hidden win rates. Four strategies for choosing which to pull,
each played 1,000 times in 500 independent games: greedy, epsilon-greedy, UCB and
Thompson sampling. The question every one of them faces is the question all of
reinforcement learning faces — explore, or exploit what you already know?
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _common import ACCENT, INK, INK_2, MUTED, save, versions

SLUG = "reinforcement-learning-starting-from-a-bandit"
ARMS = np.array([0.10, 0.25, 0.40, 0.55, 0.65])  # hidden chance each machine pays out 1
PULLS, LONG_PULLS, GAMES, EPSILON = 1_000, 5_000, 500, 0.10


def play(strategy: str, rng: np.random.Generator, pulls_per_game: int = PULLS) -> tuple[np.ndarray, np.ndarray]:
    """Every game at once: returns reward and whether-the-best-arm-was-pulled, (GAMES, pulls)."""
    k = len(ARMS)
    pulls = np.zeros((GAMES, k))  # how often each arm was tried
    wins = np.zeros((GAMES, k))  # how often it paid out
    rewards = np.zeros((GAMES, pulls_per_game))
    chose_best = np.zeros((GAMES, pulls_per_game), dtype=bool)
    rows = np.arange(GAMES)
    for t in range(pulls_per_game):
        if t < k:  # every strategy tries each arm once first
            arm = np.full(GAMES, t)
        elif strategy == "greedy":
            arm = np.argmax(wins / pulls, axis=1)
        elif strategy == "epsilon-greedy":
            arm = np.argmax(wins / pulls, axis=1)
            explore = rng.random(GAMES) < EPSILON
            arm[explore] = rng.integers(0, k, explore.sum())
        elif strategy == "UCB":  # the estimate plus a bonus for arms tried least
            arm = np.argmax(wins / pulls + np.sqrt(2 * np.log(t) / pulls), axis=1)
        elif strategy == "Thompson":  # draw a plausible win rate per arm, pull the highest
            arm = np.argmax(rng.beta(wins + 1, pulls - wins + 1), axis=1)
        reward = (rng.random(GAMES) < ARMS[arm]).astype(float)
        pulls[rows, arm] += 1
        wins[rows, arm] += reward
        rewards[:, t] = reward
        chose_best[:, t] = arm == ARMS.argmax()
    return rewards, chose_best


STRATEGIES = ["greedy", "epsilon-greedy", "UCB", "Thompson"]


def compute() -> dict:
    results: dict = {"versions": versions(), "arms": ARMS.tolist(), "pulls": PULLS, "games": GAMES, "epsilon": EPSILON, "strategies": {}}
    best = ARMS.max()
    for name in STRATEGIES:
        rewards, chose_best = play(name, np.random.default_rng(0))
        mean_reward = rewards.mean(axis=0)  # per pull, averaged over the games
        results["strategies"][name] = {
            "reward_per_pull": mean_reward.tolist(),
            "best_arm_share": chose_best.mean(axis=0).tolist(),
            "total_reward": float(rewards.sum(axis=1).mean()),
            "regret": float(PULLS * best - rewards.sum(axis=1).mean()),
            "best_arm_last_100": float(chose_best[:, -100:].mean()),
        }
        # The same strategy over a longer game: does the early exploration pay off?
        long_rewards, long_best = play(name, np.random.default_rng(0), LONG_PULLS)
        results["strategies"][name]["long"] = {
            "pulls": LONG_PULLS,
            "regret": float(LONG_PULLS * best - long_rewards.sum(axis=1).mean()),
            "best_arm_last_100": float(long_best[:, -100:].mean()),
        }
        s = results["strategies"][name]
        print(f"{name:15s} total reward {s['total_reward']:6.1f} of {PULLS * best:.0f} possible  regret {s['regret']:5.1f}  "
              f"best arm in the last 100 pulls {s['best_arm_last_100']:.0%}  |  after {LONG_PULLS:,} pulls: regret {s['long']['regret']:6.1f}, "
              f"best arm {s['long']['best_arm_last_100']:.0%}")
    return results


def draw_loop() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    ax.set_xlim(0, 8.0)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    def node(x, y, w, h, title, body):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.02", facecolor="#faf9f6", edgecolor=MUTED, linewidth=1))
        ax.text(x + w / 2, y + h - 0.08, title, ha="center", va="top", fontsize=9.5, color=INK, fontweight="semibold")
        ax.text(x + w / 2, y + 0.08, body, ha="center", va="bottom", fontsize=7.8, color=INK_2, linespacing=1.3)

    def arrow(x0, y0, x1, y1, label, above=True, color=INK_2):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=color, linewidth=1.2, connectionstyle="arc3,rad=0.25"))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + (0.32 if above else -0.32), label, ha="center", va="center", fontsize=8.5, color=color)

    # Top row: the loop in general. Bottom row: the same loop for a bandit and for RLHF.
    node(0.4, 2.0, 2.2, 1.2, "Agent", "the policy: a rule\nfor choosing actions")
    node(5.4, 2.0, 2.2, 1.2, "Environment", "the world the agent\nacts on")
    arrow(2.6, 2.95, 5.4, 2.95, "action", above=True, color=ACCENT)
    arrow(5.4, 2.25, 2.6, 2.25, "reward (and the new state)", above=False)
    rows = [
        ("Bandit", "the pull rule", "which machine", "1 or 0, no state"),
        ("RLHF for an LLM", "the language model", "the whole response", "a reward model's score"),
    ]
    ax.text(0.4, 1.55, "the same loop, twice:", fontsize=8.5, color=INK_2, va="center")
    for i, (setting, agent, action, reward) in enumerate(rows):
        y = 1.05 - i * 0.55
        ax.text(0.4, y, setting, fontsize=9, color=INK, va="center", fontweight="semibold")
        ax.text(2.0, y, f"agent: {agent}", fontsize=8.5, color=INK_2, va="center")
        ax.text(4.25, y, f"action: {action}", fontsize=8.5, color=INK_2, va="center")
        ax.text(6.2, y, f"reward: {reward}", fontsize=8.5, color=INK_2, va="center")
    save(fig, SLUG, "loop")


def spread(values: dict[str, float], gap: float) -> dict[str, float]:
    """Label positions at least `gap` apart, nudged from the values they label."""
    placed = {}
    for name, value in sorted(values.items(), key=lambda kv: kv[1]):
        placed[name] = max(value, max(placed.values(), default=-1) + gap)
    return placed


def draw_learning(results: dict) -> None:
    colors = {"greedy": MUTED, "epsilon-greedy": INK_2, "UCB": INK, "Thompson": ACCENT}
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    window = 20
    ends = [{}, {}]
    for name, s in results["strategies"].items():
        reward = np.convolve(s["reward_per_pull"], np.ones(window) / window, mode="valid")
        x = np.arange(window, len(s["reward_per_pull"]) + 1)
        axes[0].plot(x, reward, color=colors[name], linewidth=1.6)
        share = np.convolve(s["best_arm_share"], np.ones(window) / window, mode="valid")
        axes[1].plot(x, share, color=colors[name], linewidth=1.6)
        ends[0][name], ends[1][name] = reward[-1], share[-1]
    for ax, end, gap in zip(axes, ends, (0.025, 0.06)):
        for name, y in spread(end, gap).items():
            ax.annotate(name, (x[-1], y), xytext=(5, 0), textcoords="offset points", fontsize=8.5, color=colors[name], va="center")
    axes[0].axhline(max(results["arms"]), color=MUTED, linewidth=1)
    axes[0].annotate("the best machine pays 0.65", (10, max(results["arms"])), xytext=(0, 5), textcoords="offset points", fontsize=8.5, color=INK_2)
    axes[0].set_ylim(0.3, 0.72)
    axes[0].set_ylabel("reward per pull (20-pull average)")
    axes[0].set_title("What each strategy earns", fontsize=10.5)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("share of games pulling the best machine")
    axes[1].set_title("How often each finds the best machine", fontsize=10.5)
    for ax in axes:
        ax.set_xlim(0, 1_150)
        ax.set_xticks([0, 250, 500, 750, 1000])
        ax.set_xlabel("pull")
    fig.tight_layout()
    save(fig, SLUG, "learning")


if __name__ == "__main__":
    path = Path(__file__).parent / "results" / f"{SLUG}.json"
    if "--charts-only" in sys.argv:
        results = json.loads(path.read_text())
    else:
        results = compute()
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(results))
        print(f"  wrote {path.name}")
    draw_loop()
    draw_learning(results)
