from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results/baseline")
FIG_DIR = Path("figures/baseline")
NUM_EPISODES = 5


def load_episodes(prefix: str):
    dfs = []
    for i in range(1, NUM_EPISODES + 1):
        path = RESULTS_DIR / f"{prefix}_ep{i}.csv"
        dfs.append(pd.read_csv(path))
    return dfs


def average_curves(dfs, column):
    values = [df[column].values for df in dfs]
    return sum(values) / len(values)


def plot_avg_curve(fixed_dfs, random_dfs, column, ylabel, filename):
    plt.figure(figsize=(8, 5))

    avg_fixed = average_curves(fixed_dfs, column)
    avg_random = average_curves(random_dfs, column)

    steps = range(len(avg_fixed))

    plt.plot(steps, avg_fixed, label="Fixed-cycle")
    plt.plot(steps, avg_random, label="Random")

    plt.xlabel("Decision Step")
    plt.ylabel(ylabel)
    plt.title(f"Average {ylabel}")
    plt.legend()
    plt.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / filename)
    plt.close()


def plot_bar_comparison(fixed_dfs, random_dfs):
    fixed_wait = sum(df["total_waiting"].mean() for df in fixed_dfs) / len(fixed_dfs)
    random_wait = sum(df["total_waiting"].mean() for df in random_dfs) / len(random_dfs)

    fixed_reward = sum(df["reward"].mean() for df in fixed_dfs) / len(fixed_dfs)
    random_reward = sum(df["reward"].mean() for df in random_dfs) / len(random_dfs)

    plt.figure()
    plt.bar(["Fixed", "Random"], [fixed_wait, random_wait])
    plt.title("Average Waiting")
    plt.savefig(FIG_DIR / "avg_waiting_bar.png")
    plt.close()

    plt.figure()
    plt.bar(["Fixed", "Random"], [fixed_reward, random_reward])
    plt.title("Average Reward")
    plt.savefig(FIG_DIR / "avg_reward_bar.png")
    plt.close()


def main():
    fixed_dfs = load_episodes("fixed")
    random_dfs = load_episodes("random")

    plot_avg_curve(fixed_dfs, random_dfs, "total_waiting", "Total Waiting", "waiting.png")
    plot_avg_curve(fixed_dfs, random_dfs, "reward", "Reward", "reward.png")
    plot_avg_curve(fixed_dfs, random_dfs, "avg_travel_time", "Travel Time", "travel_time.png")

    plot_bar_comparison(fixed_dfs, random_dfs)

    print("Saved plots to figures/")


if __name__ == "__main__":
    main()