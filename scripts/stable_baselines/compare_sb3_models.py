from __future__ import annotations

import csv
from pathlib import Path

from scripts.stable_baselines.train_a2c import A2CAgent
from scripts.stable_baselines.train_dqn import DQNAgent
from scripts.stable_baselines.train_ppo import PPOAgent


DATASETS_DIR = Path("CityFlow/hangzhou_datasets")

STATE_MODES = [
    "capped_no_phase",
    "capped_with_phase",
    "bucketed_no_phase",
    "bucketed_with_phase",
]

REWARD_MODES = [
    "env",
    "wait_only",
    "wait_plus_vehicle",
    "delta_wait",
]


def save_rows(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    dataset_dirs = sorted(d for d in DATASETS_DIR.iterdir() if d.is_dir())
    if not dataset_dirs:
        raise ValueError(f"No dataset folders found in {DATASETS_DIR}")

    rows: list[dict] = []
    agent_classes = [DQNAgent, PPOAgent, A2CAgent]

    for dataset_dir in dataset_dirs:
        print("\n" + "=" * 90)
        print(f"Dataset: {dataset_dir.name}")

        for state_mode in STATE_MODES:
            for reward_mode in REWARD_MODES:
                for agent_cls in agent_classes:
                    print(
                        f"Training {agent_cls.__name__} | "
                        f"dataset={dataset_dir.name} | "
                        f"state={state_mode} | reward={reward_mode}"
                    )

                    agent = agent_cls(
                        dataset_dir=str(dataset_dir),
                        state_mode=state_mode,
                        reward_mode=reward_mode,
                    )
                    agent.train(total_timesteps=50_000)
                    row = agent.evaluate(episodes=10)
                    rows.append(row)
                    agent.close()

    out_path = Path("results/stable_baselines/sb3_comparison_all_datasets_states_rewards.csv")
    save_rows(rows, out_path)

    print("\nSaved comparison CSV:")
    print(out_path)

    print("\nSummary:")
    for row in rows:
        print(
            f"{row['dataset']:30s} | "
            f"{row['model']:4s} | "
            f"{row['state_mode']:18s} | "
            f"{row['reward_mode']:17s} | "
            f"reward={row['mean_total_reward']:.2f} | "
            f"final_wait={row['mean_final_waiting']:.2f} | "
            f"avg_wait={row['mean_avg_waiting']:.2f} | "
            f"travel={row['mean_avg_travel_time']:.2f}"
        )


if __name__ == "__main__":
    main()