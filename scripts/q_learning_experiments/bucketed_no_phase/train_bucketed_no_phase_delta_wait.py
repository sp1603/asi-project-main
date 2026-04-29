from __future__ import annotations

import csv
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict

import numpy as np

from env.cityflow_env import CityFlowEnv


DATASETS_DIR = Path("CityFlow/hangzhou_datasets")
EXPERIMENT_NAME = "q_learning_bucketed_no_phase_delta_wait"
CURRENT_SCENARIO_DIR: Path | None = None


def get_scenario_dir() -> Path:
    if CURRENT_SCENARIO_DIR is None:
        raise ValueError("CURRENT_SCENARIO_DIR has not been set.")
    return CURRENT_SCENARIO_DIR


def load_config() -> dict:
    with (get_scenario_dir() / "config.json").open() as f:
        return json.load(f)


def load_roadnet() -> dict:
    config = load_config()
    scenario_dir = get_scenario_dir()

    roadnet_name = Path(config["roadnetFile"]).name
    roadnet_path = scenario_dir / roadnet_name

    with roadnet_path.open() as f:
        return json.load(f)


def get_controlled_intersection_id(roadnet: dict) -> str:
    for inter in roadnet["intersections"]:
        if not inter.get("virtual", False):
            return inter["id"]
    raise ValueError("No non-virtual intersection found")


def build_lane_ids() -> list[str]:
    roadnet = load_roadnet()
    intersection_id = get_controlled_intersection_id(roadnet)

    roads_by_id = {r["id"]: r for r in roadnet["roads"]}
    intersection = next(i for i in roadnet["intersections"] if i["id"] == intersection_id)

    lane_ids: list[str] = []
    for road_id in intersection["roads"]:
        road = roads_by_id[road_id]
        if road["endIntersection"] != intersection_id:
            continue
        for i in range(len(road["lanes"])):
            lane_ids.append(f"{road_id}_{i}")

    return sorted(lane_ids)


def build_phase_ids() -> list[int]:
    roadnet = load_roadnet()
    intersection_id = get_controlled_intersection_id(roadnet)

    intersection = next(i for i in roadnet["intersections"] if i["id"] == intersection_id)

    phase_ids = [
        idx
        for idx, phase in enumerate(intersection["trafficLight"]["lightphases"])
        if phase.get("availableRoadLinks")
    ]

    if not phase_ids:
        raise ValueError("No valid traffic-light phases found")

    return phase_ids


def build_runtime_config(results_dir: Path) -> Path:
    scenario_dir = get_scenario_dir()
    config = load_config()

    roadnet_name = Path(config["roadnetFile"]).name
    flow_name = Path(config["flowFile"]).name

    roadnet_path = scenario_dir / roadnet_name
    flow_path = scenario_dir / flow_name

    if not roadnet_path.exists():
        raise FileNotFoundError(f"Missing roadnet file: {roadnet_path}")
    if not flow_path.exists():
        raise FileNotFoundError(f"Missing flow file: {flow_path}")

    runtime_config = dict(config)
    runtime_config["dir"] = str(scenario_dir.resolve()) + "/"
    runtime_config["roadnetFile"] = roadnet_name
    runtime_config["flowFile"] = flow_name

    runtime_config["roadnetLogFile"] = "replay_roadnet.json"
    runtime_config["replayLogFile"] = "replay.txt"

    runtime_config["saveReplay"] = False

    runtime_config_path = results_dir / "runtime_config.json"
    with runtime_config_path.open("w") as f:
        json.dump(runtime_config, f, indent=2)

    return runtime_config_path


def make_env(results_dir: Path) -> CityFlowEnv:
    runtime_config_path = build_runtime_config(results_dir)
    roadnet = load_roadnet()
    intersection_id = get_controlled_intersection_id(roadnet)

    return CityFlowEnv(
        config_path=str(runtime_config_path),
        intersection_id=intersection_id,
        lane_ids=build_lane_ids(),
        phase_ids=build_phase_ids(),
        action_interval=5,
        max_steps=720,
        render_mode=None,
    )


def aggregate_by_direction(obs: np.ndarray) -> np.ndarray:
    if obs.shape[0] % 4 != 0:
        raise ValueError(f"Unexpected obs size: {obs.shape[0]}")
    lanes_per_dir = obs.shape[0] // 4
    return obs.reshape(4, lanes_per_dir).sum(axis=1)


def bucket_queue_length(x: float) -> int:
    if x <= 0:
        return 0
    if x <= 2:
        return 1
    if x <= 5:
        return 2
    if x <= 10:
        return 3
    return 4


def discretize_state(obs: np.ndarray) -> tuple[int, int, int, int]:
    return tuple(bucket_queue_length(float(x)) for x in aggregate_by_direction(obs))


def reward_delta_wait(info: dict, prev_info: dict | None) -> float:
    if prev_info is None:
        return -float(info["total_waiting"])
    return float(prev_info["total_waiting"]) - float(info["total_waiting"])


def epsilon_greedy_action(
    q_table: DefaultDict[tuple[int, int, int, int], np.ndarray],
    state: tuple[int, int, int, int],
    n_actions: int,
    epsilon: float,
) -> int:
    if random.random() < epsilon:
        return random.randrange(n_actions)
    return int(np.argmax(q_table[state]))


def save_rows(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_q_table(q_table, out_path: Path) -> None:
    with out_path.open("wb") as f:
        pickle.dump(dict(q_table), f)


def train_q_learning(
    results_dir: Path,
    episodes: int = 2000,
    alpha: float = 0.10,
    gamma: float = 0.95,
    epsilon: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.997,
):
    env = make_env(results_dir)
    n_actions = env.action_space.n
    env.close()

    q_table = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))
    training_rows: list[dict] = []

    for episode in range(episodes):
        env = make_env(results_dir)
        obs, info = env.reset()
        state = discretize_state(obs)

        done = False
        total_reward = 0.0
        prev_info = None

        while not done:
            action = epsilon_greedy_action(q_table, state, n_actions, epsilon)
            next_obs, _, terminated, truncated, next_info = env.step(action)
            reward = reward_delta_wait(next_info, info)

            next_state = discretize_state(next_obs)
            done = terminated or truncated

            td_target = (
                reward if done else reward + gamma * float(np.max(q_table[next_state]))
            )
            q_table[state][action] += alpha * (td_target - q_table[state][action])

            state = next_state
            total_reward += reward
            prev_info = info
            info = next_info

        env.close()
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        training_rows.append(
            {
                "episode": episode + 1,
                "total_reward": total_reward,
                "final_waiting": float(info["total_waiting"]),
                "final_avg_travel_time": float(info["avg_travel_time"]),
                "epsilon": epsilon,
            }
        )

    return q_table, training_rows


def run_q_learning_episode(results_dir: Path, q_table):
    env = make_env(results_dir)
    obs, info = env.reset()
    state = discretize_state(obs)

    done = False
    total_reward = 0.0
    prev_info = None

    while not done:
        action = int(np.argmax(q_table[state]))
        obs, _, terminated, truncated, info = env.step(action)
        reward = reward_delta_wait(info, prev_info)

        state = discretize_state(obs)
        total_reward += reward
        prev_info = info
        done = terminated or truncated

    env.close()

    return {
        "method": EXPERIMENT_NAME,
        "total_reward": total_reward,
        "final_waiting": float(info["total_waiting"]),
        "final_avg_travel_time": float(info["avg_travel_time"]),
    }


def main() -> None:
    global CURRENT_SCENARIO_DIR

    random.seed(42)
    np.random.seed(42)

    dataset_dirs = sorted(d for d in DATASETS_DIR.iterdir() if d.is_dir())
    if not dataset_dirs:
        raise ValueError(f"No dataset folders found in {DATASETS_DIR}")

    all_summaries: list[dict] = []

    for dataset_dir in dataset_dirs:
        CURRENT_SCENARIO_DIR = dataset_dir
        results_dir = Path("results") / dataset_dir.name / EXPERIMENT_NAME
        results_dir.mkdir(parents=True, exist_ok=True)

        config = load_config()
        roadnet_name = Path(config["roadnetFile"]).name
        flow_name = Path(config["flowFile"]).name

        print("\n" + "=" * 60)
        print("Training Q-learning on Hangzhou (delta-wait reward)...")
        print(f"Scenario dir: {CURRENT_SCENARIO_DIR}")
        print(f"Roadnet file: {CURRENT_SCENARIO_DIR / roadnet_name}")
        print(f"Flow file: {CURRENT_SCENARIO_DIR / flow_name}")
        print(f"Intersection id: {get_controlled_intersection_id(load_roadnet())}")
        print(f"Lane ids: {build_lane_ids()}")
        print(f"Phase ids: {build_phase_ids()}")
        print(f"Results dir: {results_dir}")

        q_table, training_rows = train_q_learning(results_dir)

        save_rows(training_rows, results_dir / "training.csv")
        save_q_table(q_table, results_dir / "q_table.pkl")

        summary = run_q_learning_episode(results_dir, q_table)
        summary_row = {"dataset": dataset_dir.name, **summary}
        all_summaries.append(summary_row)

        print("\nFinal summary:")
        print(summary_row)

        print("\nSaved files:")
        print(f"  {results_dir / 'runtime_config.json'}")
        print(f"  {results_dir / 'training.csv'}")
        print(f"  {results_dir / 'q_table.pkl'}")

    if all_summaries:
        summary_path = Path("results") / EXPERIMENT_NAME / "summary_all_datasets.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        save_rows(all_summaries, summary_path)
        print("\nSaved cross-dataset summary:")
        print(f"  {summary_path}")


if __name__ == "__main__":
    main()