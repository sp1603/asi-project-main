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
EXPERIMENT_NAME = "q_learning_capped_no_phase_default_reward"
CURRENT_SCENARIO_DIR: Path | None = None

MAX_CARS_PER_DIRECTION = 10


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
    raise ValueError("No non-virtual intersection found in roadnet file")


def build_lane_ids() -> list[str]:
    roadnet = load_roadnet()
    intersection_id = get_controlled_intersection_id(roadnet)

    roads_by_id = {road["id"]: road for road in roadnet["roads"]}
    intersection = next(
        inter for inter in roadnet["intersections"] if inter["id"] == intersection_id
    )

    incoming_lane_ids: list[str] = []

    for road_id in intersection["roads"]:
        road = roads_by_id[road_id]
        if road["endIntersection"] != intersection_id:
            continue

        for i in range(len(road["lanes"])):
            incoming_lane_ids.append(f"{road_id}_{i}")

    incoming_lane_ids.sort()
    return incoming_lane_ids


def build_phase_ids() -> list[int]:
    roadnet = load_roadnet()
    intersection_id = get_controlled_intersection_id(roadnet)

    intersection = next(
        inter for inter in roadnet["intersections"] if inter["id"] == intersection_id
    )

    phase_ids: list[int] = []
    for idx, phase in enumerate(intersection["trafficLight"]["lightphases"]):
        if phase.get("availableRoadLinks"):
            phase_ids.append(idx)

    if not phase_ids:
        raise ValueError("No valid traffic-light phases found.")
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
    phase_ids = build_phase_ids()

    return CityFlowEnv(
        config_path=str(runtime_config_path),
        intersection_id=intersection_id,
        lane_ids=build_lane_ids(),
        phase_ids=phase_ids,
        action_interval=5,
        max_steps=720,
        render_mode=None,
    )


def aggregate_by_direction(obs: np.ndarray) -> np.ndarray:
    if obs.shape[0] % 4 != 0:
        raise ValueError(
            f"Expected observation length divisible by 4, got {obs.shape[0]}"
        )
    lanes_per_direction = obs.shape[0] // 4
    return obs.reshape(4, lanes_per_direction).sum(axis=1)


def discretize_state(
    obs: np.ndarray,
    max_cars_per_direction: int = MAX_CARS_PER_DIRECTION,
) -> tuple[int, int, int, int]:
    directional_totals = aggregate_by_direction(obs)
    capped = np.clip(np.rint(directional_totals).astype(int), 0, max_cars_per_direction)
    return tuple(int(x) for x in capped)


def reward_wait_only(info: dict, prev_info: dict | None) -> float:
    return -float(info["total_waiting"])


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


def save_q_table(
    q_table: DefaultDict[tuple[int, int, int, int], np.ndarray],
    out_path: Path,
) -> None:
    serializable_q_table = {state: values.copy() for state, values in q_table.items()}
    with out_path.open("wb") as f:
        pickle.dump(serializable_q_table, f)


def train_q_learning(
    results_dir: Path,
    episodes: int = 2000,
    alpha: float = 0.10,
    gamma: float = 0.95,
    epsilon: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.997,
) -> tuple[DefaultDict[tuple[int, int, int, int], np.ndarray], list[dict]]:
    sample_env = make_env(results_dir)
    n_actions = sample_env.action_space.n
    sample_env.close()

    q_table: DefaultDict[tuple[int, int, int, int], np.ndarray] = defaultdict(
        lambda: np.zeros(n_actions, dtype=np.float32)
    )
    training_rows: list[dict] = []

    for episode in range(episodes):
        env = make_env(results_dir)
        obs, info = env.reset()
        state = discretize_state(obs)

        done = False
        total_reward = 0.0
        rewards_this_episode: list[float] = []
        waits_this_episode: list[float] = []
        last_info = info

        while not done:
            action = epsilon_greedy_action(q_table, state, n_actions, epsilon)
            next_obs, _, terminated, truncated, next_info = env.step(action)
            reward = reward_wait_only(next_info, info)
            done = terminated or truncated

            next_state = discretize_state(next_obs)

            if done:
                td_target = reward
            else:
                td_target = reward + gamma * float(np.max(q_table[next_state]))

            td_error = td_target - float(q_table[state][action])
            q_table[state][action] += alpha * td_error

            state = next_state
            total_reward += reward
            rewards_this_episode.append(reward)
            waits_this_episode.append(float(next_info["total_waiting"]))
            last_info = next_info
            info = next_info

        env.close()
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        training_rows.append(
            {
                "episode": episode + 1,
                "total_reward": total_reward,
                "mean_reward_per_decision": float(np.mean(rewards_this_episode)),
                "final_waiting": float(last_info["total_waiting"]),
                "avg_waiting": float(np.mean(waits_this_episode)),
                "final_avg_travel_time": float(last_info["avg_travel_time"]),
                "epsilon": epsilon,
            }
        )

        if (episode + 1) % 50 == 0:
            recent = training_rows[-50:]
            print(
                f"Episode {episode + 1:4d} | "
                f"Last50 Mean Reward: {np.mean([r['total_reward'] for r in recent]):9.2f} | "
                f"Last50 Mean Final Waiting: {np.mean([r['final_waiting'] for r in recent]):7.2f} | "
                f"Last50 Mean Travel Time: {np.mean([r['final_avg_travel_time'] for r in recent]):7.2f} | "
                f"Epsilon: {epsilon:.3f}"
            )

    return q_table, training_rows


def run_q_learning_episode(
    results_dir: Path,
    q_table: DefaultDict[tuple[int, int, int, int], np.ndarray],
) -> tuple[dict, list[dict]]:
    env = make_env(results_dir)
    obs, info = env.reset()
    state = discretize_state(obs)

    done = False
    decision_step = 0
    total_reward = 0.0
    trace_rows: list[dict] = []
    prev_info: dict | None = None

    while not done:
        action = int(np.argmax(q_table[state]))
        obs, _, terminated, truncated, info = env.step(action)
        reward = reward_wait_only(info, prev_info)
        state = discretize_state(obs)
        total_reward += reward

        trace_rows.append(
            {
                "decision_step": decision_step,
                "time": info["time"],
                "action": action,
                "phase_id": info["phase_id"],
                "state_north_count": state[0],
                "state_east_count": state[1],
                "state_south_count": state[2],
                "state_west_count": state[3],
                "reward": reward,
                "total_waiting": info["total_waiting"],
                "vehicle_count": info["vehicle_count"],
                "avg_travel_time": info["avg_travel_time"],
            }
        )

        prev_info = info
        done = terminated or truncated
        decision_step += 1

    env.close()

    summary = {
        "method": EXPERIMENT_NAME,
        "total_reward": total_reward,
        "final_waiting": float(info["total_waiting"]),
        "avg_waiting": float(np.mean([r["total_waiting"] for r in trace_rows])),
        "final_avg_travel_time": float(info["avg_travel_time"]),
        "num_decisions": len(trace_rows),
    }
    return summary, trace_rows


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
        phase_ids = build_phase_ids()

        print("\n" + "=" * 60)
        print("Training Q-learning with capped counts, no phase, default reward...")
        print(f"Scenario dir: {CURRENT_SCENARIO_DIR}")
        print(f"Roadnet file: {CURRENT_SCENARIO_DIR / roadnet_name}")
        print(f"Flow file: {CURRENT_SCENARIO_DIR / flow_name}")
        print(f"Intersection id: {get_controlled_intersection_id(load_roadnet())}")
        print(f"Lane ids: {build_lane_ids()}")
        print(f"Phase ids: {phase_ids}")
        print(f"Results dir: {results_dir}")

        q_table, training_rows = train_q_learning(results_dir)

        save_rows(training_rows, results_dir / "q_learning_training.csv")
        save_q_table(q_table, results_dir / "q_table.pkl")

        q_summary, q_trace = run_q_learning_episode(results_dir, q_table)
        save_rows(q_trace, results_dir / "q_learning_trace.csv")

        summary_row = {"dataset": dataset_dir.name, **q_summary}
        all_summaries.append(summary_row)

        print("\nSingle Q-learning summary:")
        print(summary_row)

        print("\nSaved files:")
        print(f"  {results_dir / 'runtime_config.json'}")
        print(f"  {results_dir / 'q_learning_training.csv'}")
        print(f"  {results_dir / 'q_table.pkl'}")
        print(f"  {results_dir / 'q_learning_trace.csv'}")

    if all_summaries:
        summary_path = Path("results") / EXPERIMENT_NAME / "summary_all_datasets.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        save_rows(all_summaries, summary_path)
        print("\nSaved cross-dataset summary:")
        print(f"  {summary_path}")


if __name__ == "__main__":
    main()