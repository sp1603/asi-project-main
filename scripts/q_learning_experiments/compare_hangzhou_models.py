from __future__ import annotations

import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np

from env.cityflow_env import CityFlowEnv


DATASETS_DIR = Path("CityFlow/hangzhou_datasets")
COMPARISON_NAME = "hangzhou_comparison"

MAX_CARS_PER_DIRECTION = 10
FIXED_HOLD_STEPS = 3

CURRENT_DATASET_DIR: Path | None = None


def get_dataset_dir() -> Path:
    if CURRENT_DATASET_DIR is None:
        raise ValueError("CURRENT_DATASET_DIR has not been set.")
    return CURRENT_DATASET_DIR


def load_config() -> dict:
    with (get_dataset_dir() / "config.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_roadnet() -> dict:
    config = load_config()
    roadnet_name = Path(config["roadnetFile"]).name
    roadnet_path = get_dataset_dir() / roadnet_name

    with roadnet_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_runtime_config(results_dir: Path) -> Path:
    dataset_dir = get_dataset_dir()
    config = load_config()

    roadnet_name = Path(config["roadnetFile"]).name
    flow_name = Path(config["flowFile"]).name

    roadnet_path = dataset_dir / roadnet_name
    flow_path = dataset_dir / flow_name

    if not roadnet_path.exists():
        raise FileNotFoundError(f"Missing roadnet file: {roadnet_path}")
    if not flow_path.exists():
        raise FileNotFoundError(f"Missing flow file: {flow_path}")

    runtime_config = dict(config)
    runtime_config["dir"] = str(dataset_dir.resolve()) + "/"
    runtime_config["roadnetFile"] = roadnet_name
    runtime_config["flowFile"] = flow_name

    runtime_config["roadnetLogFile"] = "replay_roadnet.json"
    runtime_config["replayLogFile"] = "replay.txt"
    runtime_config["saveReplay"] = False

    runtime_config_path = results_dir / "runtime_config.json"
    with runtime_config_path.open("w", encoding="utf-8") as f:
        json.dump(runtime_config, f, indent=2)

    return runtime_config_path


def get_controlled_intersection_id(roadnet: dict) -> str:
    for inter in roadnet["intersections"]:
        if not inter.get("virtual", False):
            return inter["id"]
    raise ValueError("No non-virtual intersection found")


def extract_lane_ids(roadnet: dict, intersection_id: str) -> list[str]:
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


def extract_phase_ids(roadnet: dict, intersection_id: str) -> list[int]:
    intersection = next(i for i in roadnet["intersections"] if i["id"] == intersection_id)

    phase_ids: list[int] = []
    for idx, phase in enumerate(intersection["trafficLight"]["lightphases"]):
        if phase.get("availableRoadLinks"):
            phase_ids.append(idx)

    if not phase_ids:
        raise ValueError("No valid traffic-light phases found")

    return phase_ids


def get_intersection_id() -> str:
    return get_controlled_intersection_id(load_roadnet())


def get_lane_ids() -> list[str]:
    roadnet = load_roadnet()
    return extract_lane_ids(roadnet, get_controlled_intersection_id(roadnet))


def get_phase_ids() -> list[int]:
    roadnet = load_roadnet()
    return extract_phase_ids(roadnet, get_controlled_intersection_id(roadnet))


def get_phase_to_index() -> dict[int, int]:
    phase_ids = get_phase_ids()
    return {p: i for i, p in enumerate(phase_ids)}


MODEL_SPECS = [
    {"name": "bucketed_no_phase_default", "experiment_dir": "q_learning_bucketed_no_phase_default_reward", "state_type": "bucketed_no_phase"},
    {"name": "bucketed_no_phase_delta_wait", "experiment_dir": "q_learning_bucketed_no_phase_delta_wait", "state_type": "bucketed_no_phase"},
    {"name": "bucketed_no_phase_wait_small_vehicle", "experiment_dir": "q_learning_bucketed_no_phase_wait_small_vehicle", "state_type": "bucketed_no_phase"},
    {"name": "bucketed_no_phase_wait_vehicle", "experiment_dir": "q_learning_bucketed_no_phase_wait_vehicle", "state_type": "bucketed_no_phase"},
    {"name": "capped_no_phase_default_reward", "experiment_dir": "q_learning_capped_no_phase_default_reward", "state_type": "capped_no_phase"},
    {"name": "capped_no_phase_delta_wait", "experiment_dir": "q_learning_capped_no_phase_delta_wait", "state_type": "capped_no_phase"},
    {"name": "capped_no_phase_wait_small_vehicle", "experiment_dir": "q_learning_capped_no_phase_wait_small_vehicle", "state_type": "capped_no_phase"},
    {"name": "capped_no_phase_wait_vehicle", "experiment_dir": "q_learning_capped_no_phase_wait_vehicle", "state_type": "capped_no_phase"},
    {"name": "bucketed_with_phase_default_reward", "experiment_dir": "q_learning_bucketed_with_phase_default_reward", "state_type": "bucketed_with_phase"},
    {"name": "bucketed_with_phase_delta_wait", "experiment_dir": "q_learning_bucketed_with_phase_delta_wait", "state_type": "bucketed_with_phase"},
    {"name": "bucketed_with_phase_wait_small_vehicle", "experiment_dir": "q_learning_bucketed_with_phase_wait_small_vehicle", "state_type": "bucketed_with_phase"},
    {"name": "bucketed_with_phase_wait_vehicle", "experiment_dir": "q_learning_bucketed_with_phase_wait_vehicle", "state_type": "bucketed_with_phase"},
    {"name": "capped_with_phase_default_reward", "experiment_dir": "q_learning_capped_with_phase_default_reward", "state_type": "capped_with_phase"},
    {"name": "capped_with_phase_delta_wait", "experiment_dir": "q_learning_capped_with_phase_delta_wait", "state_type": "capped_with_phase"},
    {"name": "capped_with_phase_wait_small_vehicle", "experiment_dir": "q_learning_capped_with_phase_wait_small_vehicle", "state_type": "capped_with_phase"},
    {"name": "capped_with_phase_wait_vehicle", "experiment_dir": "q_learning_capped_with_phase_wait_vehicle", "state_type": "capped_with_phase"},
]


def get_q_table_path(dataset_name: str, experiment_dir: str) -> Path:
    return Path("results") / dataset_name / experiment_dir / "q_table.pkl"


def make_env(results_dir: Path) -> CityFlowEnv:
    runtime_config_path = build_runtime_config(results_dir)
    return CityFlowEnv(
        config_path=str(runtime_config_path),
        intersection_id=get_intersection_id(),
        lane_ids=get_lane_ids(),
        phase_ids=get_phase_ids(),
        action_interval=5,
        max_steps=720,
        render_mode=None,
    )


def aggregate(obs: np.ndarray) -> np.ndarray:
    return obs.reshape(4, -1).sum(axis=1)


def bucket(x: float) -> int:
    if x <= 0:
        return 0
    if x <= 2:
        return 1
    if x <= 5:
        return 2
    if x <= 10:
        return 3
    return 4


def state_bucketed_no_phase(obs: np.ndarray, info: dict):
    return tuple(bucket(float(x)) for x in aggregate(obs))


def state_bucketed_with_phase(obs: np.ndarray, info: dict):
    phase_to_index = get_phase_to_index()
    return tuple(bucket(float(x)) for x in aggregate(obs)) + (phase_to_index[info["phase_id"]],)


def state_capped_no_phase(obs: np.ndarray, info: dict):
    return tuple(
        int(x) for x in np.clip(np.rint(aggregate(obs)).astype(int), 0, MAX_CARS_PER_DIRECTION)
    )


def state_capped_with_phase(obs: np.ndarray, info: dict):
    phase_to_index = get_phase_to_index()
    return tuple(
        int(x) for x in np.clip(np.rint(aggregate(obs)).astype(int), 0, MAX_CARS_PER_DIRECTION)
    ) + (phase_to_index[info["phase_id"]],)


STATE_FN_MAP: dict[str, Callable] = {
    "bucketed_no_phase": state_bucketed_no_phase,
    "bucketed_with_phase": state_bucketed_with_phase,
    "capped_no_phase": state_capped_no_phase,
    "capped_with_phase": state_capped_with_phase,
}


def eval_reward(info: dict) -> float:
    return -float(info["total_waiting"])

def load_q(path: Path):
    with path.open("rb") as f:
        raw = pickle.load(f)

    n = len(next(iter(raw.values())))
    q = defaultdict(lambda: np.zeros(n, dtype=np.float32))
    for k, v in raw.items():
        q[k] = np.array(v, dtype=np.float32)
    return q


def run_q(env: CityFlowEnv, q, state_fn: Callable, name: str):
    obs, info = env.reset()
    s = state_fn(obs, info)

    waits: list[float] = []
    travel: list[float] = []
    total = 0.0

    done = False
    while not done:
        a = int(np.argmax(q[s]))
        obs, _, term, trunc, info = env.step(a)

        total += eval_reward(info)
        waits.append(float(info["total_waiting"]))
        travel.append(float(info["avg_travel_time"]))

        s = state_fn(obs, info)
        done = term or trunc

    return {
        "method": name,
        "eval_wait_only_reward": total,
        "avg_waiting": float(np.mean(waits)),
        "final_waiting": float(waits[-1]),
        "avg_travel_time": float(np.mean(travel)),
    }


def run_fixed(env: CityFlowEnv):
    _, info = env.reset()

    phase_ids = get_phase_ids()

    waits: list[float] = []
    travel: list[float] = []
    total = 0.0
    step = 0

    done = False
    while not done:
        action = (step // FIXED_HOLD_STEPS) % len(phase_ids)

        _, _, term, trunc, info = env.step(action)

        total += eval_reward(info)
        waits.append(float(info["total_waiting"]))
        travel.append(float(info["avg_travel_time"]))

        done = term or trunc
        step += 1

    return {
        "method": "fixed_cycle",
        "eval_wait_only_reward": total,
        "avg_waiting": float(np.mean(waits)),
        "final_waiting": float(waits[-1]),
        "avg_travel_time": float(np.mean(travel)),
    }


def main():
    global CURRENT_DATASET_DIR

    dataset_dirs = sorted(d for d in DATASETS_DIR.iterdir() if d.is_dir())
    if not dataset_dirs:
        raise ValueError(f"No dataset folders found in {DATASETS_DIR}")

    all_results: list[dict] = []

    for dataset_dir in dataset_dirs:
        CURRENT_DATASET_DIR = dataset_dir
        dataset_name = dataset_dir.name

        comparison_dir = Path("results") / dataset_name / COMPARISON_NAME
        comparison_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print(f"Dataset: {dataset_name}")
        print(f"Config path: {get_dataset_dir() / 'config.json'}")
        print(f"Intersection id: {get_intersection_id()}")
        print(f"Lane ids: {get_lane_ids()}")
        print(f"Phase ids: {get_phase_ids()}")

        dataset_results: list[dict] = []

        env = make_env(comparison_dir)
        fixed_result = run_fixed(env)
        env.close()
        dataset_results.append({"dataset": dataset_name, **fixed_result})

        for spec in MODEL_SPECS:
            q_table_path = get_q_table_path(dataset_name, spec["experiment_dir"])
            if not q_table_path.exists():
                print(f"Skipping {spec['name']} (missing {q_table_path})")
                continue

            q = load_q(q_table_path)
            state_fn = STATE_FN_MAP[spec["state_type"]]

            env = make_env(comparison_dir)
            result = run_q(env, q, state_fn, spec["name"])
            env.close()

            dataset_results.append({"dataset": dataset_name, **result})

        print("\nComparison:")
        for r in dataset_results:
            print(
                f"{r['method']:40s} | "
                f"reward={r['eval_wait_only_reward']:10.2f} | "
                f"avg_wait={r['avg_waiting']:8.2f} | "
                f"final_wait={r['final_waiting']:8.2f} | "
                f"travel_time={r['avg_travel_time']:8.2f}"
            )

        with (comparison_dir / "comparison_single_run.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=dataset_results[0].keys())
            writer.writeheader()
            writer.writerows(dataset_results)

        all_results.extend(dataset_results)

    combined_dir = Path("results") / COMPARISON_NAME
    combined_dir.mkdir(parents=True, exist_ok=True)

    with (combined_dir / "comparison_all_datasets.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    print("\nSaved combined comparison:")
    print(f"  {combined_dir / 'comparison_all_datasets.csv'}")


if __name__ == "__main__":
    main()