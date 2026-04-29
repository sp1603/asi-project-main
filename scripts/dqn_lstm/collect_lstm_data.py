from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np

from env.cityflow_env import CityFlowEnv


DATASET_NAME = "hangzhou_1x1_bc-tyc_18041607_1h"
DATASET_DIR = Path("CityFlow/hangzhou_datasets") / DATASET_NAME
CONFIG_PATH = DATASET_DIR / "config.json"
ROADNET_PATH = DATASET_DIR / "roadnet.json"
INTERSECTION_ID = "intersection_1_1"

OUT_PATH = Path(f"data/{DATASET_NAME}_lstm_data.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def extract_lane_ids_from_roadnet(roadnet_path: str | Path, intersection_id: str) -> list[str]:
    with open(roadnet_path, "r", encoding="utf-8") as f:
        roadnet = json.load(f)

    incoming_lanes: list[str] = []

    for road in roadnet["roads"]:
        if road["endIntersection"] == intersection_id:
            road_id = road["id"]
            for i in range(len(road["lanes"])):
                incoming_lanes.append(f"{road_id}_{i}")

    return incoming_lanes


def get_intersection_lightphases(
    roadnet_path: str | Path, intersection_id: str
) -> list[dict]:
    with open(roadnet_path, "r", encoding="utf-8") as f:
        roadnet = json.load(f)

    for inter in roadnet["intersections"]:
        if inter["id"] == intersection_id:
            return inter["trafficLight"]["lightphases"]

    raise ValueError(f"Intersection {intersection_id} not found in {roadnet_path}")


def choose_two_main_phases(
    roadnet_path: str | Path, intersection_id: str
) -> list[int]:
    lightphases = get_intersection_lightphases(roadnet_path, intersection_id)

    candidates: list[tuple[int, set[int]]] = []
    for phase_id, phase in enumerate(lightphases):
        links = set(phase.get("availableRoadLinks", []))
        if links:
            candidates.append((phase_id, links))

    if len(candidates) < 2:
        raise ValueError(
            f"Need at least 2 non-empty phases, found {len(candidates)} at {intersection_id}"
        )

    best_pair: tuple[int, int] | None = None
    best_score = -1

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            pid1, links1 = candidates[i]
            pid2, links2 = candidates[j]

            overlap = len(links1 & links2)
            if overlap != 0:
                continue

            score = len(links1) + len(links2)
            if score > best_score:
                best_score = score
                best_pair = (pid1, pid2)

    if best_pair is None:
        raise ValueError(
            "Could not find 2 non-overlapping green phases automatically."
        )

    return list(best_pair)


PHASE_IDS = choose_two_main_phases(ROADNET_PATH, INTERSECTION_ID)
PHASE_MAP = {phase_id: idx for idx, phase_id in enumerate(PHASE_IDS)}


def make_env() -> CityFlowEnv:
    lane_ids = extract_lane_ids_from_roadnet(ROADNET_PATH, INTERSECTION_ID)

    return CityFlowEnv(
        config_path=str(CONFIG_PATH),
        intersection_id=INTERSECTION_ID,
        lane_ids=lane_ids,
        phase_ids=PHASE_IDS,
        action_interval=5,
        max_steps=200,
        render_mode=None,
    )


def aggregate_by_direction(obs: np.ndarray) -> np.ndarray:
    if obs.shape[0] != 8:
        raise ValueError(f"Expected observation length 8, got {obs.shape[0]}")
    grouped = obs.reshape(4, 2)
    return grouped.sum(axis=1)


def phase_to_index(phase_id: int) -> int:
    if phase_id not in PHASE_MAP:
        raise ValueError(
            f"Unexpected phase_id {phase_id}. Expected one of {list(PHASE_MAP.keys())}"
        )
    return PHASE_MAP[phase_id]


def choose_action(policy: str, step_idx: int, n_actions: int) -> int:
    if policy == "random":
        return random.randrange(n_actions)
    if policy == "alternating":
        return step_idx % n_actions
    raise ValueError(f"Unknown policy: {policy}")


def collect_episode(env: CityFlowEnv, episode_id: int, policy: str = "random") -> list[dict]:
    rows: list[dict] = []

    obs, info = env.reset()
    done = False
    step_idx = 0

    while not done:
        directional = aggregate_by_direction(np.asarray(obs, dtype=np.float32))

        rows.append(
            {
                "episode": episode_id,
                "step": step_idx,
                "time": float(info["time"]),
                "dir_0": float(directional[0]),
                "dir_1": float(directional[1]),
                "dir_2": float(directional[2]),
                "dir_3": float(directional[3]),
                "phase": int(phase_to_index(int(info["phase_id"]))),
                "total_waiting": float(info["total_waiting"]),
                "vehicle_count": float(info["vehicle_count"]),
                "avg_travel_time": float(info["avg_travel_time"]),
            }
        )

        action = choose_action(policy, step_idx, env.action_space.n)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_idx += 1

    return rows


def save_rows(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    random.seed(42)
    np.random.seed(42)

    episodes = 300
    policy = "random"

    print(f"Using dataset: {DATASET_DIR}")
    print(f"Using phase_ids: {PHASE_IDS}")
    print(f"Phase map: {PHASE_MAP}")

    lane_ids = extract_lane_ids_from_roadnet(ROADNET_PATH, INTERSECTION_ID)
    print(f"Lane ids: {lane_ids}")

    all_rows: list[dict] = []

    for episode_id in range(1, episodes + 1):
        env = make_env()
        rows = collect_episode(env, episode_id=episode_id, policy=policy)
        env.close()

        all_rows.extend(rows)

        if episode_id % 25 == 0:
            print(f"Collected {episode_id} / {episodes} episodes")

    save_rows(all_rows, OUT_PATH)
    print(f"Saved {len(all_rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()