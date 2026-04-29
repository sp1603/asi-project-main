from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.cityflow_env import CityFlowEnv


def load_config(dataset_dir: Path) -> dict[str, Any]:
    with (dataset_dir / "config.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_roadnet(dataset_dir: Path) -> dict[str, Any]:
    config = load_config(dataset_dir)
    roadnet_path = dataset_dir / Path(config["roadnetFile"]).name
    with roadnet_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_controlled_intersection_id(roadnet: dict[str, Any]) -> str:
    for inter in roadnet["intersections"]:
        if not inter.get("virtual", False):
            return inter["id"]
    raise ValueError("No non-virtual intersection found in roadnet file")


def build_lane_ids(dataset_dir: Path) -> list[str]:
    roadnet = load_roadnet(dataset_dir)
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


def build_phase_ids(dataset_dir: Path) -> list[int]:
    roadnet = load_roadnet(dataset_dir)
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


def build_phase_to_bucket(dataset_dir: Path) -> dict[int, int]:
    phase_ids = build_phase_ids(dataset_dir)
    return {phase_id: idx for idx, phase_id in enumerate(phase_ids)}


def build_runtime_config(dataset_dir: Path, out_dir: Path) -> Path:
    config = load_config(dataset_dir)

    runtime_config = dict(config)
    runtime_config["dir"] = str(dataset_dir.resolve()) + "/"
    runtime_config["roadnetFile"] = Path(config["roadnetFile"]).name
    runtime_config["flowFile"] = Path(config["flowFile"]).name
    runtime_config["roadnetLogFile"] = "replay_roadnet.json"
    runtime_config["replayLogFile"] = "replay.txt"
    runtime_config["saveReplay"] = False

    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_config_path = out_dir / "runtime_config.json"
    with runtime_config_path.open("w", encoding="utf-8") as f:
        json.dump(runtime_config, f, indent=2)

    return runtime_config_path


def aggregate_by_direction(obs: np.ndarray) -> np.ndarray:
    if obs.shape[0] % 4 != 0:
        raise ValueError(
            f"Expected observation length divisible by 4, got {obs.shape[0]}"
        )
    lanes_per_direction = obs.shape[0] // 4
    return obs.reshape(4, lanes_per_direction).sum(axis=1)


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


class CityFlowSB3Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset_dir: str,
        runtime_config_dir: str,
        state_mode: str = "capped_no_phase",
        reward_mode: str = "env",
    ) -> None:
        super().__init__()

        self.dataset_dir = Path(dataset_dir)
        self.runtime_config_dir = Path(runtime_config_dir)
        self.state_mode = state_mode
        self.reward_mode = reward_mode

        roadnet = load_roadnet(self.dataset_dir)
        self.intersection_id = get_controlled_intersection_id(roadnet)
        self.lane_ids = build_lane_ids(self.dataset_dir)
        self.phase_ids = build_phase_ids(self.dataset_dir)
        self.phase_to_bucket = build_phase_to_bucket(self.dataset_dir)

        runtime_config_path = build_runtime_config(
            self.dataset_dir, self.runtime_config_dir
        )

        self._env = CityFlowEnv(
            config_path=str(runtime_config_path),
            intersection_id=self.intersection_id,
            lane_ids=self.lane_ids,
            phase_ids=self.phase_ids,
            action_interval=5,
            max_steps=720,
            render_mode=None,
        )

        self.prev_total_waiting: float | None = None

        raw_obs, info = self._env.reset()
        wrapped_obs = self._transform_obs(raw_obs, info)

        self.action_space = spaces.Discrete(self._env.action_space.n)
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=wrapped_obs.shape,
            dtype=np.float32,
        )

    def _transform_obs(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)

        if self.state_mode == "capped_no_phase":
            return np.minimum(obs, 10).astype(np.float32)

        if self.state_mode == "capped_with_phase":
            capped = np.minimum(obs, 10).astype(np.float32)
            phase_bucket = np.array(
                [float(self.phase_to_bucket[int(info["phase_id"])])],
                dtype=np.float32,
            )
            return np.concatenate([capped, phase_bucket])

        if self.state_mode == "bucketed_no_phase":
            directional_totals = aggregate_by_direction(obs)
            buckets = np.array(
                [bucket_queue_length(float(x)) for x in directional_totals],
                dtype=np.float32,
            )
            return buckets

        if self.state_mode == "bucketed_with_phase":
            directional_totals = aggregate_by_direction(obs)
            buckets = np.array(
                [bucket_queue_length(float(x)) for x in directional_totals],
                dtype=np.float32,
            )
            phase_bucket = np.array(
                [float(self.phase_to_bucket[int(info["phase_id"])])],
                dtype=np.float32,
            )
            return np.concatenate([buckets, phase_bucket])

        raise ValueError(f"Unknown state_mode: {self.state_mode}")

    def _compute_reward(self, env_reward: float, info: dict[str, Any]) -> float:
        total_waiting = float(info["total_waiting"])
        vehicle_count = float(info["vehicle_count"])

        if self.reward_mode == "env":
            reward = float(env_reward)
        elif self.reward_mode == "wait_only":
            reward = -total_waiting
        elif self.reward_mode == "wait_plus_vehicle":
            reward = -(total_waiting + 0.1 * vehicle_count)
        elif self.reward_mode == "delta_wait":
            if self.prev_total_waiting is None:
                reward = 0.0
            else:
                reward = self.prev_total_waiting - total_waiting
        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

        self.prev_total_waiting = total_waiting
        return float(reward)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        obs, info = self._env.reset()
        self.prev_total_waiting = float(info["total_waiting"])
        wrapped_obs = self._transform_obs(obs, info)
        return wrapped_obs, info

    def step(self, action: int):
        obs, env_reward, terminated, truncated, info = self._env.step(int(action))
        wrapped_obs = self._transform_obs(obs, info)
        wrapped_reward = self._compute_reward(float(env_reward), info)
        return (
            wrapped_obs,
            float(wrapped_reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def render(self):
        return None

    def close(self):
        self._env.close()