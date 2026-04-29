from __future__ import annotations

import json
import random
from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from env.cityflow_env import CityFlowEnv

DATASET_NAME = "hangzhou_1x1_bc-tyc_18041607_1h"
DATASET_DIR = Path("CityFlow/hangzhou_datasets") / DATASET_NAME
CONFIG_PATH = DATASET_DIR / "config.json"
ROADNET_PATH = DATASET_DIR / "roadnet.json"
INTERSECTION_ID = "intersection_1_1"

LSTM_MODEL_PATH = Path("models/hangzhou_1x1_lstm.pt")
OUTPUT_DIR = Path(f"results/dqn_lstm_{DATASET_NAME}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_CARS_PER_DIRECTION = 10


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


def aggregate_by_direction(obs: np.ndarray) -> np.ndarray:
    if obs.shape[0] != 8:
        raise ValueError(f"Expected observation length 8, got {obs.shape[0]}")
    grouped = obs.reshape(4, 2)
    return grouped.sum(axis=1)


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
        raise ValueError("Could not find 2 non-overlapping green phases automatically.")

    return list(best_pair)


PHASE_IDS = choose_two_main_phases(ROADNET_PATH, INTERSECTION_ID)
PHASE_MAP = {phase_id: idx for idx, phase_id in enumerate(PHASE_IDS)}


def phase_to_index(phase_id: int) -> int:
    if phase_id not in PHASE_MAP:
        raise ValueError(
            f"Unexpected phase_id {phase_id}. Expected one of {list(PHASE_MAP.keys())}"
        )
    return PHASE_MAP[phase_id]


def make_base_env() -> CityFlowEnv:
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


class TrafficLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden)


class LSTMPredictor:
    def __init__(self, checkpoint_path: str | Path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        cfg: dict[str, Any] = checkpoint["config"]
        stats: dict[str, Any] = checkpoint["stats"]

        self.history = int(cfg["history"])
        self.input_dim = int(cfg["input_dim"])
        self.output_dim = int(cfg["output_dim"])

        self.mean = np.asarray(stats["mean"], dtype=np.float32)
        self.std = np.asarray(stats["std"], dtype=np.float32)       

        self.model = TrafficLSTM(
            input_dim=int(cfg["input_dim"]),
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg["num_layers"]),
            output_dim=int(cfg["output_dim"]),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, history_array: np.ndarray) -> np.ndarray:
        if history_array.shape != (self.history, self.input_dim):
            raise ValueError(
                f"Expected history shape {(self.history, self.input_dim)}, got {history_array.shape}"
            )

        x = history_array.copy()

        x[:, :4] = (x[:, :4] - self.mean.reshape(4)) / self.std.reshape(4)

        x_tensor = torch.tensor(x[None, :, :], dtype=torch.float32)
        with torch.no_grad():
            pred_norm = self.model(x_tensor).cpu().numpy()[0]

        pred = pred_norm * self.std.reshape(4) + self.mean.reshape(4)
        pred = np.maximum(pred, 0.0)
        return pred.astype(np.float32)


class DQNLSTMEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        checkpoint_path: str | Path = LSTM_MODEL_PATH,
        max_cars_per_direction: int = MAX_CARS_PER_DIRECTION,
    ) -> None:
        super().__init__()

        self.base_env = make_base_env()
        self.predictor = LSTMPredictor(checkpoint_path)
        self.max_cars_per_direction = max_cars_per_direction

        self.action_space = self.base_env.action_space

        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(9,),
            dtype=np.float32,
        )

        self.history_buffer: deque[np.ndarray] = deque(maxlen=self.predictor.history)

    def _build_feature_vector(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        directional = aggregate_by_direction(np.asarray(obs, dtype=np.float32))
        directional = np.minimum(directional, self.max_cars_per_direction).astype(np.float32)
        phase_idx = np.float32(phase_to_index(int(info["phase_id"])))
        feature = np.concatenate([directional, np.array([phase_idx], dtype=np.float32)])
        return feature

    def _predict_future_counts(self) -> np.ndarray:
        if len(self.history_buffer) < self.predictor.history:
            latest = self.history_buffer[-1][:4]
            return latest.astype(np.float32)

        history_array = np.stack(self.history_buffer, axis=0)
        return self.predictor.predict(history_array)

    def _build_observation(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        feature = self._build_feature_vector(obs, info)
        current_counts = feature[:4]
        phase = feature[4:5]

        pred_counts = self._predict_future_counts()
        pred_counts = np.minimum(pred_counts, self.max_cars_per_direction)

        full_obs = np.concatenate([current_counts, phase, pred_counts]).astype(np.float32)
        return full_obs

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        obs, info = self.base_env.reset(seed=seed, options=options)

        self.history_buffer.clear()
        first_feature = self._build_feature_vector(obs, info)

        for _ in range(self.predictor.history):
            self.history_buffer.append(first_feature.copy())

        full_obs = self._build_observation(obs, info)
        return full_obs, info

    def step(self, action: int):
        obs, _, terminated, truncated, info = self.base_env.step(action)

        reward = -float(info["total_waiting"])

        feature = self._build_feature_vector(obs, info)
        self.history_buffer.append(feature)

        full_obs = self._build_observation(obs, info)
        return full_obs, reward, terminated, truncated, info

    def render(self):
        return self.base_env.render()

    def close(self):
        return self.base_env.close()


def make_training_env() -> gym.Env:
    env = DQNLSTMEnv()
    env = Monitor(env)
    return env


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print(f"Using dataset: {DATASET_DIR}")
    print(f"Using config: {CONFIG_PATH}")
    print(f"Using roadnet: {ROADNET_PATH}")
    print(f"Using phase_ids: {PHASE_IDS}")
    print(f"Using phase_map: {PHASE_MAP}")
    print(f"Using LSTM checkpoint: {LSTM_MODEL_PATH}")

    env = make_training_env()

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=2_000,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        device="auto",
    )

    total_timesteps = 50_000
    model.learn(total_timesteps=total_timesteps)

    model_path = OUTPUT_DIR / "dqn_lstm_model"
    model.save(str(model_path))

    print(f"Saved DQN model to {model_path}.zip")

    obs, info = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        done = terminated or truncated

    print("Greedy evaluation total_reward:", total_reward)
    print("Final waiting:", info["total_waiting"])
    print("Final avg travel time:", info["avg_travel_time"])

    env.close()


if __name__ == "__main__":
    main()