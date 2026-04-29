from __future__ import annotations

import json
import random
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import DQN

from env.cityflow_env import CityFlowEnv


DATASET_NAME = "hangzhou_1x1_bc-tyc_18041607_1h"
DATASET_DIR = Path("CityFlow/hangzhou_datasets") / DATASET_NAME
CONFIG_PATH = DATASET_DIR / "config.json"
ROADNET_PATH = DATASET_DIR / "roadnet.json"
INTERSECTION_ID = "intersection_1_1"

LSTM_CHECKPOINT_PATH = Path("models/hangzhou_1x1_lstm.pt")

SEQ_LEN = 12
LSTM_INPUT_SIZE = 5    
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_OUTPUT_SIZE = 4

MAX_STEPS = 200
ACTION_INTERVAL = 5

RESULTS_DIR = Path(f"results/{DATASET_NAME}/dqn_lstm_hidden")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def extract_lane_ids_from_roadnet(
    roadnet_path: str | Path,
    intersection_id: str,
) -> list[str]:
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
    roadnet_path: str | Path,
    intersection_id: str,
) -> list[dict]:
    with open(roadnet_path, "r", encoding="utf-8") as f:
        roadnet = json.load(f)

    for inter in roadnet["intersections"]:
        if inter["id"] == intersection_id:
            return inter["trafficLight"]["lightphases"]

    raise ValueError(f"Intersection {intersection_id} not found in {roadnet_path}")


def choose_two_main_phases(
    roadnet_path: str | Path,
    intersection_id: str,
) -> list[int]:
    lightphases = get_intersection_lightphases(roadnet_path, intersection_id)

    candidates: list[tuple[int, set[int]]] = []

    for phase_id, phase in enumerate(lightphases):
        links = set(phase.get("availableRoadLinks", []))
        if links:
            candidates.append((phase_id, links))

    if len(candidates) < 2:
        raise ValueError(
            f"Need at least 2 non-empty phases, found {len(candidates)}"
        )

    best_pair: tuple[int, int] | None = None
    best_score = -1

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            pid1, links1 = candidates[i]
            pid2, links2 = candidates[j]

            if len(links1 & links2) != 0:
                continue

            score = len(links1) + len(links2)

            if score > best_score:
                best_score = score
                best_pair = (pid1, pid2)

    if best_pair is None:
        raise ValueError("Could not find 2 non-overlapping green phases.")

    return list(best_pair)


PHASE_IDS = choose_two_main_phases(ROADNET_PATH, INTERSECTION_ID)
PHASE_MAP = {phase_id: idx for idx, phase_id in enumerate(PHASE_IDS)}


def aggregate_by_direction(obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)

    if obs.shape[0] != 8:
        raise ValueError(f"Expected observation length 8, got {obs.shape[0]}")

    return obs.reshape(4, 2).sum(axis=1).astype(np.float32)


def phase_to_index(phase_id: int) -> int:
    if phase_id not in PHASE_MAP:
        raise ValueError(
            f"Unexpected phase_id {phase_id}. Expected one of {list(PHASE_MAP.keys())}"
        )

    return PHASE_MAP[phase_id]


class TrafficLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = LSTM_INPUT_SIZE,
        hidden_dim: int = LSTM_HIDDEN_SIZE,
        num_layers: int = LSTM_NUM_LAYERS,
        output_dim: int = LSTM_OUTPUT_SIZE,
    ) -> None:
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)

        hidden_vector = out[:, -1, :]

        return hidden_vector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_vector = self.encode(x)
        return self.head(hidden_vector)


def load_lstm_model(device: torch.device) -> tuple[TrafficLSTM, dict]:
    checkpoint = torch.load(
        LSTM_CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    cfg = checkpoint["config"]

    model = TrafficLSTM(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        output_dim=cfg["output_dim"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    stats = checkpoint["stats"]

    return model, stats


def make_base_env() -> CityFlowEnv:
    lane_ids = extract_lane_ids_from_roadnet(ROADNET_PATH, INTERSECTION_ID)

    return CityFlowEnv(
        config_path=str(CONFIG_PATH),
        intersection_id=INTERSECTION_ID,
        lane_ids=lane_ids,
        phase_ids=PHASE_IDS,
        action_interval=ACTION_INTERVAL,
        max_steps=MAX_STEPS,
        render_mode=None,
    )


class DQNLSTMHiddenEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()

        self.base_env = make_base_env()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm_model, self.lstm_stats = load_lstm_model(self.device)

        self.history: deque[np.ndarray] = deque(maxlen=SEQ_LEN)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4 + 1 + LSTM_HIDDEN_SIZE,),
            dtype=np.float32,
        )

        self.action_space = self.base_env.action_space

    def _normalize_lstm_input(self, padded: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.lstm_stats["mean"], dtype=np.float32).reshape(-1)
        std = np.asarray(self.lstm_stats["std"], dtype=np.float32).reshape(-1)

        padded = padded.copy()
        padded[:, :4] = (padded[:, :4] - mean) / std

        return padded.astype(np.float32)

    def _history_to_tensor(self) -> torch.Tensor:
        if len(self.history) == 0:
            padded = np.zeros((SEQ_LEN, LSTM_INPUT_SIZE), dtype=np.float32)
        else:
            history_array = np.asarray(self.history, dtype=np.float32)

            if len(history_array) < SEQ_LEN:
                pad_len = SEQ_LEN - len(history_array)
                padding = np.zeros((pad_len, LSTM_INPUT_SIZE), dtype=np.float32)
                padded = np.vstack([padding, history_array])
            else:
                padded = history_array[-SEQ_LEN:]

        padded = self._normalize_lstm_input(padded)

        x = torch.tensor(
            padded,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        return x

    def _get_lstm_hidden_vector(self) -> np.ndarray:
        x = self._history_to_tensor()

        with torch.no_grad():
            hidden = self.lstm_model.encode(x)

        return hidden.squeeze(0).cpu().numpy().astype(np.float32)

    def _build_state(self, obs: np.ndarray, info: dict) -> np.ndarray:
        directional = aggregate_by_direction(np.asarray(obs, dtype=np.float32))
        phase = phase_to_index(int(info["phase_id"]))

        current_features = np.array(
            [
                directional[0],
                directional[1],
                directional[2],
                directional[3],
                float(phase),
            ],
            dtype=np.float32,
        )

        self.history.append(current_features)

        lstm_hidden = self._get_lstm_hidden_vector()

        dqn_state = np.concatenate(
            [
                directional.astype(np.float32),
                np.array([float(phase)], dtype=np.float32),
                lstm_hidden,
            ]
        ).astype(np.float32)

        return dqn_state

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self.history.clear()

        obs, info = self.base_env.reset()
        state = self._build_state(obs, info)

        return state, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.base_env.step(action)
        state = self._build_state(obs, info)

        return state, reward, terminated, truncated, info

    def close(self) -> None:
        self.base_env.close()


def evaluate_greedy(model: DQN, n_episodes: int = 5) -> dict:
    rewards = []
    final_waiting = []
    final_travel_times = []

    for episode in range(n_episodes):
        env = DQNLSTMHiddenEnv()
        obs, info = env.reset()

        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))

            total_reward += float(reward)
            done = terminated or truncated

        rewards.append(total_reward)
        final_waiting.append(float(info["total_waiting"]))
        final_travel_times.append(float(info["avg_travel_time"]))

        env.close()

    return {
        "avg_total_reward": float(np.mean(rewards)),
        "avg_final_waiting": float(np.mean(final_waiting)),
        "avg_final_travel_time": float(np.mean(final_travel_times)),
    }


def train() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print(f"Using dataset: {DATASET_DIR}")
    print(f"Using phase ids: {PHASE_IDS}")
    print(f"Phase map: {PHASE_MAP}")
    print(f"Using LSTM checkpoint: {LSTM_CHECKPOINT_PATH}")

    if not LSTM_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"LSTM checkpoint not found at {LSTM_CHECKPOINT_PATH}. "
            "Run your LSTM training script first."
        )

    env = DQNLSTMHiddenEnv()

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.95,
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log=str(RESULTS_DIR / "tensorboard"),
    )

    model.learn(total_timesteps=50_000)

    model_path = RESULTS_DIR / "dqn_lstm_hidden_model"
    model.save(str(model_path))

    env.close()

    print(f"Saved model to {model_path}")

    eval_results = evaluate_greedy(model, n_episodes=5)

    print("\nGreedy Evaluation Results")
    print(f"Average total reward: {eval_results['avg_total_reward']}")
    print(f"Average final waiting: {eval_results['avg_final_waiting']}")
    print(f"Average final travel time: {eval_results['avg_final_travel_time']}")


if __name__ == "__main__":
    train()