from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from scripts.stable_baselines.env_wrapper import CityFlowSB3Env


class BaseRLAgent(ABC):
    def __init__(
        self,
        model_name: str,
        dataset_dir: str,
        state_mode: str,
        reward_mode: str,
        results_dir: str = "results/stable_baselines",
    ) -> None:
        self.model_name = model_name
        self.dataset_dir = Path(dataset_dir)
        self.dataset_name = self.dataset_dir.name
        self.state_mode = state_mode
        self.reward_mode = reward_mode

        self.results_dir = (
            Path(results_dir)
            / self.dataset_name
            / f"{model_name}__state-{state_mode}__reward-{reward_mode}"
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.env = Monitor(
            CityFlowSB3Env(
                dataset_dir=str(self.dataset_dir),
                runtime_config_dir=str(self.results_dir),
                state_mode=state_mode,
                reward_mode=reward_mode,
            )
        )

        check_env(
            CityFlowSB3Env(
                dataset_dir=str(self.dataset_dir),
                runtime_config_dir=str(self.results_dir),
                state_mode=state_mode,
                reward_mode=reward_mode,
            ),
            warn=True,
        )

        self.model = self.build_model()

    @abstractmethod
    def build_model(self):
        raise NotImplementedError

    def train(self, total_timesteps: int = 50_000) -> None:
        self.model.learn(total_timesteps=total_timesteps)
        self.model.save(self.results_dir / f"{self.model_name}_model")

    def evaluate(self, episodes: int = 10) -> dict[str, float | str]:
        total_rewards: list[float] = []
        final_waitings: list[float] = []
        avg_waitings: list[float] = []
        avg_travel_times: list[float] = []

        for _ in range(episodes):
            obs, info = self.env.reset()
            done = False
            truncated = False
            total_reward = 0.0
            wait_trace: list[float] = []
            last_info = info

            while not (done or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.env.step(action)
                total_reward += float(reward)
                wait_trace.append(float(info["total_waiting"]))
                last_info = info

            total_rewards.append(total_reward)
            final_waitings.append(float(last_info["total_waiting"]))
            avg_waitings.append(float(np.mean(wait_trace)))
            avg_travel_times.append(float(last_info["avg_travel_time"]))

        return {
            "dataset": self.dataset_name,
            "model": self.model_name,
            "state_mode": self.state_mode,
            "reward_mode": self.reward_mode,
            "mean_total_reward": float(np.mean(total_rewards)),
            "mean_final_waiting": float(np.mean(final_waitings)),
            "mean_avg_waiting": float(np.mean(avg_waitings)),
            "mean_avg_travel_time": float(np.mean(avg_travel_times)),
        }

    def close(self) -> None:
        self.env.close()