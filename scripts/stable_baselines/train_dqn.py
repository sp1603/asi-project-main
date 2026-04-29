from __future__ import annotations

from stable_baselines3 import DQN

from scripts.stable_baselines.base_agent import BaseRLAgent


class DQNAgent(BaseRLAgent):
    def __init__(self, dataset_dir: str, state_mode: str, reward_mode: str) -> None:
        super().__init__(
            model_name="dqn",
            dataset_dir=dataset_dir,
            state_mode=state_mode,
            reward_mode=reward_mode,
        )

    def build_model(self):
        return DQN(
            policy="MlpPolicy",
            env=self.env,
            verbose=1,
            learning_rate=1e-3,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=64,
            gamma=0.95,
            train_freq=4,
            target_update_interval=1000,
        )