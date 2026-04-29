from __future__ import annotations

from stable_baselines3 import PPO

from scripts.stable_baselines.base_agent import BaseRLAgent


class PPOAgent(BaseRLAgent):
    def __init__(self, dataset_dir: str, state_mode: str, reward_mode: str) -> None:
        super().__init__(
            model_name="ppo",
            dataset_dir=dataset_dir,
            state_mode=state_mode,
            reward_mode=reward_mode,
        )

    def build_model(self):
        return PPO(
            policy="MlpPolicy",
            env=self.env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            gamma=0.95,
            gae_lambda=0.95,
            ent_coef=0.01,
        )