from __future__ import annotations

from stable_baselines3 import A2C

from scripts.stable_baselines.base_agent import BaseRLAgent


class A2CAgent(BaseRLAgent):
    def __init__(self, dataset_dir: str, state_mode: str, reward_mode: str) -> None:
        super().__init__(
            model_name="a2c",
            dataset_dir=dataset_dir,
            state_mode=state_mode,
            reward_mode=reward_mode,
        )

    def build_model(self):
        return A2C(
            policy="MlpPolicy",
            env=self.env,
            verbose=1,
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.95,
            gae_lambda=1.0,
            ent_coef=0.01,
        )