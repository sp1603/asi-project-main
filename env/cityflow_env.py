from __future__ import annotations

from typing import Any
from pathlib import Path
import json
import webbrowser
import gc
import os

import cityflow
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CityFlowEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config_path: str,
        intersection_id: str,
        lane_ids: list[str],
        phase_ids: list[int],
        action_interval: int = 10,
        max_steps: int = 200,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.config_path = str(Path(config_path).resolve())
        self.config_path_obj = Path(self.config_path)
        self.intersection_id = intersection_id
        self.lane_ids = lane_ids
        self.phase_ids = phase_ids
        self.action_interval = action_interval
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.eng: cityflow.Engine | None = None
        self.steps = 0
        self.current_phase_index = 0

        self.action_space = spaces.Discrete(len(self.phase_ids))
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(len(self.lane_ids),),
            dtype=np.float32,
        )

        self.replay_roadnet_path: Path | None = None
        self.replay_txt_path: Path | None = None
        self._load_replay_paths()

    def _load_replay_paths(self) -> None:
        with open(self.config_path_obj, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        base_dir = Path(os.getcwd()) / cfg["dir"]
        base_dir = base_dir.resolve()

        self.replay_roadnet_path = base_dir / cfg["roadnetLogFile"]
        self.replay_txt_path = base_dir / cfg["replayLogFile"]

    def _make_engine(self) -> cityflow.Engine:
        eng = cityflow.Engine(self.config_path, thread_num=1)
        return eng

    def _get_state(self) -> np.ndarray:
        assert self.eng is not None
        waiting = self.eng.get_lane_waiting_vehicle_count()
        return np.array(
            [waiting.get(lane_id, 0) for lane_id in self.lane_ids],
            dtype=np.float32,
        )

    def _get_info(self) -> dict[str, Any]:
        assert self.eng is not None
        obs = self._get_state()
        return {
            "time": self.eng.get_current_time(),
            "vehicle_count": self.eng.get_vehicle_count(),
            "total_waiting": float(np.sum(obs)),
            "avg_travel_time": self.eng.get_average_travel_time(),
            "phase_index": self.current_phase_index,
            "phase_id": self.phase_ids[self.current_phase_index],
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        if self.eng is not None:
            self.close()

        self.eng = self._make_engine()
        self.steps = 0
        self.current_phase_index = 0

        self.eng.set_tl_phase(
            self.intersection_id,
            self.phase_ids[self.current_phase_index],
        )

        obs = self._get_state()
        info = self._get_info()
        return obs, info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.eng is not None

        action = int(action)
        if action < 0 or action >= len(self.phase_ids):
            raise ValueError(f"Invalid action {action}. Must be in [0, {len(self.phase_ids) - 1}]")

        self.current_phase_index = action

        self.eng.set_tl_phase(
            self.intersection_id,
            self.phase_ids[self.current_phase_index],
        )

        for _ in range(self.action_interval):
            self.eng.next_step()
            self.steps += 1

        obs = self._get_state()
        obs = np.minimum(obs, 10)
        total_waiting = float(np.sum(obs))
        vehicle_count = self.eng.get_vehicle_count()
        reward = -(total_waiting + 0.1 * vehicle_count) 

        terminated = False
        truncated = self.steps >= self.max_steps
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        if self.eng is None:
            return

        info = self._get_info()
        print(
            f"time={info['time']}, vehicles={info['vehicle_count']}, "
            f"waiting={info['total_waiting']}, phase={self.phase_ids[info['phase_index']]}"
        )
        print("Replay files:")
        print("  roadnet:", info["replay_roadnet"])
        print("  replay :", info["replay_txt"])

    def open_cityflow_frontend(self) -> None:
        candidates = [
            Path("CityFlow/frontend/index.html"),
            Path("CityFlow/frontend/web/index.html"),
        ]
        for candidate in candidates:
            if candidate.exists():
                webbrowser.open(candidate.resolve().as_uri())
                return
        raise FileNotFoundError("Could not find CityFlow frontend index.html.")

    def close(self) -> None:
        if self.eng is not None:
            del self.eng
            self.eng = None
            gc.collect()