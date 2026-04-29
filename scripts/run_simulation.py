from __future__ import annotations

from pathlib import Path

from env.cityflow_env import CityFlowEnv


def main() -> None:
    env = CityFlowEnv(
        config_path="CityFlow/examples/config.json",
        intersection_id="intersection_1_1",
        lane_ids=[
            "road_0_1_0_0",
            "road_1_0_1_0",
            "road_2_1_2_0",
            "road_1_2_3_0",
        ],
        phase_ids=list(range(8)),
        action_interval=5,
        max_steps=200,
        render_mode="human",
    )

    obs, info = env.reset()
    print("Initial obs:", obs)
    print("Initial info:", info)

    print("\nALL LANE IDS:")
    print(env.eng.get_lane_waiting_vehicle_count().keys())

    terminated = False
    truncated = False
    action = 0

    while not terminated and not truncated:
        obs, reward, terminated, truncated, info = env.step(action)
        print("Reward:", reward)
        action = (action + 1) % env.action_space.n

    env.close()

    replay_roadnet = Path("CityFlow/examples/replay_roadnet.json")
    replay_txt = Path("CityFlow/examples/replay.txt")

    print("\nReplay generation check:")
    print("replay_roadnet.json exists:", replay_roadnet.exists())
    print("replay.txt exists:", replay_txt.exists())

    if replay_roadnet.exists() and replay_txt.exists():
        env.open_cityflow_frontend()
    else:
        print("Replay files were not generated, so frontend was not opened.")


if __name__ == "__main__":
    main()