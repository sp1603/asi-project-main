from __future__ import annotations

import csv
import json
from pathlib import Path

from env.cityflow_env import CityFlowEnv


NUM_EPISODES = 5


def build_runtime_config() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = project_root / "CityFlow" / "examples"
    config_path = dataset_dir / "config.json"

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    roadnet_name = Path(config["roadnetFile"]).name
    flow_name = Path(config["flowFile"]).name

    runtime_config = dict(config)
    runtime_config["dir"] = str(dataset_dir.resolve()) + "/"
    runtime_config["roadnetFile"] = roadnet_name
    runtime_config["flowFile"] = flow_name

    out_dir = project_root / "results" / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_config_path = out_dir / "runtime_config.json"
    with runtime_config_path.open("w", encoding="utf-8") as f:
        json.dump(runtime_config, f, indent=2)

    return runtime_config_path


def make_env() -> CityFlowEnv:
    runtime_config_path = build_runtime_config()
    return CityFlowEnv(
        config_path=str(runtime_config_path),
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
        render_mode=None,
    )


def fixed_cycle_policy(step_idx: int, num_actions: int) -> int:
    return step_idx % num_actions


def random_policy(env: CityFlowEnv) -> int:
    return env.action_space.sample()


def run_episode(env: CityFlowEnv, policy_name: str, csv_path: str) -> dict:
    obs, info = env.reset()

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    total_reward = 0.0
    total_waiting_sum = 0.0
    num_steps = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step",
            "sim_time",
            "reward",
            "total_waiting",
            "vehicle_count",
            "avg_travel_time",
        ])

        terminated = False
        truncated = False
        step_idx = 0

        while not terminated and not truncated:
            if policy_name == "fixed":
                action = fixed_cycle_policy(step_idx, env.action_space.n)
            else:
                action = random_policy(env)

            obs, reward, terminated, truncated, info = env.step(action)

            writer.writerow([
                step_idx,
                info["time"],
                reward,
                info["total_waiting"],
                info["vehicle_count"],
                info["avg_travel_time"],
            ])

            total_reward += reward
            total_waiting_sum += info["total_waiting"]
            num_steps += 1
            step_idx += 1

    env.close()

    return {
        "total_reward": total_reward,
        "avg_waiting": total_waiting_sum / num_steps,
        "final_travel_time": info["avg_travel_time"],
    }


def run_policy(policy_name: str) -> None:
    print(f"\nRunning {policy_name} for {NUM_EPISODES} episodes")

    results = []

    for ep in range(NUM_EPISODES):
        print(f"Episode {ep+1}")

        env = make_env()

        result = run_episode(
            env,
            policy_name,
            csv_path=f"results/baseline/{policy_name}_ep{ep+1}.csv",
        )

        results.append(result)

    avg_reward = sum(r["total_reward"] for r in results) / NUM_EPISODES
    avg_waiting = sum(r["avg_waiting"] for r in results) / NUM_EPISODES
    avg_travel_time = sum(r["final_travel_time"] for r in results) / NUM_EPISODES

    print("\nAVERAGED RESULTS:")
    print("avg total reward:", avg_reward)
    print("avg waiting:", avg_waiting)
    print("avg travel time:", avg_travel_time)


def main() -> None:
    run_policy("fixed")
    run_policy("random")


if __name__ == "__main__":
    main()