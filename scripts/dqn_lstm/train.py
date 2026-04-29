import json
import numpy as np
import random

from env.cityflow_env import CityFlowEnv

def extract_lane_ids_from_roadnet(roadnet_path: str, intersection_id: str):
    import json

    with open(roadnet_path, "r") as f:
        roadnet = json.load(f)

    incoming_lanes = []

    for road in roadnet["roads"]:
        if road["endIntersection"] == intersection_id:
            road_id = road["id"]

            num_lanes = len(road["lanes"])

            for i in range(num_lanes):
                lane_id = f"{road_id}_{i}"
                incoming_lanes.append(lane_id)

    return incoming_lanes

def make_env():
    dataset_path = "CityFlow/hangzhou"

    lane_ids = extract_lane_ids_from_roadnet(
        f"{dataset_path}/roadnet.json",
        "intersection_1_1"
    )

    print("lane_ids:", lane_ids)

    env = CityFlowEnv(
        config_path=f"{dataset_path}/config.json",
        intersection_id="intersection_1_1",
        lane_ids=lane_ids,
        phase_ids=[0, 2],
        action_interval=5,
        max_steps=200,
    )

    return env



def main():
    env = make_env()

    obs, info = env.reset()

    print("obs shape:", obs.shape)
    print("obs:", obs)
    print("info:", info)

    env.close()


if __name__ == "__main__":
    main()