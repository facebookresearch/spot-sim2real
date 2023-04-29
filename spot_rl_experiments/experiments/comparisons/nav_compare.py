import argparse
import time
from collections import defaultdict

import numpy as np
from spot_wrapper.spot import Spot, wrap_heading
from spot_wrapper.utils import say

from spot_rl.envs.nav_env import SpotNavEnv
from spot_rl.real_policy import NavPolicy
from spot_rl.utils.utils import construct_config

ROUTES = [
    ((6.0, 2.0, 0.0), (6.0, -1.0, 0.0)),
    ((3.5, 1.5, np.pi), (-0.25, 1.5, np.pi)),
    ((1.0, 0.0, 0.0), (8, -5.18, -np.pi / 2)),
]


def main(spot, idx):
    config = construct_config([])
    policy = NavPolicy(config.WEIGHTS.NAV, device=config.DEVICE)

    env = SpotNavEnv(config, spot)
    env.power_robot()

    start_waypoint, goal_waypoint = ROUTES[idx]

    return_to_start(spot, start_waypoint, policy, env)
    time.sleep(2)
    datas = defaultdict(list)
    times = defaultdict(list)

    for ctrl_idx, nav_func in enumerate([learned_navigate, baseline_navigate]):
        for _ in range(3):
            say("Starting episode.")
            time.sleep(3)
            st = time.time()
            traj = nav_func(spot=spot, waypoint=goal_waypoint, policy=policy, env=env)
            traj_time = time.time() - st
            datas[ctrl_idx].append((traj, traj_time))
            times[ctrl_idx].append(traj_time)
            spot.set_base_velocity(0, 0, 0, 1)
            say("Done with episode. Returning.")
            time.sleep(3)
            print("Returning...")
            if idx == 2:
                return_to_start(spot, (8.0, -1.0, np.pi), policy, env, no_learn=True)
            return_to_start(spot, start_waypoint, policy, env)
            print("Done returning.")

    for k, v in times.items():
        name = ["Learned", "BDAPI"][k]
        print(f"{name} completion times:")
        for vv in v:
            print(vv)

    for ctrl_idx, trajs in datas.items():
        for ep_id, (traj, traj_time) in enumerate(trajs):
            data = [str(traj_time)]
            for t_x_y_yaw in traj:
                data.append(",".join([str(i) for i in t_x_y_yaw]))
            name = ["learned", "bdapi"][ctrl_idx]
            with open(f"route_{idx}_ep_{ep_id}_{name}.txt", "w") as f:
                f.write("\n".join(data) + "\n")


def baseline_navigate(spot, waypoint, limits=True, **kwargs):
    goal_x, goal_y, goal_heading = waypoint
    if limits:
        cmd_id = spot.set_base_position(
            x_pos=goal_x,
            y_pos=goal_y,
            yaw=goal_heading,
            end_time=100,
            max_fwd_vel=0.5,
            max_hor_vel=0.05,
            max_ang_vel=np.deg2rad(30),
        )
    else:
        cmd_id = spot.set_base_position(
            x_pos=goal_x, y_pos=goal_y, yaw=goal_heading, end_time=100
        )
    cmd_status = None
    success = False
    traj = []
    st = time.time()
    while not success and time.time() < st + 20:
        if cmd_status != 1:
            traj.append((time.time(), *spot.get_xy_yaw()))
            time.sleep(0.5)
            feedback_resp = spot.get_cmd_feedback(cmd_id)
            cmd_status = (
                feedback_resp.feedback.synchronized_feedback.mobility_command_feedback
            ).se2_trajectory_feedback.status
        else:
            if limits:
                cmd_id = spot.set_base_position(
                    x_pos=goal_x,
                    y_pos=goal_y,
                    yaw=goal_heading,
                    end_time=100,
                    max_fwd_vel=0.5,
                    max_hor_vel=0.05,
                    max_ang_vel=np.deg2rad(30),
                )
            else:
                cmd_id = spot.set_base_position(
                    x_pos=goal_x, y_pos=goal_y, yaw=goal_heading, end_time=100
                )

        x, y, yaw = spot.get_xy_yaw()
        dist = np.linalg.norm(np.array([x, y]) - np.array([goal_x, goal_y]))
        heading_diff = abs(wrap_heading(goal_heading - yaw))
        success = dist < 0.3 and heading_diff < np.deg2rad(5)

    return traj


def learned_navigate(waypoint, policy, env, **kwargs):
    goal_x, goal_y, goal_heading = waypoint
    observations = env.reset((goal_x, goal_y), goal_heading)
    done = False
    policy.reset()
    traj = []
    while not done:
        traj.append((time.time(), *env.spot.get_xy_yaw()))
        action = policy.act(observations)
        observations, _, done, _ = env.step(base_action=action)

    return traj


def return_to_start(spot, waypoint, policy, env, no_learn=False):
    # goal_x, goal_y, goal_heading = waypoint
    if not no_learn:
        learned_navigate(waypoint, policy, env)
    baseline_navigate(spot, waypoint, limits=False)
    # spot.set_base_position(
    #     x_pos=goal_x, y_pos=goal_y, yaw=goal_heading, end_time=100, blocking=True
    # )


if __name__ == "__main__":
    spot = Spot("NavCompare")
    parser = argparse.ArgumentParser()
    parser.add_argument("idx", type=int)
    args = parser.parse_args()
    with spot.get_lease(hijack=True):
        main(spot, args.idx)
