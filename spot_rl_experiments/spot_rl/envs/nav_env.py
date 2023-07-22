# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import sys
import time
from typing import Dict, List

import numpy as np
from spot_rl.envs.base_env import SpotBaseEnv
from spot_rl.real_policy import NavPolicy
from spot_rl.utils.json_helpers import save_json_file
from spot_rl.utils.utils import (
    construct_config,
    get_default_parser,
    nav_target_from_waypoint,
)
from spot_wrapper.spot import Spot

DOCK_ID = int(os.environ.get("SPOT_DOCK_ID", 520))


def parse_arguments(args=sys.argv[1:]):
    parser = get_default_parser()
    parser.add_argument("-g", "--goal")
    parser.add_argument("-w", "--waypoints")
    parser.add_argument("-d", "--dock", action="store_true")
    parser.add_argument("-rt", "--record_trajectories", action="store_true")
    parser.add_argument("-st", "--save_trajectories")
    args = parser.parse_args(args=args)
    return args


class WaypointController:
    def __init__(self, config, spot: Spot, should_record_trajectories=False) -> None:
        # Record robot's trajectory (i.e. waypoints)
        self.recording_in_progress = False
        self.start_time = 0.0
        self.robot_trajectories = []  # type: List[List[Dict]]
        self.record_robot_trajectories = should_record_trajectories

        self.spot = spot

        # Setup
        self.policy = NavPolicy(config.WEIGHTS.NAV, device=config.DEVICE)
        self.policy.reset()

        self.nav_env = SpotNavEnv(config, self.spot)
        self.nav_env.power_robot()

    def execute(self, nav_targets) -> List[List[Dict]]:
        for nav_target in nav_targets:
            (goal_x, goal_y, goal_heading) = nav_target
            observations = self.nav_env.reset((goal_x, goal_y), goal_heading)
            done = False

            # List of Dicts to store trajectory for each of the nav_targets
            robot_trajectory = []  # type: List[Dict]
            time.sleep(1)

            self.nav_env.say(f"Navigating to {nav_target}")

            # Set start time for recording before execution of 1st nav waypoint
            if self.record_robot_trajectories and not self.recording_in_progress:
                self.start_time = time.time()
                self.recording_in_progress = True

            # Execution Loop
            while not done:
                action = self.policy.act(observations)
                observations, _, done, _ = self.nav_env.step(base_action=action)

                # Record trajectories at every step if True
                if self.record_robot_trajectories:
                    robot_trajectory.append(
                        {
                            "timestamp": time.time() - self.start_time,
                            "pose": [
                                self.nav_env.x,
                                self.nav_env.y,
                                np.rad2deg(self.nav_env.yaw),
                            ],
                        }
                    )
            # Store the trajectory for each nav_target inside the List[robot_trajectory]
            self.robot_trajectories.append(robot_trajectory)

        # Return waypoints back
        return self.robot_trajectories

    def shutdown(self, should_dock=False) -> None:
        try:
            if should_dock:
                self.nav_env.say("Executing automatic docking")
                dock_start_time = time.time()
                while time.time() - dock_start_time < 2:
                    try:
                        self.spot.dock(dock_id=DOCK_ID, home_robot=True)
                    except Exception:
                        print("Dock not found... trying again")
                        time.sleep(0.1)
            else:
                self.nav_env.say("Will sit down here")
                self.spot.sit()
        finally:
            self.spot.power_off()


class SpotNavEnv(SpotBaseEnv):
    def __init__(self, config, spot: Spot):
        super().__init__(config, spot)
        self.goal_xy = None
        self.goal_heading = None
        self.succ_distance = config.SUCCESS_DISTANCE
        self.succ_angle = np.deg2rad(config.SUCCESS_ANGLE_DIST)

    def reset(self, goal_xy, goal_heading):
        self.goal_xy = np.array(goal_xy, dtype=np.float32)
        self.goal_heading = goal_heading
        observations = super().reset()
        assert len(self.goal_xy) == 2

        return observations

    def get_success(self, observations):
        succ = self.get_nav_success(observations, self.succ_distance, self.succ_angle)
        if succ:
            self.spot.set_base_velocity(0.0, 0.0, 0.0, 1 / self.ctrl_hz)
        return succ

    def get_observations(self):
        return self.get_nav_observation(self.goal_xy, self.goal_heading)


if __name__ == "__main__":
    args = parse_arguments()
    config = construct_config(args.opts)
    # Don't need gripper camera for Nav
    config.USE_MRCNN = False

    # Get nav_targets (list) to go to
    nav_target = None
    if args.waypoints is not None:
        waypoints = [
            waypoint
            for waypoint in args.waypoints.replace(" ,", ",")
            .replace(", ", ",")
            .split(",")
            if waypoint.strip() is not None
        ]
        nav_targets = [nav_target_from_waypoint(waypoint) for waypoint in waypoints]
    else:
        assert args.goal is not None
        goal_x, goal_y, goal_heading = [float(i) for i in args.goal.split(",")]
        nav_targets = [(goal_x, goal_y, goal_heading)]

    # Default value for `args.save_trajectories` is None. Raise error to ask for correct location
    if (args.save_trajectories is not None) and (
        not os.path.isdir(args.save_trajectories)
    ):
        raise Exception(
            f"The path for saving trajectories at {args.save_trajectories} either not specified or incorrect. Please provide a correct path"
        )

    record_trajectories = (args.record_trajectories) or (
        args.save_trajectories is not None
    )

    spot = Spot("RealNavEnv")
    with spot.get_lease(hijack=True):
        wp_controller = WaypointController(
            config=config, spot=spot, should_record_trajectories=record_trajectories
        )
        try:
            robot_trajectories = wp_controller.execute(nav_targets=nav_targets)
        finally:
            wp_controller.shutdown(should_dock=args.dock)

        if args.save_trajectories is not None:
            # Ensure the folder name ends with a trailing slash
            storage_dir = os.path.join(args.save_trajectories, "")

            # save dictionary to traj.json file
            file_name = "nav_" + (
                time.strftime("%b-%d-%Y_%H%M", time.localtime()) + ".json"
            )
            file_path = storage_dir + file_name
            save_json_file(file_path=file_path, data=robot_trajectories)
