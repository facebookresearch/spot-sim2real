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
    get_waypoint_yaml,
    nav_target_from_waypoint,
)
from spot_wrapper.spot import Spot


def parse_arguments(args=sys.argv[1:]):
    parser = get_default_parser()
    parser.add_argument(
        "-g", "--goal", help="input:string -> goal x,y,theta in meters and radians"
    )
    parser.add_argument(
        "-w",
        "--waypoints",
        help="input:string -> nav target waypoints (comma seperated) where robot needs to navigate to",
    )
    parser.add_argument(
        "-d",
        "--dock",
        action="store_true",
        help="make the robot dock after finishing navigation to all waypoints",
    )
    parser.add_argument(
        "-rt",
        "--record_trajectories",
        action="store_true",
        help="record robot's trajectories while navigating to all waypoints",
    )
    parser.add_argument(
        "-stp",
        "--save_trajectories_path",
        help="input:string -> path to save robot's trajectory",
    )
    args = parser.parse_args(args=args)

    return args


def construct_config_for_nav(file_path=None, opts=[]):
    """
    Constructs and updates the config for nav

    Args:
        file_path (str): Path to the config file
        opts (list): List of options to update the config

    Returns:
        config (Config): Updated config object
    """
    config = None
    if file_path is None:
        config = construct_config(opts=opts)
    else:
        config = construct_config(file_path=file_path, opts=opts)

    # Don't need gripper camera for Nav
    config.USE_MRCNN = False
    return config


class WaypointController:
    """
    WaypointController is used to navigate the robot to a given waypoint.

    Args:
        config: Config object
        spot: Spot object
        should_record_trajectories: bool indicating whether to record robot's trajectory

    How to use:
        1. Create WaypointController object
        2. Call execute() with nav_targets list as input and get robot's trajectory as output

    Example:
        config = construct_config(opts=[])
        spot = Spot("spot_client_name")
        with spot.get_lease(hijack=True):
            spot.power_on()

            nav_targets_list = [target1, target2, ...]
            waypoint_controller = WaypointController(config, spot)
            robot_trajectories = waypoint_controller.execute(nav_targets_list)

            spot.shutdown()
    """

    def __init__(self, config, spot: Spot, should_record_trajectories=False) -> None:
        self.config = config
        self.spot = spot

        # Record robot's trajectory (i.e. waypoints)
        self.recording_in_progress = False
        self.start_time = 0.0
        self.record_robot_trajectories = should_record_trajectories

        # Setup
        self.policy = NavPolicy(config.WEIGHTS.NAV, device=config.DEVICE)
        self.policy.reset()

        self.nav_env = SpotNavEnv(config, self.spot)

    def reset_env_and_policy(self, nav_target):
        """
        Resets the nav_env and policy

        Args:
            nav_target: (x,y,theta) where robot needs to navigate to

        Returns:
            observations: observations from the nav_env

        """
        (goal_x, goal_y, goal_heading) = nav_target
        observations = self.nav_env.reset((goal_x, goal_y), goal_heading)
        self.policy.reset()

        return observations

    def execute(self, nav_targets_list) -> List[List[Dict]]:
        """
        Executes the navigation to the given nav_targets_list and returns the robot's trajectory

        Args:
            nav_targets_list: List of nav_targets (x,y,theta) where robot needs to navigate to

        Returns:
            robot_trajectories: [[Dict]] where each Dict contains timestamp and pose of the robot, inner list contains trajectory for each nav_target and outer list is a collection of each of the nav_target's trajectory
        """

        robot_trajectories = []  # type: List[List[Dict]]
        for nav_target in nav_targets_list:
            observations = self.reset_env_and_policy(nav_target)
            done = False

            # List of Dicts to store trajectory for each of the nav_targets in nav_targets_list
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
            robot_trajectories.append(robot_trajectory)

        # Return waypoints back
        return robot_trajectories


class SpotNavEnv(SpotBaseEnv):
    def __init__(self, config, spot: Spot):
        super().__init__(config, spot)
        self._goal_xy = None
        self._enable_nav_goal_change = config.ENABLE_NAV_GOAL_CHANGE
        self.goal_heading = None
        self.succ_distance = config.SUCCESS_DISTANCE
        self.succ_angle = np.deg2rad(config.SUCCESS_ANGLE_DIST)

    def reset(self, goal_xy, goal_heading):
        self._goal_xy = np.array(goal_xy, dtype=np.float32)
        self.goal_heading = goal_heading
        observations = super().reset()
        assert len(self._goal_xy) == 2

        return observations

    def get_success(self, observations):
        succ = self.get_nav_success(observations, self.succ_distance, self.succ_angle)
        if succ:
            self.spot.set_base_velocity(0.0, 0.0, 0.0, 1 / self.ctrl_hz)
        return succ

    def get_observations(self):
        # Update the point goal given depth and rgb images
        if self._enable_nav_goal_change:
            arm_depth, arm_depth_bbox = self.get_gripper_images()
            # Update the goal xy here
            self._goal_xy = self.spot.get_new_goal_given_obj_img(
                arm_depth, arm_depth_bbox
            )

        return self.get_nav_observation(self._goal_xy, self.goal_heading)


if __name__ == "__main__":
    args = parse_arguments()
    config = construct_config_for_nav(opts=args.opts)
    waypoints_yaml_dict = get_waypoint_yaml()

    # Get nav_targets_list (list) to go to
    nav_targets_list = None
    if args.waypoints is not None:
        waypoints = [
            waypoint
            for waypoint in args.waypoints.replace(" ,", ",")
            .replace(", ", ",")
            .split(",")
            if waypoint.strip() is not None
        ]
        nav_targets_list = [
            nav_target_from_waypoint(waypoint, waypoints_yaml_dict)
            for waypoint in waypoints
        ]
    else:
        assert args.goal is not None
        goal_x, goal_y, goal_heading = [float(i) for i in args.goal.split(",")]
        nav_targets_list = [(goal_x, goal_y, goal_heading)]

    # Default value for `args.save_trajectories_path` is None. Raise error to ask for correct location
    if (args.save_trajectories_path is not None) and (
        not os.path.isdir(args.save_trajectories_path)
    ):
        raise Exception(
            f"The path for saving trajectories at {args.save_trajectories_path} either not specified or incorrect. Please provide a correct path"
        )

    record_trajectories = (args.record_trajectories) or (
        args.save_trajectories_path is not None
    )

    spot = Spot("RealNavEnv")
    with spot.get_lease(hijack=True):
        spot.power_robot()
        wp_controller = WaypointController(
            config=config, spot=spot, should_record_trajectories=record_trajectories
        )
        try:
            robot_trajectories = wp_controller.execute(
                nav_targets_list=nav_targets_list
            )
        finally:
            spot.shutdown(should_dock=args.dock)

        if args.save_trajectories_path is not None:
            # Ensure the folder name ends with a trailing slash
            storage_dir = os.path.join(args.save_trajectories_path, "")

            # save dictionary to traj.json file
            file_name = "nav_" + (
                time.strftime("%b-%d-%Y_%H%M", time.localtime()) + ".json"
            )
            file_path = storage_dir + file_name
            save_json_file(file_path=file_path, data=robot_trajectories)
