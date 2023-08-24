# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import time

import magnum as mn
import numpy as np
from spot_rl.envs.base_env import SpotBaseEnv
from spot_rl.real_policy import PlacePolicy
from spot_rl.utils.generate_place_goal import get_global_place_target
from spot_rl.utils.geometry_utils import (
    generate_intermediate_point,
    get_RPY_from_vector,
    is_position_within_bounds,
)
from spot_rl.utils.utils import (
    construct_config,
    get_default_parser,
    get_waypoint_yaml,
    place_target_from_waypoint,
)
from spot_wrapper.spot import Spot

DOCK_ID = int(os.environ.get("SPOT_DOCK_ID", 520))


def parse_arguments(args=sys.argv[1:]):
    parser = get_default_parser()
    parser.add_argument(
        "-p",
        "--place_target",
        help="input:float,float,float -> place target x,y,z in meters from the global frame (or robot's base frame if -l is specified)",
    )
    parser.add_argument(
        "-w",
        "--waypoints",
        type=str,
        help="input:string -> place target waypoints (comma separated place_target names) where robot needs to place the object",
    )
    parser.add_argument(
        "-l",
        "--target_is_local",
        action="store_true",
        help="whether the place target specified is in the local frame of the robot",
    )
    args = parser.parse_args(args=args)

    return args


def construct_config_for_place(file_path=None, opts=[]):
    config = None
    if file_path is None:
        config = construct_config(opts=opts)
    else:
        config = construct_config(file_path=file_path, opts=opts)

    # Don't need cameras for Place
    config.USE_HEAD_CAMERA = False
    config.USE_MRCNN = False

    return config


class PlaceController:
    """
    Place controller is used to execute place for given place targets

    Args:
        config: Config object
        spot: Spot object
        use_policies (bool): Whether to use policies or use BD API to execute place

    How to use:
        1. Create PlaceController object
        2. Call execute() with place_target_list as input
        3. Call shutdown() to stop the robot

    Example:
        config = construct_config_for_place(opts=[])
        spot = Spot("PlaceController")
        place_target_list = [target1, target2, ...]
        place_controller = PlaceController(config, spot, use_policies=True)
        place_result = place_controller.execute(place_target_list, is_local=False)
        place_controller.shutdown()
    """

    def __init__(self, config, spot: Spot, use_policies=True):
        self.spot = spot
        self.use_policies = use_policies
        self.config = config

        # Setup
        if self.use_policies:
            self.policy = PlacePolicy(config.WEIGHTS.PLACE, device=config.DEVICE)
            self.policy.reset()

        self.place_env = SpotPlaceEnv(config, spot)
        self.place_env.power_robot()

    def reset_env_and_policy(self, place_target, is_local):
        """
        Resets the place_env and policy

        Args:
            place_target (np.array([x,y,z])): Place target in either global frame or base frame of the robot
            is_local (bool): Whether the place target is in the base frame of the robot

        Returns:
            observations: Initial observations from the place_env
        """
        observations = self.place_env.reset(place_target, is_local)
        self.policy.reset()

        return observations

    def execute(self, place_target_list, is_local=False):
        """
        Execute place for each place target in place_target_list

        Args:
            place_target_list (list): List of place targets to go and place
            is_local (bool): Whether the place target is in the local frame of the robot

        Returns:
            success_list (list): List of dicts containing the following keys:
                - time_taken (float): Time taken to place the object
                - success (bool): Whether the place was successful
                - place_target (np.array([x,y,z])): Place target in base frame
                - ee_pos (np.array([x,y,z])): End effector position in base frame
        """
        success_list = []
        for place_target in place_target_list:
            start_time = time.time()
            if self.use_policies:
                observations = self.reset_env_and_policy(place_target, is_local)
                done = False

                while not done:
                    action = self.policy.act(observations)
                    observations, _, done, _ = self.place_env.step(arm_action=action)

            else:
                # Get reset arm position (Is there a better way to do this without using place_env?????????????)
                self.place_env.reset(place_target, is_local)

                # End effector positions in base frame (as needed by the API)
                curr_ee_pos = self.place_env.get_gripper_position_in_base_frame_spot()
                goal_ee_pos = self.place_env.get_base_frame_place_target_spot()
                intr_ee_pos = generate_intermediate_point(curr_ee_pos, goal_ee_pos)

                # Get direction vector from current ee position to goal ee position for EE orientation
                dir_rpy_to_intr = get_RPY_from_vector(goal_ee_pos - curr_ee_pos)

                # Go to intermediate point
                self.spot.move_gripper_to_point(
                    intr_ee_pos,
                    dir_rpy_to_intr,
                    self.config.ARM_TRAJECTORY_TIME_IN_SECONDS,
                    timeout_sec=10,
                )

                # Direct the gripper to face downwards
                dir_rpy_to_goal = [0.0, np.pi / 2, 0.0]

                # Go to goal point
                self.spot.move_gripper_to_point(
                    goal_ee_pos,
                    dir_rpy_to_goal,
                    self.config.ARM_TRAJECTORY_TIME_IN_SECONDS,
                    timeout_sec=10,
                )

            # Record the success
            local_place_target_spot = self.place_env.get_base_frame_place_target_spot()
            local_ee_pose_spot = (
                self.place_env.get_gripper_position_in_base_frame_spot()
            )
            success_list.append(
                {
                    "time_taken": time.time() - start_time,
                    "success": is_position_within_bounds(
                        local_place_target_spot,
                        local_ee_pose_spot,
                        self.config.SUCC_XY_DIST,
                        self.config.SUCC_Z_DIST,
                        convention="spot",
                    ),
                    "place_target": local_place_target_spot,
                    "ee_pos": local_ee_pose_spot,
                }
            )

            # Open gripper to drop the object
            self.spot.open_gripper()
            # Add sleep as open_gripper() is a non-blocking call
            time.sleep(1)

        return success_list

    def shutdown(self, should_dock=False):
        try:
            if should_dock:
                self.place_env.say("Executing automatic docking")
                dock_start_time = time.time()
                while time.time() - dock_start_time < 2:
                    try:
                        self.spot.dock(dock_id=DOCK_ID, home_robot=True)
                    except Exception:
                        print("Dock not found... trying again")
                        time.sleep(0.1)
            else:
                self.place_env.say("Will sit down here")
                self.spot.sit()
        finally:
            self.spot.power_off()


class SpotPlaceEnv(SpotBaseEnv):
    def __init__(self, config, spot: Spot):
        super().__init__(config, spot)
        self.place_target = None
        self.place_target_is_local = False

        self.ee_gripper_offset = mn.Vector3(config.EE_GRIPPER_OFFSET)
        self.placed = False

    def reset(self, place_target, target_is_local=False, *args, **kwargs):
        assert place_target is not None
        self.place_target = np.array(place_target)
        self.place_target_is_local = target_is_local

        # Move arm to initial configuration
        self.spot.close_gripper()
        cmd_id = self.spot.set_arm_joint_positions(
            positions=self.initial_arm_joint_angles, travel_time=0.75
        )
        self.spot.block_until_arm_arrives(cmd_id, timeout_sec=2)

        observations = super().reset()
        self.placed = False
        return observations

    def step(self, place=False, *args, **kwargs):
        gripper_pos_in_base_frame = self.get_gripper_position_in_base_frame_hab()
        place_target_in_base_frame = self.get_base_frame_place_target_hab()
        place = is_position_within_bounds(
            gripper_pos_in_base_frame,
            place_target_in_base_frame,
            self.config.SUCC_XY_DIST,
            self.config.SUCC_Z_DIST,
            convention="habitat",
        )

        return super().step(place=place, *args, **kwargs)

    def get_success(self, observations):
        return self.place_attempted

    def get_observations(self):
        observations = {
            "joint": self.get_arm_joints(),
            "obj_start_sensor": self.get_place_sensor(),
        }

        return observations


if __name__ == "__main__":
    args = parse_arguments()
    config = construct_config_for_place(opts=args.opts)
    waypoints_yaml_dict = get_waypoint_yaml()

    # Get place_target_list (list) to go and pick from
    place_target_list = None
    if args.waypoints is not None:
        waypoints = [
            waypoint
            for waypoint in args.waypoints.replace(" ,", ",")
            .replace(", ", ",")
            .split(",")
            if waypoint.strip() is not None
        ]
        place_target_list = [
            place_target_from_waypoint(waypoint, waypoints_yaml_dict)
            for waypoint in waypoints
        ]
    else:
        assert args.place_target is not None
        place_target_list = [[float(i) for i in args.place_target.split(",")]]

    spot = Spot("RealPlaceEnv")
    with spot.get_lease(hijack=True):
        place_controller = PlaceController(config, spot, use_policies=True)
        try:
            place_result = place_controller.execute(
                place_target_list, args.target_is_local
            )
        finally:
            place_controller.shutdown(False)

    print(f"Place results - {place_result}")
