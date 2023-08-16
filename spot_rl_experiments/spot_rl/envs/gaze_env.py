# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import sys
import time
from typing import Dict, List

import cv2
import numpy as np
import rospy
from spot_rl.envs.base_env import SpotBaseEnv
from spot_rl.real_policy import GazePolicy
from spot_rl.utils.utils import (
    construct_config,
    get_default_parser,
    map_user_input_to_boolean,
)
from spot_wrapper.spot import Spot, wrap_heading

DOCK_ID = int(os.environ.get("SPOT_DOCK_ID", 520))


def parse_arguments(args=sys.argv[1:]):
    parser = get_default_parser()
    parser.add_argument(
        "-t", "--target-object", type=str, help="name of the target object"
    )
    parser.add_argument(
        "-dp",
        "--dont_pick_up",
        action="store_true",
        help="robot should attempt pick but not actually pick",
    )
    parser.add_argument(
        "-ms", "--max_episode_steps", type=int, help="max episode steps"
    )
    args = parser.parse_args(args=args)

    if args.max_episode_steps is not None:
        args.max_episode_steps = int(args.max_episode_steps)
    return args


def construct_config_for_gaze(file_path=None, opts=[]):
    """
    Constructs and updates the config for gaze

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
        config = construct_config(file_path=file_path, opts=[])

    # Don't need head cameras for Gaze
    config.USE_HEAD_CAMERA = False
    return config


def update_config_for_multiple_gaze(config, dont_pick_up, max_episode_steps):
    """
    Updates the config for multiple gaze

    Args:
        config (Config): Config object
        dont_pick_up (bool): Whether to pick up the object or not
        max_episode_steps (int): Max episode steps

    Returns:
        config (Config): Updated config object
    """
    # Update the config based on the input argument
    config.DONT_PICK_UP = dont_pick_up

    # Update max episode steps based on the input argument
    config.MAX_EPISODE_STEPS = max_episode_steps
    return config


class GazeController:
    """
    GazeController is used to gaze at, and pick given objects.

    Args:
        config (Config): Config object
        spot (Spot): Spot object

    How to use:
        1. Create a GazeController object
        2. Call execute() method with the target object list
        3. Call shutdown() to stop the robot

    Example:
        config = construct_config(opts=[])
        spot = Spot("spot_client_name")
        gaze_target_list = ["apple", "banana"]
        gaze_controller = GazeController(config, spot)
        gaze_results = gaze_controller.execute(gaze_target_list)
        gaze_controller.shutdown()
    """

    def __init__(self, config, spot):
        self.config = config
        self.spot = spot

        # Setup
        self.policy = GazePolicy(config.WEIGHTS.GAZE, device=config.DEVICE)
        self.policy.reset()

        self.gaze_env = SpotGazeEnv(config, spot)
        self.gaze_env.power_robot()

    def execute(self, target_object_list, take_user_input=False):
        """
        Gaze at the target object list and pick up the objects if specified in the config

        CAUTION: The robot will drop the object after picking it, please use objects that are not fragile

        Args:
            target_object_list (list): List of target objects to gaze at
            take_user_input (bool): Whether to take user input for the success of the gaze

        Returns:
            gaze_success_list (list): List of dictionaries containing the target object name, time taken and success
        """
        gaze_success_list = []
        print(f"Target object list : {target_object_list}")
        for target_object in target_object_list:
            observations = self.gaze_env.reset(target_obj_name=target_object)
            done = False
            start_time = time.time()
            self.gaze_env.say(f"Gaze at target object - {target_object}")
            while not done:
                action = self.policy.act(observations)
                observations, _, done, _ = self.gaze_env.step(arm_action=action)
            self.gaze_env.say("Gaze finished")

            # Ask user for feedback about the success of the gaze and update the "success" flag accordingly
            success_status_from_user_feedback = True
            if take_user_input:
                user_prompt = f"Did the robot successfully picked at the right object - {target_object}?"
                success_status_from_user_feedback = map_user_input_to_boolean(
                    user_prompt
                )

            gaze_success_list.append(
                {
                    "target_object": target_object,
                    "time_taken": time.time() - start_time,
                    "success": self.gaze_env.grasp_attempted
                    and success_status_from_user_feedback,
                }
            )
        return gaze_success_list

    def shutdown(self, should_dock=False) -> None:
        """
        Stops the robot and docks it if should_dock is True else sits the robot down

        Args:
            should_dock: bool indicating whether to dock the robot or not
        """
        try:
            if should_dock:
                self.gaze_env.say("Executing automatic docking")
                dock_start_time = time.time()
                while time.time() - dock_start_time < 2:
                    try:
                        self.spot.dock(dock_id=DOCK_ID, home_robot=True)
                    except Exception:
                        print("Dock not found... trying again")
                        time.sleep(0.1)
            else:
                self.gaze_env.say("Will sit down here")
                self.spot.sit()
        finally:
            self.spot.power_off()


class SpotGazeEnv(SpotBaseEnv):
    def __init__(self, config, spot):
        super().__init__(config, spot)
        self.target_obj_name = self.config.TARGET_OBJ_NAME

    def reset(self, target_obj_name=None, *args, **kwargs):
        # Move arm to initial configuration
        cmd_id = self.spot.set_arm_joint_positions(
            positions=self.initial_arm_joint_angles, travel_time=1
        )
        self.spot.block_until_arm_arrives(cmd_id, timeout_sec=1)
        print("Open gripper called in Gaze")
        self.spot.open_gripper()

        # Update target object name as provided in config
        if target_obj_name is None:
            target_obj_name = self.config.TARGET_OBJ_NAME
        observations = super().reset(target_obj_name=target_obj_name, *args, **kwargs)
        rospy.set_param("object_target", target_obj_name)

        return observations

    def step(self, base_action=None, arm_action=None, grasp=False, place=False):
        grasp = self.should_grasp()

        observations, reward, done, info = super().step(
            base_action, arm_action, grasp, place
        )

        return observations, reward, done, info

    def get_observations(self):
        arm_depth, arm_depth_bbox = self.get_gripper_images()
        observations = {
            "joint": self.get_arm_joints(),
            "arm_depth": arm_depth,
            "arm_depth_bbox": arm_depth_bbox,
        }

        return observations

    def get_success(self, observations):
        return self.grasp_attempted


if __name__ == "__main__":
    spot = Spot("RealGazeEnv")
    args = parse_arguments()
    config = construct_config_for_gaze(opts=args.opts)
    max_episode_steps = (
        args.max_episode_steps
        if args.max_episode_steps is not None
        else config.MAX_EPISODE_STEPS
    )
    config = update_config_for_multiple_gaze(
        config, dont_pick_up=args.dont_pick_up, max_episode_steps=max_episode_steps
    )

    target_objects_list = []
    if args.target_object is not None:
        target_objects_list = [
            target
            for target in args.target_object.replace(" ,", ",")
            .replace(", ", ",")
            .split(",")
            if target.strip() is not None
        ]

    print(f"Target_objects list - {target_objects_list}")
    with spot.get_lease(hijack=True):
        gaze_controller = GazeController(config, spot)
        try:
            gaze_result = gaze_controller.execute(target_objects_list)
            print(gaze_result)
        finally:
            gaze_controller.shutdown()
