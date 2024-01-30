# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import sys
import time
from typing import Dict, List

import numpy as np
import rospy
from spot_rl.envs.base_env import SpotBaseEnv
from spot_rl.real_policy import OpenDrawerPolicy
from spot_rl.utils.utils import (
    construct_config,
    get_default_parser,
    map_user_input_to_boolean,
)
from spot_wrapper.spot import Spot


def parse_arguments(args=sys.argv[1:]):
    parser = get_default_parser()
    parser.add_argument(
        "-t", "--target-object", type=str, help="name of the target object"
    )
    parser.add_argument(
        "-ms", "--max_episode_steps", type=int, help="max episode steps"
    )
    args = parser.parse_args(args=args)

    if args.max_episode_steps is not None:
        args.max_episode_steps = int(args.max_episode_steps)
    return args


def construct_config_for_open_drawer(file_path=None, opts=[], max_episode_steps=None):
    """
    Constructs and updates the config for open drawer

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

    # Don't need head cameras for open drawer
    config.USE_HEAD_CAMERA = False

    # Update max episode steps based on the input argument
    if max_episode_steps is not None:
        print(
            f"WARNING: Overriding max_espisode_steps in config from {config.MAX_EPISODE_STEPS} to {max_episode_steps}"
        )
        config.MAX_EPISODE_STEPS = max_episode_steps
    return config


class OpenDrawerController:
    """
    OpenDrawerController is used to open the drawer

    Args:
        config (Config): Config object
        spot (Spot): Spot object

    How to use:
        1. Create a OpenDrawerController object
        2. Call execute() method with the target object list

    Example:
        config = construct_config_for_open_drawer(opts=[])
        spot = Spot("spot_client_name")
        with spot.get_lease(hijack=True):
            spot.power_robot()

            open_drawer_controller = OpenDrawerController(config, spot)
            open_drawer_controller.execute()

            spot.shutdown(should_dock=True)
    """

    def __init__(self, config, spot):
        self.config = config
        self.spot = spot

        # Load the necessary checkpoints
        ckpt_dict = {}
        ckpt_dict["entire_net"] = config.WEIGHTS.OPEN_DRAWER
        # Use config.device as the device for open drawer policy
        self.policy = OpenDrawerPolicy(ckpt_dict, device=config.DEVICE, config=config)

        self.policy.reset()

        self.open_drawer_env = SpotOpenDrawerEnv(config, spot)

    def reset_env_and_policy(self):
        """
        Resets the open_drawer_env and policy

        Args:

        Returns:
            observations: observations from the open_drawer_env

        """
        observations = self.open_drawer_env.reset()
        self.policy.reset()

        return observations

    def execute(self):
        """
        Open the drawer

        CAUTION: The robot will drop the object after picking it, please use objects that are not fragile

        Args:
        Returns:
            None
        """

        observations = self.reset_env_and_policy()
        done = False

        while not done:
            # Get the action
            action = self.policy.act(observations)

            ee_action, gripper_action = None, None
            # first 6 are ee actions, then 1 is the gripper action
            ee_action = action[0:6]
            gripper_action = action[6]
            print("ee_action:", ee_action, "gripper_action:", gripper_action)

            # Check if arm_action contains NaN values
            # assert not np.isnan(np.array(arm_action)).any()
            observations, _, done, _ = self.open_drawer_env.step(
                ee_action=ee_action, gripper_action=gripper_action
            )

        return None


class SpotOpenDrawerEnv(SpotBaseEnv):
    def __init__(self, config, spot):
        # Select suitable keys
        max_joint_movement_key = "MAX_JOINT_MOVEMENT_MOBILE_GAZE"

        super().__init__(
            config,
            spot,
            stopwatch=None,
            max_joint_movement_key=max_joint_movement_key,
        )

    def reset(self, *args, **kwargs):
        # # Move arm to initial configuration
        # cmd_id = self.spot.set_arm_joint_positions(
        #     positions=self.initial_arm_joint_angles, travel_time=1
        # )
        # self.spot.block_until_arm_arrives(cmd_id, timeout_sec=1)

        # Update target object name as provided in config
        observations = super().reset(*args, **kwargs)

        return observations

    def step(self, ee_action=None, gripper_action=None):
        observations, reward, done, info = super().step(
            ee_action=ee_action,
            gripper_action=gripper_action,
        )
        return observations, reward, done, info

    def remap_observation_keys_for_hab3(self, observations):
        """
        Change observation keys as per hab3.

        @INFO: Policies trained on older hab versions DON'T need remapping
        """
        open_drawer_observations = {}
        open_drawer_observations["articulated_agent_arm_depth"] = observations[
            "arm_depth"
        ]
        return open_drawer_observations

    def get_observations(self):
        arm_depth, arm_depth_bbox = self.get_gripper_images()
        observations = {
            "joint": self.get_arm_joints(),
            "arm_depth": arm_depth,
            "arm_depth_bbox": arm_depth_bbox,
        }

        # Remap observation keys for open drawer as it was trained with Habitat version3

        observations = self.remap_observation_keys_for_hab3(observations)

        return observations

    def get_success(self, observations):
        return self.grasp_attempted


if __name__ == "__main__":
    spot = Spot("RealOpenDrawerEnv")
    args = parse_arguments()
    config = construct_config_for_open_drawer(
        opts=args.opts,
        max_episode_steps=args.max_episode_steps,
    )

    with spot.get_lease(hijack=True):
        spot.power_robot()
        open_drawer_controller = OpenDrawerController(config, spot)
        open_drawer_controller.execute()

        breakpoint()
        spot.shutdown(should_dock=True)
