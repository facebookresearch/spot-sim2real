# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import sys
from typing import Dict, List

import numpy as np
import rospy
from spot_rl.envs.base_env import SpotBaseEnv
from spot_wrapper.spot import Spot


class SpotGazeEnv(SpotBaseEnv):
    def __init__(self, config, spot: Spot, use_mobile_pick: bool = False):
        # Select suitable keys
        max_joint_movement_key = (
            "MAX_JOINT_MOVEMENT_MOBILE_GAZE"
            if use_mobile_pick
            else "MAX_JOINT_MOVEMENT"
        )
        max_lin_dist_key = (
            "MAX_LIN_DIST_MOBILE_GAZE" if use_mobile_pick else "MAX_LIN_DIST"
        )
        max_ang_dist_key = (
            "MAX_ANG_DIST_MOBILE_GAZE" if use_mobile_pick else "MAX_ANG_DIST"
        )

        super().__init__(
            config,
            spot,
            stopwatch=None,
            max_joint_movement_key=max_joint_movement_key,
            max_lin_dist_key=max_lin_dist_key,
            max_ang_dist_key=max_ang_dist_key,
        )
        self.target_obj_name = None
        self._use_mobile_pick = use_mobile_pick
        self.initial_arm_joint_angles = np.deg2rad(config.GAZE_ARM_JOINT_ANGLES)

    def reset(self, target_obj_name, *args, **kwargs):
        # Move arm to initial configuration
        cmd_id = self.spot.set_arm_joint_positions(
            positions=self.initial_arm_joint_angles, travel_time=1
        )
        self.spot.block_until_arm_arrives(cmd_id, timeout_sec=1)
        print("Open gripper called in Gaze")
        self.spot.open_gripper()

        # Update target object name as provided in config
        observations = super().reset(target_obj_name=target_obj_name, *args, **kwargs)
        rospy.set_param("object_target", target_obj_name)
        rospy.set_param("is_gripper_blocked", 0)
        return observations

    def step(self, base_action=None, arm_action=None, grasp=False, place=False):
        grasp = self.should_grasp()

        observations, reward, done, info = super().step(
            base_action,
            arm_action,
            grasp,
            place,
        )
        return observations, reward, done, info

    def remap_observation_keys_for_hab3(self, observations):
        """
        Change observation keys as per hab3.

        @INFO: Policies trained on older hab versions DON'T need remapping
        """
        mobile_gaze_observations = {}
        mobile_gaze_observations["arm_depth_bbox_sensor"] = observations[
            "arm_depth_bbox"
        ]
        mobile_gaze_observations["articulated_agent_arm_depth"] = observations[
            "arm_depth"
        ]
        mobile_gaze_observations["joint"] = observations["joint"]
        return mobile_gaze_observations

    def get_observations(self):
        arm_depth, arm_depth_bbox = self.get_gripper_images()
        observations = {
            "joint": self.get_arm_joints(),
            "arm_depth": arm_depth,
            "arm_depth_bbox": arm_depth_bbox,
        }

        # Remap observation keys for mobile gaze as it was trained with Habitat version3
        if self._use_mobile_pick:
            observations = self.remap_observation_keys_for_hab3(observations)

        return observations

    def get_success(self, observations):
        return self.grasp_attempted
