# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import sys
from typing import Dict, List

import magnum as mn
import numpy as np
import rospy
from spot_rl.envs.base_env import SpotBaseEnv
from spot_wrapper.spot import Spot


class SpotOpenDrawerEnv(SpotBaseEnv):
    def __init__(self, config, spot: Spot, use_mobile_pick: bool = False):
        # Select suitable keys
        max_joint_movement_key = "MAX_JOINT_MOVEMENT"
        max_lin_dist_key = "MAX_LIN_DIST"
        max_ang_dist_key = "MAX_ANG_DIST"

        super().__init__(
            config,
            spot,
            stopwatch=None,
            max_joint_movement_key=max_joint_movement_key,
            max_lin_dist_key=max_lin_dist_key,
            max_ang_dist_key=max_ang_dist_key,
        )
        self.target_obj_name = None

    def reset(self, *args, **kwargs):
        # Move arm to initial configuration
        cmd_id = self.spot.set_arm_joint_positions(
            positions=self.initial_arm_joint_angles, travel_time=1
        )
        self.spot.block_until_arm_arrives(cmd_id, timeout_sec=1)

        self.spot.open_gripper()

        # Update target object name as provided in config
        observations = super().reset(*args, **kwargs)
        rospy.set_param("object_target", "drawer handle")

        return observations

    def step(self, base_action=None, arm_action=None, grasp=False, place=False):
        grasp = self.should_grasp()
        print("lock:", self.locked_on_object_count)
        # TODO here, to minitor the lock time
        # if  self.locked_on_object_count:
        #     breakpoint()
        observations, reward, done, info = super().step(
            base_action,
            arm_action,
            grasp,
            place,
        )
        return observations, reward, done, info

    def get_observations(self):
        arm_depth, arm_depth_bbox = self.get_gripper_images()
        current_gripper_location = self.spot.get_ee_pos_in_body_frame()[0]
        observations = {
            "joint": self.get_arm_joints(),
            "articulated_agent_arm_depth": arm_depth,
            "is_holding": np.zeros((1,)),
            "ee_pos": current_gripper_location,
        }
        return observations

    def get_success(self, observations):
        return self.grasp_attempted
