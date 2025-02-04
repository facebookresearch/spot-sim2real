# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import sys
import time
from typing import Any, Dict

import magnum as mn
import numpy as np
import rospy
from spot_rl.envs.base_env import SpotBaseEnv, pad_action, rescale_actions
from spot_wrapper.spot import Spot, wrap_heading


class SpotPickEnv(SpotBaseEnv):
    def __init__(self, config, spot: Spot):
        super().__init__(
            config,
            spot,
        )
        self.grasp_attempted = False

    def reset(self, *args, **kwargs):
        # Move arm to initial configuration
        cmd_id = self.spot.set_arm_joint_positions(
            positions=self.initial_arm_joint_angles, travel_time=1
        )

        # Block until arm arrives with incremental timeout for 3 attempts
        timeout_sec = 1.0
        max_allowed_timeout_sec = 3.0
        status = False
        while status is False and timeout_sec <= max_allowed_timeout_sec:
            status = self.spot.block_until_arm_arrives(cmd_id, timeout_sec=timeout_sec)
            timeout_sec += 1.0

        print("Opening gripper for pick")
        self.spot.open_gripper()

        # Update target object name as provided in config
        observations = super().reset(*args, **kwargs)
        rospy.set_param("is_gripper_blocked", 0)
        self.grasp_attempted = False
        return observations

    def process_base_action(self, base_action):
        base_action = rescale_actions(base_action, silence_only=True)
        if np.count_nonzero(base_action) > 0:
            # Command velocities using the input action
            lin_dist, ang_dist = base_action

            # Scale the linear and angular velocities
            nav_velocity_scaling = rospy.get_param("nav_velocity_scaling", 1.0)
            lin_dist *= self._max_lin_dist_scale * nav_velocity_scaling
            ang_dist *= np.deg2rad(self._max_ang_dist_scale * nav_velocity_scaling)

            target_yaw = wrap_heading(self.yaw + ang_dist)
            # No horizontal velocity
            ctrl_period = 1 / self.ctrl_hz
            # Don't even bother moving if it's just for a bit of distance
            if abs(lin_dist) < 0.05 and abs(ang_dist) < np.deg2rad(3):
                base_action = None
                target_yaw = None
            else:
                base_action = [lin_dist / ctrl_period, 0, ang_dist / ctrl_period]
                self.prev_base_moved = True
        else:
            base_action = None
            self.prev_base_moved = False
        return base_action

    def process_arm_action(self):
        arm_action = rescale_actions(arm_action)
        if np.count_nonzero(arm_action) > 0:
            arm_action *= self._max_joint_movement_scale
            arm_action = self.current_arm_pose + pad_action(arm_action)
            arm_action = np.clip(
                arm_action, self.arm_lower_limits, self.arm_upper_limits
            )
        else:
            arm_action = None
        return arm_action

    def pre_step(self, action_dict):
        # Update the action_dict with grasp and place flags
        base_action = action_dict.get("base_action", None)
        if base_action is not None:
            base_action = self.process_base_action(base_action)
        arm_action = action_dict.get("arm_action", None)
        arm_ee_action = action_dict.get("arm_ee_action", None)
        arm_action = self.process_base_action(arm_action or arm_ee_action)

        grasp = action_dict.get("grasp", False)
        place = action_dict.get("place", False)

        return arm_action, base_action

    def step(self, action_dict: Dict[str, Any]):
        arm_action, base_action = self.pre_step(action_dict)
        self.spot.set_base_vel_and_arm_pos(
            *base_action,
            arm_action,
            travel_time=self.config.ARM_TRAJECTORY_TIME_IN_SECONDS,
            disable_obstacle_avoidance=self.config.DISABLE_OBSTACLE_AVOIDANCE,
        )

        observations, reward, done, info = super().step(
            action_dict=action_dict,
        )
        return observations, reward, done, info

    def get_observations(self):
        observations = {
            "joint": self.get_arm_joints(),
            "arm_rgb": self.get_gripper_images(),
        }

        return observations

    def get_success(self, observations):
        return self.grasp_attempted
