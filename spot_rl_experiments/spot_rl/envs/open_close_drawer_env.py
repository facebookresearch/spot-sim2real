# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import sys
from typing import Any, Dict

import magnum as mn
import numpy as np
import quaternion
import rospy
from spot_rl.envs.base_env import SpotBaseEnv
from spot_wrapper.spot import Spot


class SpotOpenCloseDrawerEnv(SpotBaseEnv):
    def __init__(self, config, spot: Spot):
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
        # TODO: finalize what is the target object name
        # possible candidate: drawer handle/cup/purple cube
        self.target_obj_name = "cup"
        self.ee_gripper_offset = mn.Vector3(config.EE_GRIPPER_OFFSET)

        # The initial joint angles is in the stow location
        self.initial_arm_joint_angles = np.deg2rad([0, -180, 180, 0, 0, 0])

    def reset(self, *args, **kwargs):

        # Move arm to initial configuration
        cmd_id = self.spot.set_arm_joint_positions(
            positions=self.initial_arm_joint_angles, travel_time=1
        )

        # Make the arm to be in true nominal location
        # ee_position = self.get_gripper_position_in_base_frame_spot()
        # self.spot.move_gripper_to_point(ee_position, [0, 0, 0])

        # Block until arm arrives with incremental timeout for 3 attempts
        timeout_sec = 1.0
        max_allowed_timeout_sec = 3.0
        status = False
        while status is False and timeout_sec <= max_allowed_timeout_sec:
            status = self.spot.block_until_arm_arrives(cmd_id, timeout_sec=timeout_sec)
            timeout_sec += 1.0

        self.initial_ee_orientation = self.spot.get_ee_rotation_in_body_frame_quat()

        print("Open gripper called in OpenCloseDrawer")
        self.spot.open_gripper()

        # Update target object name as provided in config
        observations = super().reset(
            target_obj_name=self.target_obj_name, *args, **kwargs
        )
        rospy.set_param("object_target", self.target_obj_name)

        return observations

    def step(self, action_dict: Dict[str, Any]):
        # Update the action_dict with place flag
        action_dict["place"] = False
        observations, reward, done, info = super().step(
            action_dict=action_dict,
        )
        # We close gripper here
        # TODO: clean up debug msg
        print(f" action_dict: {action_dict}")
        if action_dict["close_gripper"] >= 0:
            self.spot.close_gripper()
        return observations, reward, done, info

    def angle_between_quat(self, q1, q2):
        q1_inv = np.conjugate(q1)
        dp = quaternion.as_float_array(q1_inv * q2)
        return 2 * np.arctan2(np.linalg.norm(dp[1:]), np.abs(dp[0]))

    def angle_to_forward(self, x, y):
        if np.linalg.norm(x) != 0:
            x_norm = x / np.linalg.norm(x)
        else:
            x_norm = x

        if np.linalg.norm(y) != 0:
            y_norm = y / np.linalg.norm(y)
        else:
            y_norm = y

        return np.arccos(np.clip(np.dot(x_norm, y_norm), -1, 1))

    def get_angle(self, rel_pos):
        """Get angle"""
        forward = np.array([1.0, 0, 0])
        rel_pos = np.array(rel_pos)
        forward = forward[[0, 1]]
        rel_pos = rel_pos[[0, 1]]

        heading_angle = self.angle_to_forward(forward, rel_pos)
        c = np.cross(forward, rel_pos) < 0
        if not c:
            heading_angle = -1.0 * heading_angle
        return heading_angle

    def get_cur_ee_orientation_offset(self):
        # Get base to hand's transformation
        ee_transform = self.spot.get_magnum_Matrix4_spot_a_T_b("vision", "hand")
        # Get the base transformation
        base_transform = self.spot.get_magnum_Matrix4_spot_a_T_b("vision", "body")
        # Do offset: move the base center forward to be close to the gripper base
        base_transform.translation = base_transform.transform_point(
            mn.Vector3(0.292, 0, 0)
        )
        # Get ee relative to base
        ee_position = (base_transform.inverted() @ ee_transform).translation
        base_T_ee_yaw = self.get_angle(ee_position)
        return base_T_ee_yaw

    def get_observations(self):
        # Get the depth images and handle bounding box
        arm_depth, arm_depth_bbox = self.get_gripper_images()

        # Get the delta ee orientation to the initial orientation
        current_ee_orientation = self.spot.get_ee_rotation_in_body_frame_quat()
        delta_ee = np.array(
            self.angle_between_quat(
                self.initial_ee_orientation, current_ee_orientation
            ),
            dtype=np.float32,
        )

        # Remove the offset from the base to ee
        delta_ee = np.array(
            [delta_ee - abs(self.get_cur_ee_orientation_offset())], dtype=np.float32
        )

        # Construct the observation
        observations = {
            "articulated_agent_arm_depth": arm_depth,
            "joint": self.get_arm_joints(self.config.JOINT_BLACKLIST_OPEN_CLOSE_DRAWER),
            "ee_pos": self.get_gripper_position_in_base_frame_spot(),
            "is_holding": np.zeros((1,)),
            "handle_bbox": arm_depth_bbox,
            "art_pose_delta_sensor": delta_ee,
        }
        # TODO: clean up the debug msg
        print(
            f"ee_pos: {self.get_gripper_position_in_base_frame_spot()}; pose_delta: {delta_ee}"
        )
        return observations

    def get_success(self, observations):
        # TODO: better way to handle this
        return False
