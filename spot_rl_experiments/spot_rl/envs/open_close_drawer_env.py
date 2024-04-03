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
from spot_rl.utils.heuristic_nav import get_3d_point
from spot_wrapper.spot import Spot, image_response_to_cv2, scale_depth_img
from spot_wrapper.utils import angle_between_quat


class SpotOpenCloseDrawerEnv(SpotBaseEnv):
    def __init__(self, config, spot: Spot):
        # Select suitable keys
        max_joint_movement_key = "MAX_JOINT_MOVEMENT_OPEN_CLOSE_DRAWER"
        max_lin_dist_key = "MAX_LIN_DIST_OPEN_CLOSE_DRAWER"
        max_ang_dist_key = "MAX_ANG_DIST_OPEN_CLOSE_DRAWER"

        super().__init__(
            config,
            spot,
            stopwatch=None,
            max_joint_movement_key=max_joint_movement_key,
            max_lin_dist_key=max_lin_dist_key,
            max_ang_dist_key=max_ang_dist_key,
        )

        self.ee_gripper_offset = mn.Vector3(config.EE_GRIPPER_OFFSET)

        # The initial joint angles is in the stow location
        self.initial_arm_joint_angles = np.deg2rad([0, -180, 180, 0, 0, 0])

        # The arm joint min max overwrite
        self.arm_lower_limits = np.deg2rad(config.ARM_LOWER_LIMITS_OPEN_CLOSE_DRAWER)
        self.arm_upper_limits = np.deg2rad(config.ARM_UPPER_LIMITS_OPEN_CLOSE_DRAWER)

        # Flag for done
        self._success = False

        # Mode for opening or closing
        self._mode = "open"

        # Distance threshold to call IK to approach the drawers
        self._dis_threshold_ee_to_handle = (
            config.OPEM_CLOSE_DRAWER_DISTANCE_BETWEEN_EE_HANDLE
        )

    def reset(self, goal_dict=None, *args, **kwargs):
        self.spot.open_gripper()

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

        # Make the arm to be in true nominal location
        ee_position = self.get_gripper_position_in_base_frame_spot()
        self.spot.move_gripper_to_point(ee_position, [0, 0, 0])

        # Move arm to initial configuration again to ensure it is in the good location
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

        self.initial_ee_orientation = self.spot.get_ee_quaternion_in_body_frame()

        # Update target object name as provided in config
        observations = super().reset(target_obj_name="drawer handle", *args, **kwargs)
        rospy.set_param("object_target", self.target_obj_name)

        # Flag for done
        self._success = False

        # Get the mode: open or close drawers
        self._mode = goal_dict["mode"]

        return observations

    def compute_distance_to_handle(self):
        "Compute the distance in the bounding box center"
        return (
            self.target_object_distance,
            self.obj_center_pixel[0],
            self.obj_center_pixel[1],
        )

    def bd_open_drawer_api(self):
        """BD API to open the drawer in x direction"""
        raise NotImplementedError

    def approach_handle_and_grasp(self, z, pixel_x, pixel_y):
        """This method does IK to approach the handle and close the gripper."""
        imgs = self.spot.get_hand_image()

        # Get the camera intrinsics
        cam_intrinsics = imgs[0].source.pinhole.intrinsics

        # Get the transformation
        vision_T_base = self.spot.get_magnum_Matrix4_spot_a_T_b("vision", "body")

        # Get the 3D point in the hand RGB frame
        point_in_hand_image_3d = get_3d_point(cam_intrinsics, (pixel_x, pixel_y), z)

        # Get the vision to hand
        vision_T_hand_image: mn.Matrix4 = self.spot.get_magnum_Matrix4_spot_a_T_b(
            "vision", "hand_color_image_sensor", imgs[0].shot.transforms_snapshot
        )
        point_in_global_3d = vision_T_hand_image.transform_point(
            mn.Vector3(*point_in_hand_image_3d)
        )

        # Get the transformation of the gripper
        vision_T_hand = self.spot.get_magnum_Matrix4_spot_a_T_b("vision", "hand")
        # Get the location relative to the gripper
        point_in_hand_3d = vision_T_hand.inverted().transform_point(point_in_global_3d)
        # Offset the x and z direction in hand frame
        ee_offset_x = 0.05
        ee_offset_z = -0.05
        point_in_hand_3d[0] += ee_offset_x
        point_in_hand_3d[2] += ee_offset_z
        # Make it back to global frame
        point_in_global_3d = vision_T_hand.transform_point(point_in_hand_3d)

        # Get the point in the base frame
        point_in_base_3d = vision_T_base.inverted().transform_point(point_in_global_3d)

        # Make it to be numpy
        point_in_base_3d = np.array(
            [
                point_in_base_3d.x,
                point_in_base_3d.y,
                point_in_base_3d.z,
            ]
        )

        # Get the current ee rotation in body frame
        ee_rotation = self.spot.get_ee_quaternion_in_body_frame()

        # Move the gripper to target using current gripper pose in the body frame
        # while maintaining the gripper orientation
        self.spot.move_gripper_to_point(
            point_in_base_3d,
            [ee_rotation.w, ee_rotation.x, ee_rotation.y, ee_rotation.z],
        )

        # Close the gripper
        self.spot.close_gripper()

        # Get the transformation of the gripper
        vision_T_hand = self.spot.get_magnum_Matrix4_spot_a_T_b("vision", "hand")
        # Get the location that we want to move to for retracting/moving forward the arm. Pull/push the drawer by 20 cm
        pull_push_distance = -0.2 if self._mode == "open" else 0.25
        move_target = vision_T_hand.transform_point(
            mn.Vector3([pull_push_distance, 0, 0])
        )
        # Get the move_target in base frame
        move_target = vision_T_base.inverted().transform_point(move_target)

        # Retract the arm based on the current gripper location
        self.spot.move_gripper_to_point(
            move_target, [ee_rotation.w, ee_rotation.x, ee_rotation.y, ee_rotation.z]
        )

        # Open the gripper and retract the arm
        self.spot.open_gripper()
        # [0.55, 0, 0.27] is the gripper nominal location
        # [0,0,0] is the roll pitch yaw
        self.spot.move_gripper_to_point([0.55, 0, 0.27], [0, 0, 0])

        # Change the flag to finish
        self._success = True
        rospy.set_param("is_tracking_enabled", False)

    def step(self, action_dict: Dict[str, Any]):

        # Update the action_dict with place flag
        action_dict["place"] = False
        observations, reward, done, info = super().step(
            action_dict=action_dict,
        )

        # Get bounding box
        bbox = observations["handle_bbox"]

        # Compute the distance from the gripper to bounding box
        # The distance is called z here
        z = float("inf")
        # We only compute the distance if bounding box detects something
        if np.sum(bbox) > 0:
            rospy.set_param("is_tracking_enabled", True)
            z, pixel_x, pixel_y = self.compute_distance_to_handle()

        # We close gripper here
        if z != 0 and z < self._dis_threshold_ee_to_handle:
            # Do IK to approach the target
            self.approach_handle_and_grasp(z, pixel_x, pixel_y)
            # If we can do IK, then we call it successful
            done = self._success

        return observations, reward, done, info

    def get_observations(self):
        # Get the depth images and handle bounding box
        arm_depth, arm_depth_bbox = self.get_gripper_images()

        # Get the delta ee orientation to the initial orientation
        current_ee_orientation = self.spot.get_ee_quaternion_in_body_frame()
        delta_ee = np.array(
            angle_between_quat(self.initial_ee_orientation, current_ee_orientation),
            dtype=np.float32,
        )

        # Remove the offset from the base to ee
        delta_ee = np.array(
            [delta_ee - abs(self.spot.get_cur_ee_pose_offset())], dtype=np.float32
        )

        # Construct the observation
        observations = {
            "articulated_agent_arm_depth": arm_depth,
            "joint": self.get_arm_joints(self.config.JOINT_BLACKLIST_OPEN_CLOSE_DRAWER),
            "ee_pos": self.get_gripper_position_in_base_frame_spot(),
            "handle_bbox": arm_depth_bbox,
            "art_pose_delta_sensor": delta_ee,
        }
        return observations

    def get_success(self, observations=None):
        return self._success
