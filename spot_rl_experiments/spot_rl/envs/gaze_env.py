# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import sys
import time
from typing import Any, Dict

import magnum as mn
import numpy as np
import rospy
from spot_rl.envs.base_env import SpotBaseEnv
from spot_rl.utils.heuristic_nav import get_3d_point
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

        # Block until arm arrives with incremental timeout for 3 attempts
        timeout_sec = 1.0
        max_allowed_timeout_sec = 3.0
        status = False
        while status is False and timeout_sec <= max_allowed_timeout_sec:
            status = self.spot.block_until_arm_arrives(cmd_id, timeout_sec=timeout_sec)
            timeout_sec += 1.0

        print("Open gripper called in Gaze")
        self.spot.open_gripper()

        # Update target object name as provided in config
        observations = super().reset(target_obj_name=target_obj_name, *args, **kwargs)
        rospy.set_param("object_target", target_obj_name)
        rospy.set_param("is_gripper_blocked", 0)
        return observations

    def step(self, action_dict: Dict[str, Any]):
        grasp = self.should_grasp()

        # Update the action_dict with grasp and place flags
        action_dict["grasp"] = grasp
        action_dict["place"] = False  # TODO: Why is gaze getting flag for place?

        observations, reward, done, info = super().step(
            action_dict=action_dict,
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


class SpotSemanticGazeEnv(SpotBaseEnv):
    def __init__(self, config, spot: Spot):
        # Select suitable keys
        max_joint_movement_key = "MAX_JOINT_MOVEMENT_SEMANTIC_GAZE"
        max_lin_dist_key = "MAX_LIN_DIST_SEMANTIC_GAZE"
        max_ang_dist_key = "MAX_ANG_DIST_SEMANTIC_GAZE"

        super().__init__(
            config,
            spot,
            stopwatch=None,
            max_joint_movement_key=max_joint_movement_key,
            max_lin_dist_key=max_lin_dist_key,
            max_ang_dist_key=max_ang_dist_key,
        )
        self.target_obj_name = None
        self.initial_arm_joint_angles = np.deg2rad(config.GAZE_ARM_JOINT_ANGLES)
        self.grasping_type = "topdown"

    def reset(self, target_obj_name, grasping_type, *args, **kwargs):
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

        print("Open gripper called in Gaze")
        self.spot.open_gripper()

        # Update target object name as provided in config
        observations = super().reset(target_obj_name=target_obj_name, *args, **kwargs)
        rospy.set_param("object_target", target_obj_name)
        rospy.set_param("is_gripper_blocked", 0)
        self.grasping_type = grasping_type
        return observations

    def approach_object(self):
        """Approach the object based on pixel x, y, and the depth image"""
        raw_z, pixel_x, pixel_y = (
            self.target_object_distance,
            self.obj_center_pixel[0],
            self.obj_center_pixel[1],
        )
        while raw_z > 0.05:
            print(f"raw_z: {raw_z} pixel_x: {pixel_x}, pixel_y: {pixel_y}")
            z = raw_z

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
            point_in_hand_3d = vision_T_hand.inverted().transform_point(
                point_in_global_3d
            )
            # Offset the x and z direction in hand frame
            ee_offset_x = 0.0
            ee_offset_z = 0.0
            point_in_hand_3d[0] += ee_offset_x
            point_in_hand_3d[2] += ee_offset_z
            # Make it back to global frame
            point_in_global_3d = vision_T_hand.transform_point(point_in_hand_3d)

            # Get the point in the base frame
            point_in_base_3d = vision_T_base.inverted().transform_point(
                point_in_global_3d
            )

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
            # Recompute the distance
            self.get_gripper_images(save_image=True)
            raw_z, pixel_x, pixel_y = (
                self.target_object_distance,
                self.obj_center_pixel[0],
                self.obj_center_pixel[1],
            )

    def step(self, action_dict: Dict[str, Any]):
        grasp = self.should_grasp(target_object_distance_treshold=0.5)

        if grasp:
            self.approach_object()
            self.get_gripper_images(save_image=True)
            self.spot.close_gripper()
            time.sleep(1.0)
            self.grasp_attempted = True
            arm_positions = np.deg2rad(self.config.PLACE_ARM_JOINT_ANGLES)
            # Record the grasping pose (in roll pitch yaw) of the gripper
            (
                _,
                self.ee_orientation_at_grasping,
            ) = self.spot.get_ee_pos_in_body_frame()
            self.spot.set_arm_joint_positions(positions=arm_positions, travel_time=1.0)
            # wait for arm to return
            time.sleep(1.0)
            # Do not need to call API
            grasp = False

        # Update the action_dict with grasp and place flags
        action_dict["grasp"] = grasp
        action_dict["place"] = False  # TODO: Why is gaze getting flag for place?

        observations, reward, done, info = super().step(
            action_dict=action_dict,
        )
        return observations, reward, done, info

    def remap_observation_keys_for_hab3(self, observations):
        """
        Change observation keys as per hab3.

        @INFO: Policies trained on older hab versions DON'T need remapping
        """
        semantic_gaze_observations = {}
        semantic_gaze_observations["arm_depth_bbox_sensor"] = observations[
            "arm_depth_bbox"
        ]
        semantic_gaze_observations["articulated_agent_arm_depth"] = observations[
            "arm_depth"
        ]
        semantic_gaze_observations["joint"] = observations["joint"]
        return semantic_gaze_observations

    def get_observations(self):
        arm_depth, arm_depth_bbox = self.get_gripper_images()
        observations = {
            "joint": self.get_arm_joints(),
            "arm_depth": arm_depth,
            "arm_depth_bbox": arm_depth_bbox,
        }
        # Remap observation keys for mobile gaze as it was trained with Habitat version3
        observations = self.remap_observation_keys_for_hab3(observations)

        # Get the observation for top down or side grasping
        # Get base to hand's transformation
        ee_T = self.spot.get_magnum_Matrix4_spot_a_T_b("vision", "hand")
        # Get the base transformation
        base_T = self.spot.get_magnum_Matrix4_spot_a_T_b("vision", "body")
        base_to_ee_T = base_T.inverted() @ ee_T
        target_vector = np.array([0, 0, 1.0])
        # Get the direction vector
        dir_vector = np.array(base_to_ee_T.transform_vector(target_vector))

        if self.grasping_type == "topdown":
            delta = 1.0 - abs(dir_vector[0])
        elif self.grasping_type == "side":
            delta = abs(dir_vector[0])
        print(f"delta {delta} {self.grasping_type} {dir_vector}")
        observations["topdown_or_side_grasping"] = np.array(
            [delta],
            dtype=np.float32,
        )
        return observations

    def get_success(self, observations):
        return self.grasp_attempted
