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
from spot_wrapper.spot import Spot, image_response_to_cv2, scale_depth_img


def get_3d_point(cam_intrinsics, pixel_uv, z):
    """Helper function to compute 3D point"""
    # Get camera intrinsics
    fx = cam_intrinsics.focal_length.x
    fy = cam_intrinsics.focal_length.y
    cx = cam_intrinsics.principal_point.x
    cy = cam_intrinsics.principal_point.y

    # print(fx, fy, cx, cy)
    # Get 3D point
    x = (pixel_uv[0] - cx) * z / fx
    y = (pixel_uv[1] - cy) * z / fy
    return np.array([x, y, z])


class SpotOpenCloseDrawerEnv(SpotBaseEnv):
    def __init__(self, config, spot: Spot):
        # Select suitable keys
        max_joint_movement_key = "MAX_JOINT_MOVEMENT_OPEN_CLOSE_DRAWER"
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

        # The number of times calls close gripper
        self._ee_close_times = 0

        # Flag for done
        self._done = False

    def reset(self, *args, **kwargs):

        print("Open gripper called in OpenCloseDrawer")
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

        self.initial_ee_orientation = self.spot.get_ee_rotation_in_body_frame_quat()

        # Update target object name as provided in config
        observations = super().reset(
            target_obj_name=self.target_obj_name, *args, **kwargs
        )
        rospy.set_param("object_target", self.target_obj_name)

        # The number of times calls close gripper
        self._ee_close_times = 0

        # Flag for done
        self._done = False

        return observations

    def compute_distance_to_handle(self, bbox, average_mode=True):
        "Compute the distance in the bounding box center"
        imgs = self.spot.get_hand_image()
        unscaled_dep_img = image_response_to_cv2(imgs[1])

        # Locate the bbox location
        height_center_bbox = bbox.shape[0] // 2  # 240 //2
        width_center_bbox = bbox.shape[1] // 2  # 228 //2
        bbox_where = np.argwhere(bbox[:, :, 0])
        (x_min, y_min), (x_max, y_max) = bbox_where.min(0), bbox_where.max(0) + 1
        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2

        # Offset in the bbox
        delta_x = -height_center_bbox + x_center
        delta_y = -width_center_bbox + y_center

        # Center of depth image
        height_center = unscaled_dep_img.shape[0] // 2  # 480 //2
        width_center = unscaled_dep_img.shape[1] // 2  # 640 //2

        # Get the z depth
        if average_mode:
            # x_offset. y_offset
            x_offset = height_center - height_center_bbox
            y_offset = width_center - width_center_bbox
            x_min_depth = x_offset + x_min
            y_min_depth = y_offset + y_min
            x_max_depth = x_offset + x_max
            y_max_depth = y_offset + y_max
            z = (
                np.average(
                    unscaled_dep_img[x_min_depth:x_max_depth, y_min_depth:y_max_depth]
                )
                * 0.001
            )  # from mm to meter
        else:
            # Center of the depth image with offset
            z = (
                unscaled_dep_img[height_center + delta_x, width_center + delta_y]
                * 0.001
            )  # from mm to meter
        return z, height_center + delta_x, width_center + delta_y

    def approach_handle_and_grasp(self, z, pixel_x, pixel_y):
        """This method doing IK to approach the handle and close the gripper."""
        imgs = self.spot.get_hand_image()

        # Get the camera intrinsics
        cam_intrinsics = imgs[0].source.pinhole.intrinsics

        # Get the transformation
        body_T_hand: mn.Matrix4 = self.spot.get_magnum_Matrix4_spot_a_T_b(
            "body", "hand_color_image_sensor", imgs[0].shot.transforms_snapshot
        )

        # Get the 3D point
        point_in_local_3d = get_3d_point(cam_intrinsics, (pixel_x, pixel_y), z)

        # Get the point in the hand frame
        point_in_global_3d: mn.Vector3 = body_T_hand.transform_point(
            mn.Vector3(*point_in_local_3d)
        )
        # Small offset to move the gripper forward
        offset_x = 0.1
        point_in_global_3d = np.array(
            [
                point_in_global_3d.x + offset_x,
                point_in_global_3d.y,
                point_in_global_3d.z,
            ]
        )

        # Move the gripper to target
        self.spot.move_gripper_to_point(point_in_global_3d, [0, 0, 0])

        # Close the gripper
        self.spot.close_gripper()

        # Pull the drawer by 40 cm
        move_target = [
            point_in_global_3d[0] - 0.4,
            point_in_global_3d[1],
            point_in_global_3d[2],
        ]
        self.spot.move_gripper_to_point(move_target, [0, 0, 0])

        # Open the gripper and retract the arm
        self.spot.open_gripper()
        self.spot.move_gripper_to_point([0.55, 0, 0.27], [0, 0, 0])

        # Change the flag to finish
        self._done = True

    def step(self, action_dict: Dict[str, Any]):
        # Update the action_dict with place flag
        action_dict["place"] = False
        observations, reward, done, info = super().step(
            action_dict=action_dict,
        )

        # Get bounding box
        bbox = observations["handle_bbox"]

        # Compute the distance from the gripper to bounding box
        z = float("inf")
        if np.sum(bbox) > 0:
            z, pixel_x, pixel_y = self.compute_distance_to_handle(bbox)
        print(f"distance to bbox {z}")

        # We close gripper here
        # TODO: clean up debug msg
        print(f" action_dict: {action_dict} {self._ee_close_times}")
        if (action_dict["close_gripper"] >= 0 and np.sum(bbox) > 0 and False) or (
            z != 0 and z <= 0.3
        ):
            self._ee_close_times += 1
            # Do IK to approach the target
            self.approach_handle_and_grasp(z, pixel_x, pixel_y)

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
            # TODO: ckpt 12 series does not have is_holding sensor
            # "is_holding": np.zeros((1,)),
            "handle_bbox": arm_depth_bbox,
            "art_pose_delta_sensor": delta_ee,
        }
        # TODO: clean up the debug msg
        print(
            f"ee_pos: {self.get_gripper_position_in_base_frame_spot()}; pose_delta: {delta_ee}"
        )
        return observations

    def get_success(self, observations):
        return self._done
