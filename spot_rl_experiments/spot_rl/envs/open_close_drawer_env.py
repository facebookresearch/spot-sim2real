# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import sys
import time
from typing import Any, Dict

import cv2
import magnum as mn
import numpy as np
import quaternion
import rospy

try:
    import sophuspy as sp
except Exception as e:
    print(f"Cannot import sophuspy due to {e}. Import sophus instead")
    import sophus as sp
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME
from bosdyn.client.math_helpers import Quat, SE3Pose
from spot_rl.envs.base_env import SpotBaseEnv
from spot_rl.utils.pixel_to_3d_conversion_utils import (
    get_3d_point,
    project_3d_to_pixel_uv,
    sample_patch_around_point,
)
from spot_rl.utils.rospy_light_detection import detect_with_rospy_subscriber
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

        # Get the receptacle type
        self._rep_type = "drawer"

        # Distance threshold to call IK to approach the drawers
        self._dis_threshold_ee_to_handle = (
            config.OPEM_CLOSE_DRAWER_DISTANCE_BETWEEN_EE_HANDLE
        )

        # Flag for using Boston Dynamics API to open the cabinet or not
        self._use_bd_api = False

        # Get the cabinet door location
        self._cab_door = "left"

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
        rospy.set_param("enable_tracking", True)

        # Flag for done
        self._success = False

        # Get the mode: open or close drawers
        self._mode = goal_dict["mode"]

        # Get the receptacle type
        self._rep_type = goal_dict["rep_type"]

        # Get the cabinet door location
        self._cab_door = goal_dict["cab_door"]

        return observations

    def compute_distance_to_handle(self):
        "Compute the distance in the bounding box center"
        return (
            self.target_object_distance,
            self.obj_center_pixel[0],
            self.obj_center_pixel[1],
        )

    def bd_open_drawer_api(self):
        """BD API to open the drawer"""
        raise NotImplementedError

    def bd_open_cabinet_api(self):
        """BD API to open the cabinet"""
        command = self.spot.construct_cabinet_task(
            0.25, force_limit=40, target_angle=1.74, position_control=True
        )
        task_duration = 10000000
        command.full_body_command.constrained_manipulation_request.end_time.CopyFrom(
            self.spot.robot.time_sync.robot_timestamp_from_local_secs(
                time.time() + task_duration
            )
        )
        self.spot.command_client.robot_command_async(command)
        time.sleep(10)

    def open_drawer(self):
        """Heuristics to open the drawer"""
        # Get the transformation
        vision_T_base = self.spot.get_magnum_Matrix4_spot_a_T_b("vision", "body")
        ee_rotation = self.spot.get_ee_quaternion_in_body_frame()

        # Get the transformation of the gripper
        vision_T_hand = self.spot.get_magnum_Matrix4_spot_a_T_b("vision", "hand")
        # Get the location that we want to move to for retracting/moving forward the arm. Pull/push the drawer by 20 cm
        pull_push_distance = -0.2 if self._mode == "open" else 0.4
        move_target = vision_T_hand.transform_point(
            mn.Vector3([pull_push_distance, 0, 0])
        )
        # Get the move_target in base frame
        move_target = vision_T_base.inverted().transform_point(move_target)

        # Retract the arm based on the current gripper location
        # This doesn't stop coming back because of arm follow command, check with BD ?
        print(
            f"self.get_gripper_position_in_base_frame_spot() {self.get_gripper_position_in_base_frame_spot()}"
        )
        print(f"move_target {move_target}")
        # Set the robot base velocity to reset the base motion
        self.spot.set_base_velocity(x_vel=0, y_vel=0, ang_vel=0, vel_time=0.8)
        self.spot.move_gripper_to_point(
            move_target,
            [ee_rotation.w, ee_rotation.x, ee_rotation.y, ee_rotation.z],
            seconds_to_goal=7,
        )

    def open_cabinet(self):
        """Heuristics to open the cabinet"""
        # Get the location of the rotataional axis
        base_T_hand = self.spot.get_magnum_Matrix4_spot_a_T_b("body", "hand")

        # Assuming that the cabinet door's size is panel_size
        # Assuming that the door axis is on the left side of the hand
        panel_size = 0.55
        if self._cab_door == "right":
            panel_size = -panel_size

        base_T_hand.translation = base_T_hand.transform_point(
            mn.Vector3(0.0, 0.0, -panel_size)
        )
        target_degree = 70
        interval = 10
        # Loop over to create a circular motion for the gripper
        for cur_ang_in_deg in range(10, target_degree + 10, interval):
            if cur_ang_in_deg < target_degree:
                # Keep closing the gripper to grasp the handle tightly
                self.spot.close_gripper()
            else:
                # In the final stage, we open the gripper to let gripper be away from the handle
                self.spot.open_gripper()
            # Angle in degree
            cur_ang = np.deg2rad(cur_ang_in_deg)

            # For right side of the hand
            if self._cab_door == "right":
                cur_ang = -cur_ang

            # Rotate the trans by this degree
            cur_base_T_hand = base_T_hand @ mn.Matrix4.rotation_y(mn.Rad(-cur_ang))
            # Get the point in that frame
            ee_target_point = cur_base_T_hand.transform_point(
                mn.Vector3(0.0, 0.0, panel_size)
            )
            # 1.5 to scale up the angle for tracking the circular motion better
            self.spot.move_gripper_to_point(
                np.array(ee_target_point), [np.pi / 2, -cur_ang * 1.5, 0.0]
            )
            print(f"Deg:{cur_ang_in_deg}; ee pos: {ee_target_point}; yaw: {-cur_ang}")

        # Robot backing up a bit to avoid gripper from colliding with the handle
        self.spot.set_base_velocity(
            x_vel=-0.25,
            y_vel=0,
            ang_vel=0,
            vel_time=0.8,
        )

    def approach_handle_and_grasp(self):
        """This method does IK to approach the handle and close the gripper to grasp the handle."""

        ########################################################
        ### Step 1: Get the location of handle in hand frame ###
        ########################################################
        # Always get the latest detection, gripper moves a lot and loses sight
        _, pixel_x, pixel_y = self.compute_distance_to_handle()
        imgs = self.spot.get_hand_image()

        image_rgb = image_response_to_cv2(imgs[0])
        depth_raw = image_response_to_cv2(imgs[1])

        # Always get the latest detection, gripper moves a lot and loses sight, maybe comment later
        x1, y1, x2, y2 = detect_with_rospy_subscriber(
            rospy.get_param("/object_target", "drawer handle"), self.config.IMAGE_SCALE
        )
        pixel_x, pixel_y = (x1 + x2) // 2, (y1 + y2) // 2

        z = sample_patch_around_point(pixel_x, pixel_y, depth_raw) * 1e-3

        # Get the camera intrinsics
        cam_intrinsics = imgs[0].source.pinhole.intrinsics

        # z -= 0.05 offset to do pinch grasping
        point_in_hand_image_3d = get_3d_point(cam_intrinsics, (pixel_x, pixel_y), z)

        body_T_hand: mn.Matrix4 = self.spot.get_magnum_Matrix4_spot_a_T_b(
            GRAV_ALIGNED_BODY_FRAME_NAME,
            "link_wr1",
        )
        hand_T_gripper: mn.Matrix4 = self.spot.get_magnum_Matrix4_spot_a_T_b(
            "arm0.link_wr1",
            "hand_color_image_sensor",
            imgs[0].shot.transforms_snapshot,
        )
        body_T_gripper: mn.Matrix4 = body_T_hand @ hand_T_gripper
        point_in_base_3d = body_T_gripper.transform_point(
            mn.Vector3(*point_in_hand_image_3d)
        )

        print(
            f"Drawer point in gripper 3D {point_in_hand_image_3d}, point in body {point_in_base_3d}"
        )

        # Debug -- verify point in image, comment later
        image_rgb = cv2.circle(
            image_rgb, (int(pixel_x), int(pixel_y)), radius=4, color=(0, 0, 255)
        )
        cv2.namedWindow("Handle", cv2.WINDOW_NORMAL)
        cv2.imshow("Handle", image_rgb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        point_in_base_3d = mn.Vector3(*point_in_base_3d)

        # Make it to be numpy
        point_in_base_3d = np.array(
            [
                point_in_base_3d.x,
                point_in_base_3d.y,
                point_in_base_3d.z,
            ]
        )

        ###############################################
        ### Step 2: Move the gripper to that handle ###
        ###############################################

        # Get the current ee rotation in body frame
        # Fixed quaternion for better pose
        ee_rotation = quaternion.quaternion(
            0.99998224, 0.00505713, 0.00285832, 0.00132725
        )  # self.spot.get_ee_quaternion_in_body_frame()

        # For the cabnet part: rotation the gripper by 90 degree
        if self._rep_type == "cabinet":
            self.spot.move_gripper_to_point(
                point_in_base_3d,
                [np.pi / 2, 0, 0],
            )
        elif self._rep_type == "drawer":
            # Move the gripper to target using current gripper pose in the body frame
            # while maintaining the gripper orientation
            is_point_reachable_without_mobility = (
                self.spot.query_IK_reachability_of_gripper(
                    SE3Pose(
                        *point_in_base_3d,
                        Quat(
                            ee_rotation.w, ee_rotation.x, ee_rotation.y, ee_rotation.z
                        ),
                    )
                )
            )

            self.spot.move_arm_to_point_with_body_follow(
                [point_in_base_3d],
                [[ee_rotation.w, ee_rotation.x, ee_rotation.y, ee_rotation.z]],
                seconds=5,
                allow_body_follow=not is_point_reachable_without_mobility,
                arm_offset=[0.9, 0, 0.35],
            )

        #################################
        ### Step 3: Close the gripper ###
        #################################
        self.spot.close_gripper()
        # Pause a bit to ensure the gripper grapes the handle
        time.sleep(2)

        ############################################
        ### Step 4: Execute post-grasping motion ###
        ############################################

        if self._rep_type == "cabinet":
            # Call API to open cab
            if self._use_bd_api:
                self.bd_open_cabinet_api()
            else:
                self.open_cabinet()
        elif self._rep_type == "drawer":
            # Call API to open drawer
            if self._use_bd_api:
                self.bd_open_drawer_api()
            else:
                self.open_drawer()

        #############################
        ### Step 5: Reset the arm ###
        #############################

        # Open the gripper and retract the arm
        self.spot.open_gripper()
        # [0.55, 0, 0.27] is the gripper nominal location
        # [0,0,0] is the roll pitch yaw
        self.spot.move_gripper_to_point([0.55, 0, 0.27], [0, 0, 0])

        # Change the flag to finish
        self._success = True

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
            z, pixel_x, pixel_y = self.compute_distance_to_handle()

        # We close gripper here
        if z != 0 and z < self._dis_threshold_ee_to_handle:
            # Do IK to approach the target
            self.approach_handle_and_grasp()
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
