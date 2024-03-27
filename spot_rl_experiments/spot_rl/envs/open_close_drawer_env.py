# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import sys
import time
from math import tan
from termios import VEOL
from typing import Any, Dict

import magnum as mn
import numpy as np
import quaternion
import rospy
from bosdyn.api import basic_command_pb2, geometry_pb2
from bosdyn.client.robot_command import RobotCommandBuilder
from google.protobuf import wrappers_pb2  # type: ignore
from spot_rl.envs.base_env import SpotBaseEnv
from spot_rl.utils.heuristic_nav import get_3d_point
from spot_wrapper.spot import Spot, image_response_to_cv2, scale_depth_img
from spot_wrapper.utils import angle_between_quat

POSITION_MODE = (
    basic_command_pb2.ConstrainedManipulationCommand.Request.CONTROL_MODE_POSITION
)
VELOCITY_MODE = (
    basic_command_pb2.ConstrainedManipulationCommand.Request.CONTROL_MODE_VELOCITY
)


# This function is used to scale the velocity limit given
# the force limit. This scaling ensures that when the measured arm
# velocity is zero but desired velocity is max (vel_limit), we request
# max (force_limit) amount of force in that direction.
def scale_velocity_lim_given_force_lim(force_limit):
    internal_vel_tracking_gain = 7000.0 / 333.0
    vel_limit = force_limit / internal_vel_tracking_gain
    return vel_limit


# This function is used to scale the rotational velocity limit given
# the torque limit. This scaling ensures that when the measured arm
# velocity is zero but desired velocity is max (vel_limit), we request
# max (torque_limit) amount of torque in that direction.
def scale_rot_velocity_lim_given_torque_lim(torque_limit):
    internal_vel_tracking_gain = 300.0 / 333.0
    vel_limit = torque_limit / internal_vel_tracking_gain
    return vel_limit


def get_position_and_vel_values(
    target_position,
    velocity_normalized,
    force_or_torque_limit,
    position_control,
    pure_rot_move=False,
):
    position_sign = 1
    position_value = 0
    if target_position is not None:
        position_sign = np.sign(target_position)
        position_value = abs(target_position)

    # Scale the velocity in a way to ensure we hit force_limit when arm is not moving but velocity_normalized is max.
    velocity_normalized = max(min(velocity_normalized, 1.0), -1.0)
    if not pure_rot_move:
        velocity_limit_from_force = scale_velocity_lim_given_force_lim(
            force_or_torque_limit
        )
        # Tangential velocity in units of m/s
        velocity_with_unit = velocity_normalized * velocity_limit_from_force
    else:
        velocity_limit_from_torque = scale_rot_velocity_lim_given_torque_lim(
            force_or_torque_limit
        )
        # Rotational velocity in units or rad/s
        velocity_with_unit = velocity_limit_from_torque * velocity_normalized

    if position_control:
        if target_position is None:
            print(
                "Error! In position control mode, target_position must be set. Exiting."
            )
            return
        # For position moves, the velocity is treated as an unsigned velocity limit
        velocity_with_unit = abs(velocity_with_unit)

    return position_sign, position_value, velocity_with_unit


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

        # Flag for using API to open the cabinet or not
        self._use_bd_api = False

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

        # Get the receptacle type
        self._rep_type = goal_dict["rep_type"]

        assert self._rep_type in [
            "drawer",
            "cabinet",
        ], f"Do not support repcetacle type {self._rep_type} in open/close skills"

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
        command = self.construct_cabinet_task(
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
        """Herusitics to open the drawer"""
        # Get the transformation
        vision_T_base = self.spot.get_magnum_Matrix4_spot_a_T_b("vision", "body")
        ee_rotation = self.spot.get_ee_quaternion_in_body_frame()

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

    def open_cabinet(self):
        """Herustics to open the cabinet"""
        # Get the location of the rotataional axis
        base_T_hand = self.spot.get_magnum_Matrix4_spot_a_T_b("body", "hand")

        # Assuming that there is no gripper rotation
        # Assuming that the cabniet door panel width is 0.45
        # Assuming that the door axis is on the left of the hand
        panel_size = 0.45
        base_T_hand.translation = base_T_hand.transform_point(
            mn.Vector3(0.0, 0.0, -panel_size)
        )
        for cur_ang_in_deg in range(5, 60, 5):
            # angle in degree
            cur_ang = np.deg2rad(cur_ang_in_deg)
            # Rotate the trans by this degree
            cur_base_T_hand = base_T_hand @ mn.Matrix4.rotation_y(mn.Rad(-cur_ang))
            # Get the point in that frame
            ee_target_point = cur_base_T_hand.transform_point(
                mn.Vector3(0.0, 0.0, panel_size)
            )
            self.spot.move_gripper_to_point(
                np.array(ee_target_point), [np.pi / 2, -cur_ang * 2, 0.0]
            )
            self.spot.close_gripper()
            print(f"{cur_ang_in_deg} ee pos: {ee_target_point}; yaw: {-cur_ang}")

    def construct_cabinet_task(
        self,
        velocity_normalized,
        force_limit=40,
        target_angle=None,
        position_control=False,
        reset_estimator_bool=True,
    ):
        """Helper function for opening/closing cabinets

        params:
        + velocity_normalized: normalized task tangential velocity in range [-1.0, 1.0]
        In position mode, this normalized velocity is used as a velocity limit for the planned trajectory.
        + force_limit (optional): positive value denoting max force robot will exert along task dimension
        + target_angle: target angle displacement (rad) in task space. This is only used if position_control == True
        + position_control: if False will move the affordance in velocity control, if True will move by target_angle
        with a max velocity of velocity_limit
        + reset_estimator_bool: boolean that determines if the estimator should compute a task frame from scratch.
        Only set to False if you want to re-use the estimate from the last constrained manipulation action.

        Output:
        + command: api command object

        Notes:
        In this function, we assume the initial motion of the cabinet is
        along the x-axis of the hand (forward and backward). If the initial
        grasp is such that the initial motion needs to be something else,
        change the force direction.
        """
        angle_sign, angle_value, tangential_velocity = get_position_and_vel_values(
            target_angle, velocity_normalized, force_limit, position_control
        )

        frame_name = "hand"
        force_lim = force_limit
        # Setting a placeholder value that doesn't matter, since we don't
        # apply a pure torque in this task.
        torque_lim = 5.0
        force_direction = geometry_pb2.Vec3(x=angle_sign * -1.0, y=0.0, z=0.0)
        torque_direction = geometry_pb2.Vec3(x=0.0, y=0.0, z=0.0)
        init_wrench_dir = geometry_pb2.Wrench(
            force=force_direction, torque=torque_direction
        )
        task_type = (
            basic_command_pb2.ConstrainedManipulationCommand.Request.TASK_TYPE_R3_CIRCLE_FORCE
        )
        reset_estimator = wrappers_pb2.BoolValue(value=reset_estimator_bool)
        control_mode = POSITION_MODE if position_control else VELOCITY_MODE

        command = RobotCommandBuilder.constrained_manipulation_command(
            task_type=task_type,
            init_wrench_direction_in_frame_name=init_wrench_dir,
            force_limit=force_lim,
            torque_limit=torque_lim,
            tangential_speed=tangential_velocity,
            frame_name=frame_name,
            control_mode=control_mode,
            target_angle=angle_value,
            reset_estimator=reset_estimator,
        )
        return command

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
        ee_offset_x = 0.05 if self._rep_type == "drawer" else 0.0
        ee_offset_z = -0.05 if self._rep_type == "drawer" else 0.0
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

        # For the cabnet part: rotation the gripper by 90 degree
        if self._rep_type == "cabinet":
            self.spot.move_gripper_to_point(
                point_in_base_3d,
                [np.pi / 2, 0, 0],
            )

        # Close the gripper
        self.spot.close_gripper()
        time.sleep(2)

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
