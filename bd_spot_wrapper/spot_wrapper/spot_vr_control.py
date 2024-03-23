# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
import numpy as np
import rospy
from spot_wrapper.spot import Spot
from utils_transformation import (
    angle_between,
    cam_pose_from_opengl_to_opencv,
    cam_pose_from_xzy_to_xyz,
    euler_from_matrix,
    euler_from_quaternion,
    quaternion_from_matrix,
)

# Define the update freq in seconds
UPDATE_PERIOD = 0.2

# Where the gripper goes to upon initialization
INITIAL_POINT = np.array([0.5, 0.0, 0.35])
INITIAL_RPY = np.deg2rad([0.0, 0.0, 0.0])


class Spot_VR_Controller:
    """Control Spot by reading rospy parameters from VR"""

    def __init__(self, spot: Spot):
        self.spot = spot
        # Power on spot
        self.spot.power_robot()
        # Open the gripper
        self.spot.open_gripper()
        # Move arm to initial configuration
        self.move_to_initial()
        # Get the initial transformation for body to hand
        self._init_robot_trans = spot.get_magnum_Matrix4_spot_a_T_b("body", "hand")
        # Get the initial trans of the VR
        self._init_vr_trans = self.get_init_transformation_vr()
        while self._init_vr_trans is None:
            self._init_vr_trans = self.get_init_transformation_vr()
        print("Got the VR transformation...")
        # Get the initial VR roll pitch yaw
        cur_trans_xyz = cam_pose_from_xzy_to_xyz(
            cam_pose_from_opengl_to_opencv(np.array(self._init_vr_trans))
        )
        self._init_roll, self._init_pitch, self._init_yaw = euler_from_matrix(
            cur_trans_xyz
        )

    def move_to_initial(self):
        """Move the gripper to the initial position"""
        point = INITIAL_POINT
        rpy = INITIAL_RPY
        self.spot.move_gripper_to_point(point, rpy, timeout_sec=2)
        self.cement_arm_joints()

    def cement_arm_joints(self):
        """make the arm to be stable, and follow the xyz and rpy"""
        arm_proprioception = self.spot.get_arm_proprioception()
        current_positions = np.array(
            [v.position.value for v in arm_proprioception.values()]
        )
        self.spot.set_arm_joint_positions(
            positions=current_positions, travel_time=UPDATE_PERIOD
        )

    def get_cur_vr_button(self):
        """Get the button being pressed"""
        hand_held = rospy.get_param("buttonHeld", None)
        if hand_held is not None and len(hand_held) > 0:
            return True
        return False

    def get_cur_vr_pose(self):
        """Get the VR pose"""
        hand_pos = rospy.get_param("hand_0_pos", None)
        hand_rot = rospy.get_param("hand_0_rot", None)
        return hand_pos, hand_rot

    def get_init_transformation_vr(self):
        """Get the init transformation of the VR"""
        hand_pos, hand_rot = self.get_cur_vr_pose()
        if hand_pos is None or hand_rot is None:
            return None

        hand_rot_quat = mn.Quaternion(
            mn.Vector3(hand_rot[0], hand_rot[1], hand_rot[2]), hand_rot[3]
        )

        # Get the transformation
        trans = mn.Matrix4.from_(hand_rot_quat.to_matrix(), mn.Vector3(hand_pos))

        return trans

    def _get_xyz_from_habitat_to_spot(self, habitat_pos):
        return [-habitat_pos[2], -habitat_pos[0], habitat_pos[1]]

    def _get_rpy_from_habitat_to_spot(self, habitat_rot):
        return np.array([-habitat_rot[1], habitat_rot[0], -habitat_rot[2]])

    def _get_target_xyz(self):
        """Get the target xyz for the gripper"""
        # Get the current VR hand location
        cur_pos, _ = self.get_cur_vr_pose()
        cur_pos_relative_to_init = self._init_vr_trans.inverted().transform_point(
            cur_pos
        )
        # Make the xyz to be correct
        cur_pos_relative_to_init = self._get_xyz_from_habitat_to_spot(
            cur_pos_relative_to_init
        )
        # Get the point in robot frame
        target_ee_pos = self._init_robot_trans.transform_point(cur_pos_relative_to_init)
        return target_ee_pos

    def _get_target_rpy(self):
        """Get the target rpy for the gripper"""
        # Get the current transformation
        cur_trans = self.get_init_transformation_vr()
        cur_trans_xyz = cam_pose_from_xzy_to_xyz(
            cam_pose_from_opengl_to_opencv(np.array(cur_trans))
        )
        cur_roll, cur_pitch, cur_yaw = euler_from_matrix(cur_trans_xyz)
        # Compute the delta for roll pitch yaw
        delta_r = angle_between(cur_roll, self._init_roll)
        delta_p = angle_between(cur_pitch, self._init_pitch)
        delta_y = angle_between(cur_yaw, self._init_yaw)
        # Adjust the roll pitch yaw to be human understandable
        target_ee_rot = self._get_rpy_from_habitat_to_spot([delta_r, delta_p, delta_y])
        return target_ee_rot

    def track_vr(self):
        """Track VR using Spot arm."""
        # get xyz
        target_ee_pos = self._get_target_xyz()
        # get rpy
        target_ee_rot = self._get_target_rpy()

        print(f"target pos and rot: {target_ee_pos}, {target_ee_rot}")

        # Move the gripper
        self.spot.move_gripper_to_point(
            target_ee_pos, target_ee_rot, seconds_to_goal=1.0, timeout_sec=0.1
        )
        self.cement_arm_joints()

        # Get the parameter for the button
        if self.get_cur_vr_button():
            self.spot.close_gripper()
        else:
            self.spot.open_gripper()
