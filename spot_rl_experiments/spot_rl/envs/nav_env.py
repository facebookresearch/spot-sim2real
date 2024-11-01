# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os

import magnum as mn
import numpy as np
import rospy
from bosdyn.client.frame_helpers import get_a_tform_b
from bosdyn.client.math_helpers import quat_to_eulerZYX
from spot_rl.envs.base_env import SpotBaseEnv
from spot_wrapper.spot import Spot, wrap_heading

DOCK_ID = int(os.environ.get("SPOT_DOCK_ID", 549))


class SpotNavEnv(SpotBaseEnv):
    def __init__(self, config, spot: Spot):
        super().__init__(config, spot)
        self._goal_xy = None
        self._enable_nav_by_hand = False
        self._enable_dynamic_yaw = False
        self.goal_heading = None
        self.succ_distance = config.SUCCESS_DISTANCE
        self.succ_angle = np.deg2rad(config.SUCCESS_ANGLE_DIST)

        self.initial_arm_joint_angles = np.deg2rad(config.GAZE_ARM_JOINT_ANGLES_EXPLORE)

    def enable_nav_by_hand(self):
        if not self._enable_nav_by_hand:
            self._enable_nav_by_hand = True
            print(
                f"{self.node_name} Enabling nav goal change get_nav_observation by base switched to get_nav_observation by hand fn"
            )
            self.backup_fn_of_get_nav_observation_that_operates_by_robot_base = (
                self.get_nav_observation
            )
            self.get_nav_observation = self.get_nav_observation_by_hand

    def disable_nav_by_hand(self):
        if self._enable_nav_by_hand:
            self.get_nav_observation = (
                self.backup_fn_of_get_nav_observation_that_operates_by_robot_base
            )
            self._enable_nav_by_hand = False
            print(
                f"{self.node_name} Disabling nav goal change get_nav_observation by base fn restored"
            )

    def reset(self, goal_xy, goal_heading, dynamic_yaw=False):

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

        self._enable_dynamic_yaw = dynamic_yaw
        self._goal_xy = np.array(goal_xy, dtype=np.float32)
        self.goal_heading = goal_heading
        observations = super().reset()

        assert len(self._goal_xy) == 2

        if self._enable_dynamic_yaw:
            self.succ_distance = self.config.SUCCESS_DISTANCE_FOR_DYNAMIC_YAW_NAV
            self.succ_angle = np.deg2rad(
                self.config.SUCCESS_ANGLE_DIST_FOR_DYNAMIC_YAW_NAV
            )
        else:
            self.succ_distance = self.config.SUCCESS_DISTANCE
            self.succ_angle = np.deg2rad(self.config.SUCCESS_ANGLE_DIST)

        # Make sure the we use gripper image for the detection of the object
        rospy.set_param("is_gripper_blocked", 0)

        return observations

    def get_success(self, observations, succ_set_base=True):
        succ = self.get_nav_success(observations, self.succ_distance, self.succ_angle)
        if succ and succ_set_base:
            self.spot.set_base_velocity(0.0, 0.0, 0.0, 1 / self.ctrl_hz)
        return succ

    def get_hand_xy_theta(self, use_boot_origin=False):
        """
        Much like spot.get_xy_yaw(), this function returns x,y,yaw of the hand camera instead of base such as in spot.get_xy_yaw()
        Accepts the same parameter use_boot_origin of type bool like the function mentioned in above line, this determines whether the calculation is from the vision frame or robot'home
        If true, then the location is calculated from the vision frame else from home/dock
        Returns x,y,theta useful in head/hand based navigation used in Heurisitic Mobile Navigation
        """
        vision_T_hand = get_a_tform_b(
            self.spot.robot_state_client.get_robot_state().kinematic_state.transforms_snapshot,
            "vision",
            "hand",
        )
        theta = quat_to_eulerZYX(vision_T_hand.rotation)[0]
        point_in_global_2d = np.array([vision_T_hand.x, vision_T_hand.y])
        return (
            (point_in_global_2d[0], point_in_global_2d[1], theta)
            if use_boot_origin
            else self.spot.xy_yaw_global_to_home(
                point_in_global_2d[0], point_in_global_2d[1], theta
            )
        )

    def get_nav_observation_by_hand(self, goal_xy, goal_heading):

        observations = self.get_head_depth()

        # Get rho theta observation
        x, y, yaw = self.get_hand_xy_theta()
        curr_xy = np.array([x, y], dtype=np.float32)
        rho = np.linalg.norm(curr_xy - goal_xy)
        theta = wrap_heading(np.arctan2(goal_xy[1] - y, goal_xy[0] - x) - yaw)
        rho_theta = np.array([rho, theta], dtype=np.float32)

        # Get goal heading observation
        goal_heading_ = -np.array([wrap_heading(goal_heading - yaw)], dtype=np.float32)
        observations["target_point_goal_gps_and_compass_sensor"] = rho_theta
        observations["goal_heading"] = goal_heading_

        return observations

    def get_current_angle_for_target_facing(self):
        vector_robot_to_target = self._goal_xy - np.array([self.x, self.y])
        vector_robot_to_target = vector_robot_to_target / np.linalg.norm(
            vector_robot_to_target
        )
        vector_forward_robot = np.array(
            self.curr_transform.transform_vector(mn.Vector3(1, 0, 0))
        )[[0, 1]]
        vector_forward_robot = vector_forward_robot / np.linalg.norm(
            vector_forward_robot
        )

        return vector_robot_to_target, vector_forward_robot

    def get_observations(self):
        if self._enable_dynamic_yaw:
            # Modify the goal_heading here based on the current robot orientation
            (
                vector_robot_to_target,
                vector_forward_robot,
            ) = self.get_current_angle_for_target_facing()
            x1 = (
                vector_robot_to_target[1] * vector_forward_robot[0]
                - vector_robot_to_target[0] * vector_forward_robot[1]
            )
            x2 = (
                vector_robot_to_target[0] * vector_forward_robot[0]
                + vector_robot_to_target[1] * vector_forward_robot[1]
            )
            rotation_delta = np.arctan2(x1, x2)
            self.goal_heading = wrap_heading(self.yaw + rotation_delta)

        return self.get_nav_observation(self._goal_xy, self.goal_heading)

    def step(self, *args, **kwargs):
        observations, reward, done, info = super().step(*args, **kwargs)

        # Check if we need to dock the robot
        if kwargs["action_dict"].get("should_dock", False):
            try:
                self.spot.dock(dock_id=DOCK_ID, home_robot=True)
            except Exception as e:
                print(f"error while docking {str(e)}")

        # Slow the base down if we are close to the nav target to slow down the the heading changes
        dist_to_goal, _ = observations["target_point_goal_gps_and_compass_sensor"]
        abs_good_heading = abs(observations["goal_heading"][0])

        if self._enable_dynamic_yaw:
            if dist_to_goal < 1.5 and abs_good_heading < np.rad2deg(45):
                self.slowdown_base = 0.5
            else:
                self.slowdown_base = -1

        # Slow down the base for exploration
        is_exploring = rospy.get_param("nav_velocity_scaling", 1.0) != 1.0
        if dist_to_goal < 1.0 and abs_good_heading < np.rad2deg(45) and is_exploring:
            self.slowdown_base = 0.5
        else:
            self.slowdown_base = -1

        return observations, reward, done, info
