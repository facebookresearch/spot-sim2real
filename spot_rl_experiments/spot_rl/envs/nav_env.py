# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from bosdyn.client.frame_helpers import get_a_tform_b
from bosdyn.client.math_helpers import quat_to_eulerZYX
from spot_rl.envs.base_env import SpotBaseEnv
from spot_wrapper.spot import Spot, wrap_heading


class SpotNavEnv(SpotBaseEnv):
    def __init__(self, config, spot: Spot):
        super().__init__(config, spot)
        self._goal_xy = None
        self._enable_nav_by_hand = False
        self.goal_heading = None
        self.succ_distance = config.SUCCESS_DISTANCE
        self.succ_angle = np.deg2rad(config.SUCCESS_ANGLE_DIST)

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

    def reset(self, goal_xy, goal_heading):
        self._goal_xy = np.array(goal_xy, dtype=np.float32)
        self.goal_heading = goal_heading
        observations = super().reset()
        assert len(self._goal_xy) == 2

        return observations

    def get_success(self, observations):
        succ = self.get_nav_success(observations, self.succ_distance, self.succ_angle)
        if succ:
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

    def get_observations(self):
        return self.get_nav_observation(self._goal_xy, self.goal_heading)
