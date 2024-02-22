# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict

import magnum as mn
import numpy as np
import quaternion
from spot_rl.envs.base_env import SpotBaseEnv
from spot_rl.utils.geometry_utils import is_position_within_bounds
from spot_wrapper.spot import Spot


class SpotPlaceEnv(SpotBaseEnv):
    def __init__(self, config, spot: Spot):
        super().__init__(config, spot)
        self.place_target = None
        self.place_target_is_local = False

        self.ee_gripper_offset = mn.Vector3(config.EE_GRIPPER_OFFSET)
        self.placed = False

    def reset(self, place_target, target_is_local=False, *args, **kwargs):
        assert place_target is not None
        self.place_target = np.array(place_target)
        self.place_target_is_local = target_is_local

        self.reset_arm()
        # Set the initial ee pose
        self.initial_ee_pose = self.spot.get_ee_pos_in_body_frame_quat()
        # Set the target pose
        self.target_object_pose = self.spot.get_ee_pos_in_body_frame_quat()
        # self.target_object_pose = quaternion.quaternion(
        #     0.709041893482208,
        #     0.704837739467621,
        #     -0.00589140923693776,
        #     -0.0207040011882782,
        # )
        observations = super().reset()
        self.placed = False
        return observations

    def step(self, action_dict: Dict[str, Any], *args, **kwargs):
        # gripper_pos_in_base_frame = self.get_gripper_position_in_base_frame_hab()
        # place_target_in_base_frame = self.get_base_frame_place_target_hab()
        # place = is_position_within_bounds(
        #     gripper_pos_in_base_frame,
        #     place_target_in_base_frame,
        #     self.config.SUCC_XY_DIST,
        #     self.config.SUCC_Z_DIST,
        #     convention="habitat",
        # )
        place = np.linalg.norm(self.get_place_sensor(True)) < 0.2
        print("dis to goal:", np.linalg.norm(self.get_place_sensor(True)))
        
        # Update the action_dict with place flag
        action_dict["place"] = place

        print("place in base place env:", place)
        return super().step(action_dict=action_dict, *args, **kwargs)

    def get_success(self, observations):
        return self.place_attempted

    def get_observations(self):
        observations = {
            "joint": self.get_arm_joints(),
            "obj_start_sensor": self.get_place_sensor(),
        }

        return observations


class SpotSemanticPlaceEnv(SpotPlaceEnv):
    def __init__(self, config, spot: Spot):
        super().__init__(config, spot)
        self.initial_ee_pose = None
        self.target_object_pose = None
        # Overwrite joint limits for semantic_place skills
        self.arm_lower_limits = np.deg2rad(config.ARM_LOWER_LIMITS_FOR_SEMANTIC_PLACE)
        self.arm_upper_limits = np.deg2rad(config.ARM_UPPER_LIMITS_FOR_SEMANTIC_PLACE)

    def get_observations(self):
        assert self.initial_ee_pose is not None
        assert self.target_object_pose is not None

        # Get the gaol sensor
        obj_goal_sensor = self.get_place_sensor(True)
        # obj_goal_sensor = self.get_place_sensor_norm()

        # Get the delta ee orientation
        current_gripper_orientation = self.spot.get_ee_pos_in_body_frame_quat()
        delta_ee = self.spot.angle_between_quat(
            self.initial_ee_pose, current_gripper_orientation
        )
        delta_ee = np.array([delta_ee], dtype=np.float32)

        # Get the delta object orientation
        delta_obj = self.spot.angle_between_quat(
            self.target_object_pose, current_gripper_orientation
        )
        delta_obj = np.array([delta_obj], dtype=np.float32)

        # Get the jaw image
        arm_depth, _ = self.get_gripper_images()

        print("rpy to init ee:", delta_ee)
        print("rpy to targ obj:", delta_obj)
        print("xyz to targ obj:", obj_goal_sensor)
        # self.spot.move_gripper_to_point(np.array([1.35, 0.17, 0.35]),[0.0,0,0])

        observations = {
            "obj_goal_sensor": obj_goal_sensor,
            "relative_initial_ee_orientation": delta_ee,
            "relative_target_object_orientation": delta_obj,
            "articulated_agent_jaw_depth": arm_depth,
            "joint": self.get_arm_joints(semantic_place=True),
            "is_holding": np.ones((1,)),
        }

        return observations

    def step(self, grip_action=None, *args, **kwargs):
        # <= 0 for unsnap
        place = grip_action <= 0.0
        print("grip_action in sem place env:", grip_action)
        return super().step(place=place, semantic_place=place, *args, **kwargs)
