# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict

import magnum as mn
import numpy as np
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

        observations = super().reset()
        self.placed = False
        return observations

    def step(self, action_dict: Dict[str, Any], *args, **kwargs):
        gripper_pos_in_base_frame = self.get_gripper_position_in_base_frame_hab()
        place_target_in_base_frame = self.get_base_frame_place_target_hab()
        place = is_position_within_bounds(
            gripper_pos_in_base_frame,
            place_target_in_base_frame,
            self.config.SUCC_XY_DIST,
            self.config.SUCC_Z_DIST,
            convention="habitat",
        )

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
        # Overwrite joint limits for semantic_place skills
        self.arm_lower_limits = np.deg2rad(config.ARM_LOWER_LIMITS_FOR_SEMANTIC_PLACE)
        self.arm_upper_limits = np.deg2rad(config.ARM_UPPER_LIMITS_FOR_SEMANTIC_PLACE)

    def get_observations(self):
        assert self.initial_ee_pose is not None
        obj_goal_sensor = self.get_place_sensor()
        current_gripper_orientation = self.spot.get_ee_pos_in_body_frame()[-1]
        delta = self.initial_ee_pose - current_gripper_orientation
        delta = (delta + np.pi) % (2 * np.pi) - np.pi
        arm_depth, _ = self.get_gripper_images()
        print("init rpy delta:", delta)
        print("dis to xyz:", obj_goal_sensor)
        observations = {
            "obj_goal_sensor": obj_goal_sensor,
            "relative_initial_ee_orientation": delta,
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
