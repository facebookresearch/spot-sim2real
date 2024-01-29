# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


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

    def step(self, place=False, *args, **kwargs):
        gripper_pos_in_base_frame = self.get_gripper_position_in_base_frame_hab()
        place_target_in_base_frame = self.get_base_frame_place_target_hab()
        place = is_position_within_bounds(
            gripper_pos_in_base_frame,
            place_target_in_base_frame,
            self.config.SUCC_XY_DIST,
            self.config.SUCC_Z_DIST,
            convention="habitat",
        )

        return super().step(place=place, *args, **kwargs)

    def get_success(self, observations):
        return self.place_attempted

    def get_observations(self):
        observations = {
            "joint": self.get_arm_joints(),
            "obj_start_sensor": self.get_place_sensor(),
        }

        return observations
