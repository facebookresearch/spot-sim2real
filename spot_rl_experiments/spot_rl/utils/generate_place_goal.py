# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import magnum as mn
import numpy as np
from spot_rl.envs.base_env import SpotBaseEnv
from spot_wrapper.spot import Spot

EE_GRIPPER_OFFSET = [0.2, 0.0, 0.05]


def get_global_place_target(spot: Spot):
    base_place_target = get_local_place_target(spot)
    x, y, yaw = spot.get_xy_yaw()
    base_T_global = mn.Matrix4.from_(
        mn.Matrix4.rotation_z(mn.Rad(yaw)).rotation(),
        mn.Vector3(mn.Vector3(x, y, 0.5)),
    )
    global_place_target = base_T_global.transform_point(base_place_target)
    print("global_place_target: ", global_place_target)
    return global_place_target


def get_local_place_target(spot: Spot):
    position, rotation = spot.get_base_transform_to("link_wr1")
    position = [position.x, position.y, position.z]
    rotation = [rotation.x, rotation.y, rotation.z, rotation.w]
    wrist_T_base = SpotBaseEnv.spot2habitat_transform(position, rotation)
    gripper_T_base = wrist_T_base @ mn.Matrix4.translation(
        mn.Vector3(EE_GRIPPER_OFFSET)
    )
    base_place_target_habitat = np.array(gripper_T_base.translation)
    base_place_target = base_place_target_habitat[[0, 2, 1]]

    # Convert base_place_target from numpy.array to magnum.Vector3
    base_place_target = mn.Vector3(
        base_place_target[0], base_place_target[1], base_place_target[2]
    )
    print("base_place_target: ", base_place_target)
    return base_place_target


if __name__ == "__main__":
    spot = Spot("PlaceGoalGenerator")
    global_place_target = get_global_place_target(spot)
    print(global_place_target)
