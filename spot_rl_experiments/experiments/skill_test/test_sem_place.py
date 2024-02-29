# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import quaternion
from spot_rl.envs.skill_manager import SpotSkillManager

if __name__ == "__main__":
    from spot_rl.utils.utils import map_user_input_to_boolean

    # Know which location we are doing experiments
    in_fre_lab = map_user_input_to_boolean("Are you Tushar in FRE? Y/N ")
    if in_fre_lab:
        # at FRE
        place_target = "place_taget_test_table"
    else:
        # at NYC
        place_target = "test_desk"  # "high_shelf", "test_desk" #"shelf" #"drawer_pretend_to_be_dish_washer"  # "test_semantic_place_table"

    spotskillmanager = SpotSkillManager(use_mobile_pick=True, use_semantic_place=True)
    # spotskillmanager.spot.move_gripper_to_point(np.array([0.55, 0, 0.26]),[0.57,1.4,0])
    # quat = spotskillmanager.spot.get_ee_pos_in_body_frame_quat()
    # quat_value = spotskillmanager.spot.angle_between_quat(quat, quat)
    # spotskillmanager.place_controller.env.get_place_sensor()
    contnue = True
    while contnue:
        if in_fre_lab:
            spotskillmanager.nav(place_target)
        else:
            # spotskillmanager.nav("nyc_mg_pos1")
            # spotskillmanager.pick("glass bottle")
            # spotskillmanager.nav(place_target)
            pass

        spotskillmanager.place(place_target)
        contnue = map_user_input_to_boolean("Do you want to do it again ? Y/N ")

    # Navigate to dock and shutdown
    # spotskillmanager.dock()
