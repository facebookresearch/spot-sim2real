# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import copy

import cv2
import numpy as np
import quaternion
import rospy
from bosdyn.client.frame_helpers import get_a_tform_b
from scipy.spatial.transform import Rotation
from spot_rl.envs.skill_manager import SpotSkillManager
from spot_rl.utils.construct_configs import construct_config

if __name__ == "__main__":
    from perception_and_utils.utils.generic_utils import map_user_input_to_boolean

    # Know which location we are doing experiments
    in_fre_lab = True  # map_user_input_to_boolean("Are you Tushar in FRE? Y/N ")
    if in_fre_lab:
        # at FRE
        place_target = "place_taget_test_table"
        place_target_before = "place_taget_test_table_before"
    else:
        # at NYC
        place_target = "high_shelf"  # "high_shelf", "test_desk" #"shelf" #"drawer_pretend_to_be_dish_washer"  # "test_semantic_place_table"

    spotskillmanager = SpotSkillManager(use_mobile_pick=False, use_semantic_place=True)
    # spotskillmanager.gaze_controller.reset_skill("bottle")

    spot = spotskillmanager.spot

    # spotskillmanager.pick("bottle")
    contnue = True  # map_user_input_to_boolean("Should we continue ?")

    config = construct_config()

    while contnue:
        if in_fre_lab:
            # spotskillmanager.nav(place_target)

            if config.get("ENABLE_POSE_CORRECTION", False):
                rospy.set_param("pose_correction_success", False)
            spotskillmanager.pick("bottle")
            # TODO: Move the pose correction inside reset_arm
            if rospy.get_param("pose_correction_success", False):
                spot.move_gripper_to_point(
                    (0.55, 0.0, 0.5), np.deg2rad(rospy.get_param("pose_correction"))
                )
        else:
            # spotskillmanager.nav("nyc_mg_pos1")
            # spotskillmanager.pick("glass bottle")
            # spotskillmanager.nav(place_target)
            pass

        spotskillmanager.place_controller.config.RUNNING_AFTER_GRASP_FOR_PLACE = True
        spotskillmanager.place(place_target)
        # spot.open_gripper()
        contnue = map_user_input_to_boolean("Do you want to do it again ? Y/N ")

    # Navigate to dock and shutdown
    # spotskillmanager.dock()
