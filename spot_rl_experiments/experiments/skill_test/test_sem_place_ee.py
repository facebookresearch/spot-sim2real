# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import rospy
from perception_and_utils.utils.generic_utils import map_user_input_to_boolean
from spot_rl.envs.skill_manager import SpotSkillManager

if __name__ == "__main__":
    # Know which location we are doing experiments
    in_fre_lab = map_user_input_to_boolean("Are you JT in FRE? Y/N ")
    enable_estimation_before_place = map_user_input_to_boolean(
        "Enable estimation before place? Y/N "
    )

    if in_fre_lab:
        # at FRE
        place_target = "coffee_table"
    else:
        # at NYC
        place_target = "test_desk"

    spotskillmanager = SpotSkillManager(
        use_mobile_pick=False, use_semantic_place=True, use_semantic_place_ee=True
    )

    is_local = False
    if enable_estimation_before_place:
        place_target = None
        is_local = True

    # Start testing
    contnue = True
    while contnue:
        rospy.set_param("is_gripper_blocked", 0)
        spotskillmanager.place(place_target, is_local=is_local, visualize=True)
        contnue = map_user_input_to_boolean("Do you want to do it again ? Y/N ")
