# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import rospy
from perception_and_utils.utils.generic_utils import map_user_input_to_boolean
from spot_rl.envs.skill_manager import SpotSkillManager

if __name__ == "__main__":
    spotskillmanager = SpotSkillManager(
        use_mobile_pick=False,
        use_semantic_place=True,
        use_semantic_place_ee_no_waypoint=True,
    )
    is_local = True

    # Start testing
    contnue = True
    while contnue:
        rospy.set_param("is_gripper_blocked", 0)
        spotskillmanager.place(
            None,
            is_local=is_local,
            visualize=True,
            ee_orientation_at_grasping="side_right",
        )
        contnue = map_user_input_to_boolean("Do you want to do it again ? Y/N ")
