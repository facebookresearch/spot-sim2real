# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from spot_rl.envs.skill_manager import SpotSkillManager

if __name__ == "__main__":
    from perception_and_utils.utils.generic_utils import map_user_input_to_boolean

    spotskillmanager = SpotSkillManager(use_mobile_pick=False, use_semantic_place=False)
    contnue = True
    object_name = "cup"
    while contnue:
        spotskillmanager.nav("dining_table_demo")
        spotskillmanager.pick(
            object_name,
            enable_pose_correction=False,
            enable_pose_estimation=True,
            enable_force_control=True,
        )
        spotskillmanager.nav("kitchen_counter")
        spotskillmanager.place(
            "", is_local=True, visualize=False, enable_waypoint_estimation=True
        )
        contnue = map_user_input_to_boolean("Do you want to do it again ? Y/N ")

    # Navigate to dock and shutdown
    spotskillmanager.get_env().reset_arm()
    spotskillmanager.sit()
