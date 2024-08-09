# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from spot_rl.envs.skill_manager import SpotSkillManager

if __name__ == "__main__":
    from perception_and_utils.utils.generic_utils import map_user_input_to_boolean

    spotskillmanager = SpotSkillManager(use_mobile_pick=False, use_semantic_place=True)
    contnue = True
    object_name = "pen_box"
    mesh_name = "black toy car"
    while contnue:
        spotskillmanager.spot.stand()
        spotskillmanager.current_receptacle_name = (
            "white_table"  # To test the grasping config
        )
        spotskillmanager.pick(
            object_name,
            mesh_name=mesh_name,
            enable_pose_correction=False,
            enable_pose_estimation=False,
            enable_force_control=False,
        )
        spotskillmanager.spot.open_gripper()
        spotskillmanager.get_env().reset_arm()
        contnue = map_user_input_to_boolean("Do you want to do it again ? Y/N ")

    # Navigate to dock and shutdown
    spotskillmanager.get_env().reset_arm()
    spotskillmanager.sit()
