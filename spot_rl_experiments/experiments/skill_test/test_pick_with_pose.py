# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from spot_rl.envs.skill_manager import SpotSkillManager

if __name__ == "__main__":
    from perception_and_utils.utils.generic_utils import map_user_input_to_boolean

    spotskillmanager = SpotSkillManager(use_mobile_pick=False, use_semantic_place=True)
    contnue = True
    object_name = "cup"  # "penguin plush toy"
    while contnue:
        spotskillmanager.spot.stand()
        spotskillmanager.pick(
            object_name, enable_pose_correction=True, enable_pose_estimation=True
        )
        spotskillmanager.get_env().reset_arm()
        contnue = map_user_input_to_boolean("Do you want to do it again ? Y/N ")

    # Navigate to dock and shutdown
    spotskillmanager.get_env().reset_arm()
    spotskillmanager.sit()
