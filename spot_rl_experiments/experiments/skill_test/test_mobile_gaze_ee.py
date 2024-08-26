# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
from spot_rl.envs.skill_manager import SpotSkillManager

if __name__ == "__main__":
    from perception_and_utils.utils.generic_utils import map_user_input_to_boolean

<<<<<<< HEAD
    spotskillmanager = SpotSkillManager(use_mobile_pick=True, use_pick_ee=True)
=======
    spotskillmanager = SpotSkillManager(use_mobile_pick=False)
>>>>>>> 69d9e3f (Minor chagnges for new policy weights)
    contnue = True
    while contnue:
        spotskillmanager.pick("Frosted Flakes Cup")
        spotskillmanager.spot.open_gripper()
        contnue = map_user_input_to_boolean("Do you want to do it again ? Y/N ")

    # Navigate to dock and shutdown
    spotskillmanager.dock()
