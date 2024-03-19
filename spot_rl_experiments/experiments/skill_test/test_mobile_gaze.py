# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from spot_rl.envs.skill_manager import SpotSkillManager

if __name__ == "__main__":
    from perception_and_utils.utils.generic_utils import map_user_input_to_boolean

    # Pick Targets from Aria with +- 10 degree angle change
    pick_targets = [
        "kitchen",
        "kitchen_-10",
        "kitchen_+10",
        "console",
        "console_+15",
        "console_-15",
    ]
    contnue = True
    i: int = 0
    n: int = len(pick_targets)
    while contnue:
        # pick_from = pick_targets[i]
        # i += 1
        spotskillmanager = SpotSkillManager(use_mobile_pick=True)
        # spotskillmanager.nav(pick_from)
        # x = input(f"Press Enter to continue to mobile gaze from {pick_from}")
        spotskillmanager.pick("cup")
        spotskillmanager.spot.open_gripper()
        contnue = map_user_input_to_boolean("Do you want to do it again ? Y/N ")

    # Navigate to dock and shutdown
    spotskillmanager.dock()
