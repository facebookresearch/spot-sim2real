# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from spot_rl.envs.skill_manager import SpotSkillManager

if __name__ == "__main__":
    from spot_rl.utils.utils import map_user_input_to_boolean

    place_target = "place_taget_test_table"
    spotskillmanager = SpotSkillManager(use_mobile_pick=True, use_semantic_place=True)
    contnue = True
    while contnue:
        spotskillmanager.nav(place_target)
        # spotskillmanager.pick("bottle")
        spotskillmanager.place(place_target)
        contnue = map_user_input_to_boolean("Do you want to do it again ? Y/N ")

    # Navigate to dock and shutdown
    # spotskillmanager.dock()
