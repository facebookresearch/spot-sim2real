# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import time

from spot_rl.skills.atomic_skills import Pick
from spot_rl.utils.construct_configs import construct_config_for_gaze
from spot_wrapper.spot import Spot
from spot_wrapper.utils import say

"""
This script will gaze and attempt to pick at each object from the list of target objects - 3 times.
Robot will move until the point of grasping it, but not take away the object
"""

names = ["ball", "penguin", "rubiks_cube", "lion", "toy_car", "yellow_truck"]


def main(spot, bd=False):
    config = construct_config_for_gaze(opts=[])
    config.DONT_PICK_UP = True
    config.OBJECT_LOCK_ON_NEEDED = 5
    config.TERMINATE_ON_GRASP = True
    config.FORGET_TARGET_OBJECT_STEPS = 1000000
    if bd:
        config.GRASP_EVERY_STEP = True
        config.MAX_JOINT_MOVEMENT = 0.0  # freeze arm
        config.MAX_EPISODE_STEPS = 20
    else:
        config.MAX_EPISODE_STEPS = 150

    spot.power_robot()
    gaze_controller = Pick(config, spot, use_mobile_pick=False)
    for _ in range(3):
        time.sleep(2)
        for name in names:
            say(f"Looking at {name}")
            goal_dict = {"target_object": name, "take_user_input": True}
            _, fdbk_msg = gaze_controller.execute(goal_dict=goal_dict)
            print(fdbk_msg)
            time.sleep(2)
        time.sleep(2)

    spot.shutdown(should_dock=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bd", action="store_true")
    args = parser.parse_args()
    spot = Spot("RealGazeEnv")
    with spot.get_lease(hijack=True):
        main(spot, bd=args.bd)
