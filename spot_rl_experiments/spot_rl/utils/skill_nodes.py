# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os
import os.path as osp
import time
from collections import deque

import numpy as np
import rospy
import tqdm
from spot_rl.envs.skill_manager import SpotSkillManager
from spot_rl.utils.utils import ros_topics as rt


class SpotRosSkillExecutor:
    def __init__(self, spotskillmanager):
        self.spotskillmanager = spotskillmanager

    def execute_skills(self):
        # Get the current skill name
        skill_name = rospy.get_param("/skill_name", "None")
        skill_input = rospy.get_param("/skill_input", "None")
        print(f"skill_name {skill_name} skill_input {skill_input}")

        # Select the skill to execute
        if skill_name == "Navigate":
            succeded, msg = self.spotskillmanager.nav(skill_input)
        elif skill_name == "Pick":
            succeded, msg = self.spotskillmanager.pick(skill_input)
        elif skill_name == "Place":
            succeded, msg = self.spotskillmanager.place(skill_input)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--useful_parameters", action="store_true")
    _ = parser.parse_args()

    # Clean up the ros parameters
    rospy.set_param("/skill_name", "None")
    rospy.set_param("/skill_input", "None")
    # Call the skill manager
    spotskillmanager = SpotSkillManager(use_mobile_pick=True)

    exe = None
    # try:
    exe = SpotRosSkillExecutor(spotskillmanager)
    while not rospy.is_shutdown():
        exe.execute_skills()
    # except Exception as e:
    #     print("Ending script.")


if __name__ == "__main__":
    main()
