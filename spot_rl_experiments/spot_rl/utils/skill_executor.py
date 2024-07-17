# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import os.path as osp
import time

import rospy
from spot_rl.envs.skill_manager import SpotSkillManager
from spot_rl.utils.utils import get_skill_name_and_input_from_ros
from spot_rl.utils.utils import ros_topics as rt


class SpotRosSkillExecutor:
    """This class reads the ros butter to execute skills"""

    def __init__(self, spotskillmanager):
        self.spotskillmanager = spotskillmanager
        self._cur_skill_name_input = None

    def reset_skill_msg(self):
        """Reset the skill message. The format is skill name, success flag, and message string.
        This is useful for returning the message (e.g., if skill fails or not) from spot-sim2real to high-level planner.
        """
        rospy.set_param("/skill_name_suc_msg", "None,None,None")

    def reset_skill_name_input(self, skill_name, succeded, msg):
        """Reset skill name and input, and publish the message"""
        rospy.set_param("/skill_name_input", "None,None")
        rospy.set_param("/skill_name_suc_msg", f"{skill_name},{succeded},{msg}")
        print("skill done")

    def execute_skills(self):
        """Execute skills."""

        # Get the current skill name
        skill_name, skill_input = get_skill_name_and_input_from_ros()

        if skill_name != "None":
            print(f"current skill_name {skill_name} skill_input {skill_input}")

        # Select the skill from the ros buffer and call the skill
        if skill_name == "nav":
            # Reset the skill message
            self.reset_skill_msg()
            # Call the skill
            succeded, msg = self.spotskillmanager.nav(skill_input)
            # Reset skill name and input and publish message
            self.reset_skill_name_input(skill_name, succeded, msg)
        elif skill_name == "pick":
            self.reset_skill_msg()
            succeded, msg = self.spotskillmanager.pick(skill_input)
            self.reset_skill_name_input(skill_name, succeded, msg)
        elif skill_name == "place":
            self.reset_skill_msg()
            succeded, msg = self.spotskillmanager.place(skill_input)
            self.reset_skill_name_input(skill_name, succeded, msg)
        elif skill_name == "opendrawer":
            self.reset_skill_msg()
            succeded, msg = self.spotskillmanager.opendrawer()
            self.reset_skill_name_input(skill_name, succeded, msg)
        elif skill_name == "closedrawer":
            self.reset_skill_msg()
            succeded, msg = self.spotskillmanager.closedrawer()
            self.reset_skill_name_input(skill_name, succeded, msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--useful_parameters", action="store_true")
    _ = parser.parse_args()

    # Clean up the ros parameters
    rospy.set_param("/skill_name_input", "None,None")
    rospy.set_param("/skill_name_suc_msg", "None,None,None")

    # Call the skill manager
    spotskillmanager = SpotSkillManager(use_mobile_pick=True, use_semantic_place=False)

    executor = None
    try:
        executor = SpotRosSkillExecutor(spotskillmanager)
        # While loop to run in the background
        while not rospy.is_shutdown():
            executor.execute_skills()
    except Exception as e:
        print(f"Ending script: {e}")


if __name__ == "__main__":
    main()
