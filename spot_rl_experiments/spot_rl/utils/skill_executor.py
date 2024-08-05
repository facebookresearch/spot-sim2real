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

    def execute_skills(self):
        """Execute skills."""

        # Get the current skill name
        skill_name, skill_input = get_skill_name_and_input_from_ros()

        # Select the skill from the ros buffer and call the skill
        if skill_name == "nav":
            print(f"current skill_name {skill_name} skill_input {skill_input}")
            # Reset the skill message
            self.reset_skill_msg()
            # For navigation target
            nav_target_xyz = rospy.get_param("nav_target_xyz", "None,None,None|")
            # Call the skill
            if "None" not in nav_target_xyz:
                # TODO: spot-sim2real: A temp hack for theta value
                # Need a way to find the theta value
                nav_target_xyz = nav_target_xyz.split("|")[0:-1]
                for nav_target in nav_target_xyz:
                    _nav_target = nav_target.split(",")
                    # This z and y are flipped due to hab convention
                    x, y, theta = (
                        float(_nav_target[0]),
                        float(_nav_target[2]),
                        float(0.0),
                    )
                    succeded, msg = self.spotskillmanager.nav(x, y, theta)
                    if not succeded:
                        break
            else:
                succeded, msg = self.spotskillmanager.nav(skill_input)
            # Reset skill name and input and publish message
            self.reset_skill_name_input(skill_name, succeded, msg)
            # Reset the navigation target
            rospy.set_param("nav_target_xyz", "None,None,None|")
        elif skill_name == "pick":
            print(f"current skill_name {skill_name} skill_input {skill_input}")
            self.reset_skill_msg()
            succeded, msg = self.spotskillmanager.pick(skill_input)
            self.reset_skill_name_input(skill_name, succeded, msg)
        elif skill_name == "place":
            print(f"current skill_name {skill_name} skill_input {skill_input}")
            self.reset_skill_msg()
            # Use the following for the hardcode waypoint place
            # succeded, msg = self.spotskillmanager.place(0.6, 0.0, 0.4, True)
            # Call semantic place skills
            succeded, msg = self.spotskillmanager.place(
                None, is_local=True, visualize=False
            )
            self.reset_skill_name_input(skill_name, succeded, msg)
        elif skill_name == "opendrawer":
            print(f"current skill_name {skill_name} skill_input {skill_input}")
            self.reset_skill_msg()
            succeded, msg = self.spotskillmanager.opendrawer()
            self.reset_skill_name_input(skill_name, succeded, msg)
        elif skill_name == "closedrawer":
            print(f"current skill_name {skill_name} skill_input {skill_input}")
            self.reset_skill_msg()
            succeded, msg = self.spotskillmanager.closedrawer()
            self.reset_skill_name_input(skill_name, succeded, msg)
        # elif skill_name == "findreceptacle":
        #     self.reset_skill_msg()
        #     succeded, msg = True, rospy.get_param("findreceptacle", "cabinet")
        #     self.reset_skill_name_input(skill_name, succeded, msg)
        # elif skill_name == "findobject":
        #     self.reset_skill_msg()
        #     succeded, msg = True, rospy.get_param("findobject", "cup")
        #     self.reset_skill_name_input(skill_name, succeded, msg)
        elif skill_name == "findagentaction":
            print(f"current skill_name {skill_name} skill_input {skill_input}")
            self.reset_skill_msg()
            succeded, msg = True, rospy.get_param("human_state", "standing")
            self.reset_skill_name_input(skill_name, succeded, msg)
        # elif skill_name == "findroom":
        #     self.reset_skill_msg()
        #     succeded, msg = True, rospy.get_param("findroom", "nyc_lab")
        #     self.reset_skill_name_input(skill_name, succeded, msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--useful_parameters", action="store_true")
    _ = parser.parse_args()

    # Clean up the ros parameters
    rospy.set_param("/skill_name_input", "None,None")
    rospy.set_param("/skill_name_suc_msg", "None,None,None")

    # Call the skill manager
    spotskillmanager = SpotSkillManager(use_mobile_pick=True, use_semantic_place=True)
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
