# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import os.path as osp
import time
import traceback

import numpy as np
import rospy
from spot_rl.envs.skill_manager import SpotSkillManager
from spot_rl.utils.heuristic_nav import scan_arm
from spot_rl.utils.retrieve_robot_poses_from_cg import get_view_poses
from spot_rl.utils.utils import get_skill_name_and_input_from_ros
from spot_rl.utils.utils import ros_topics as rt
from spot_rl.utils.waypoint_estimation_based_on_robot_poses_from_cg import (
    get_navigation_points,
)


class SpotRosSkillExecutor:
    """This class reads the ros buffer to execute skills"""

    def __init__(self, spotskillmanager):
        self.spotskillmanager: SpotSkillManager = spotskillmanager
        self._cur_skill_name_input = None
        self.reset_image_viz_params()

    def reset_skill_msg(self):
        """Reset the skill message. The format is skill name, success flag, and message string.
        This is useful for returning the message (e.g., if skill fails or not) from spot-sim2real to high-level planner.
        """
        rospy.set_param("/skill_name_suc_msg", f"{str(time.time())},None,None,None")

    def reset_skill_name_input(self, skill_name, succeded, msg):
        """Reset skill name and input, and publish the message"""
        rospy.set_param("/skill_name_input", f"{str(time.time())},None,None")
        if succeded:
            msg = "Successful execution!"  # This is to make sure habitat-llm use the correct success msg
        rospy.set_param(
            "/skill_name_suc_msg", f"{str(time.time())},{skill_name},{succeded},{msg}"
        )

    def reset_image_viz_params(self):
        """Reset the image viz params"""
        rospy.set_param("/viz_pick", "None")
        rospy.set_param("/viz_object", "None")
        rospy.set_param("/viz_place", "None")

    def check_pick_condition(self):
        """ "Check pick condition in spot-sim2real side"""
        robot_holding = (
            self.spotskillmanager.spot.robot_state_client.get_robot_state().manipulator_state.is_gripper_holding_item
        )
        if robot_holding:
            return (
                False,
                "The arm is currently grasping the object. Make the agent place the grasped object first.",
            )
        else:
            return True, ""

    def execute_skills(self):
        """Execute skills."""

        # Get the current skill name
        skill_name, skill_input = get_skill_name_and_input_from_ros()

        robot_holding = (
            self.spotskillmanager.spot.robot_state_client.get_robot_state().manipulator_state.is_gripper_holding_item
        )
        # Select the skill from the ros buffer and call the skill
        if skill_name == "nav":
            print(f"current skill_name {skill_name} skill_input {skill_input}")
            # Reset the skill message
            self.reset_skill_msg()
            # For navigation target
            nav_target_xyz = rospy.get_param("nav_target_xyz", "None,None,None|")
            # Call the skill
            if "None" not in nav_target_xyz:
                if robot_holding:
                    rospy.set_param("/viz_place", nav_target_xyz)
                else:
                    rospy.set_param("/viz_pick", nav_target_xyz)
                nav_target_xyz = nav_target_xyz.split("|")[0:-1]
                for nav_i, nav_target in enumerate(nav_target_xyz):
                    _nav_target = nav_target.split(",")
                    # This z and y are flipped due to hab convention
                    x, y = (
                        float(_nav_target[0]),
                        float(_nav_target[2]),
                    )
                    print(f"nav to {x} {y}, {nav_i+1}/{len(nav_target_xyz)}")
                    succeded, msg = self.spotskillmanager.nav(x, y)
                    if not succeded:
                        break
            else:
                if robot_holding:
                    rospy.set_param("/viz_place", skill_input)
                else:
                    rospy.set_param("/viz_pick", skill_input)
                succeded, msg = self.spotskillmanager.nav(skill_input)

            # Run scan arm if gripper is NOT holding any item
            print(
                f"Navigation finished, succeded={succeded} , robot_holding={robot_holding}"
            )
            if succeded and not robot_holding:
                rospy.set_param("/is_arm_scanning", f"{str(time.time())},True")
                time.sleep(1)
                print("Scanning area with arm")
                scan_arm(self.spotskillmanager.spot)
            else:
                print("Will not scan arm")
            rospy.set_param("/is_arm_scanning", f"{str(time.time())},False")

            # Reset skill name and input and publish message
            self.reset_skill_name_input(skill_name, succeded, msg)
            # Reset the navigation target
            rospy.set_param("nav_target_xyz", "None,None,None|")
            rospy.set_param("/viz_pick", "None")
            rospy.set_param("/viz_place", "None")
        elif skill_name == "nav_path_planning_with_view_poses":
            print(f"current skill_name {skill_name} skill_input {skill_input}")
            # Get the bbox center and bbox extent
            bbox_info = skill_input.split(";")  # in the format of x,y,z
            assert (
                len(bbox_info) >= 6
            ), f"Wrong size of the bbox info, it should be 6, but got {len(bbox_info)}"
            bbox_center = np.array([float(v) for v in bbox_info[0:3]])
            bbox_extent = np.array([float(v) for v in bbox_info[3:6]])
            query_class_names = bbox_info[6:]
            query_class_names[0] = query_class_names[0].replace("_", " ")
            if robot_holding:
                rospy.set_param("/viz_place", query_class_names[0])
            else:
                rospy.set_param("/viz_pick", query_class_names[0])
            # Get the view poses
            view_poses, category_tag = get_view_poses(
                bbox_center, bbox_extent, query_class_names, False
            )

            # Get the robot x, y, yaw
            x, y, _ = self.spotskillmanager.spot.get_xy_yaw()
            # Get the navigation points
            nav_pts = get_navigation_points(
                view_poses, bbox_center, bbox_extent, [x, y], False, "pathplanning.png"
            )

            # Sequentially give the point
            if len(nav_pts) > 0:
                final_pt_i = len(nav_pts) - 1
                for pt_i, pt in enumerate(nav_pts):
                    x, y, yaw = pt
                    if pt_i == final_pt_i:
                        # Do normal point nav with yaw for the final location
                        if category_tag is not None and "object" in category_tag:
                            # increse nav error threshold for final nav
                            # take a backup of prev error margins
                            navconfig = self.spotskillmanager.nav_controller.env.config
                            backup_success_distance, backup_success_angle = (
                                navconfig.SUCCESS_DISTANCE,
                                navconfig.SUCCESS_ANGLE_DIST,
                            )
                            navconfig.SUCCESS_DISTANCE, navconfig.SUCCESS_ANGLE_DIST = (
                                backup_success_distance * 2.0,
                                backup_success_angle * 2.0,
                            )
                            succeded, msg = self.spotskillmanager.nav(x, y, yaw, False)
                            navconfig.SUCCESS_DISTANCE, navconfig.SUCCESS_ANGLE_DIST = (
                                backup_success_distance,
                                backup_success_angle,
                            )
                        else:
                            succeded, msg = self.spotskillmanager.nav(x, y, yaw, False)
                    else:
                        # Do dynamic point yaw here for the intermediate points
                        succeded, msg = self.spotskillmanager.nav(x, y)
            else:
                succeded = False
                msg = "Cannot navigate to the point"

            # Run scan arm if gripper is NOT holding any item
            print(
                f"Navigation finished, succeded={succeded} , robot_holding={robot_holding}"
            )
            if succeded and not robot_holding:
                rospy.set_param("/is_arm_scanning", f"{str(time.time())},True")
                time.sleep(1)
                print("Scanning area with arm")
                scan_arm(self.spotskillmanager.spot)
            else:
                print("Will not scan arm")
            rospy.set_param("/is_arm_scanning", f"{str(time.time())},False")

            # Reset skill name and input and publish message
            self.reset_skill_name_input(skill_name, succeded, msg)
            rospy.set_param("/viz_pick", "None")
            rospy.set_param("/viz_place", "None")
        elif skill_name == "pick":
            print(f"current skill_name {skill_name} skill_input {skill_input}")
            rospy.set_param("/viz_object", skill_input)
            # Set the multi class object target
            rospy.set_param("multi_class_object_target", skill_input)
            self.reset_skill_msg()
            pick_pass, pick_msg = self.check_pick_condition()
            if pick_pass:
                succeded, msg = self.spotskillmanager.pick(skill_input)
            else:
                succeded = False
                msg = pick_msg
            self.reset_skill_name_input(skill_name, succeded, msg)
            rospy.set_param("/viz_object", "None")
        elif skill_name == "place":
            print(f"current skill_name {skill_name} skill_input {skill_input}")
            self.reset_skill_msg()
            if self.spotskillmanager.allow_semantic_place:
                # Call semantic place skills
                rospy.set_param("is_gripper_blocked", 0)
                succeded, msg = self.spotskillmanager.place(
                    skill_input,
                    is_local=True,
                    visualize=False,
                    enable_waypoint_estimation=True,
                )
            else:
                # Use the following for the hardcode waypoint for static place
                succeded, msg = self.spotskillmanager.place(
                    0.6, 0.0, 0.4, is_local=True
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
        elif skill_name == "findagentaction":
            print(f"current skill_name {skill_name} skill_input {skill_input}")
            self.reset_skill_msg()
            succeded, msg = True, rospy.get_param("human_state", "standing")
            self.reset_skill_name_input(skill_name, succeded, msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--useful_parameters", action="store_true")
    _ = parser.parse_args()

    # Clean up the ros parameters
    rospy.set_param("/skill_name_input", f"{str(time.time())},None,None")
    rospy.set_param("/skill_name_suc_msg", f"{str(time.time())},None,None,None")

    # Call the skill manager
    spotskillmanager = SpotSkillManager(use_mobile_pick=True, use_semantic_place=True)
    executor = None
    try:
        executor = SpotRosSkillExecutor(spotskillmanager)
        # While loop to run in the background
        while not rospy.is_shutdown():
            executor.execute_skills()
    except Exception as e:
        print(f"Ending script: {e}\n Full exception : {traceback.print_exc()}")


if __name__ == "__main__":
    main()
