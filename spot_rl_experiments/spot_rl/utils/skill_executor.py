# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import os.path as osp
import threading
import time
import traceback

import numpy as np
import rospy
from spot_rl.envs.skill_manager import SpotSkillManager
from spot_rl.utils.heuristic_nav import scan_arm, scan_base
from spot_rl.utils.retrieve_robot_poses_from_cg import get_view_poses
from spot_rl.utils.utils import get_skill_name_and_input_from_ros
from spot_rl.utils.utils import ros_topics as rt
from spot_rl.utils.waypoint_estimation_based_on_robot_poses_from_cg import (
    get_navigation_points,
)
from std_msgs.msg import String

LOG_PATH = "../../spot_rl_experiments/experiments/skill_test/logs/"

ENABLE_ARM_SCAN = True


class SpotRosSkillExecutor:
    """This class reads the ros buffer to execute skills"""

    def __init__(self, spotskillmanager):
        self.spotskillmanager: SpotSkillManager = spotskillmanager
        self._cur_skill_name_input = None
        self._is_robot_on_dock = False
        self.reset_image_viz_params()
        self.episode_log = {"actions": []}
        self.total_steps = 0
        self.total_time = 0

        # Listen to cancel msg
        self.end = False
        thread = threading.Thread(target=self.read_cancel_msg)
        thread.start()

        # Reset
        rospy.set_param("place_target_xyz", f"{None},{None},{None}|")
        rospy.set_param("robot_target_ee_rpy", f"{None},{None},{None}|")
        self.detection_topic = "/dwg_obj_pub"
        # which behaviour do you want, continuous dwg additions + scan arm or only do dwg additions in scan_arm
        self._use_continuos_dwg_or_stop_add = "continous"  # "stopnadd"
        flag = self._use_continuos_dwg_or_stop_add == "continous"
        rospy.set_param("/enable_dwg_object_addition", f"{str(time.time())},{flag}")
        # Creating a publisher for Multiclass owlvit detecetions
        self.detection_publisher = rospy.Publisher(
            self.detection_topic, String, queue_size=1, tcp_nodelay=True
        )

    def reset_skill_msg(self):
        """Reset the skill message. The format is skill name, success flag, and message string.
        This is useful for returning the message (e.g., if skill fails or not) from spot-sim2real to high-level planner.
        """
        rospy.set_param("/skill_name_suc_msg", f"{str(time.time())},None,None,None")

    def reset_skill_name_input(self, skill_name, succeded, msg):
        """Reset skill name and input, and publish the message"""
        rospy.set_param("/skill_name_input", f"{str(time.time())},None,None")
        rospy.set_param("place_target_xyz", f"{None},{None},{None}|")
        rospy.set_param("robot_target_ee_rpy", f"{None},{None},{None}|")

        # Check if we need to return the msg based on the human action
        if "None" not in rospy.get_param(
            "/human_action", f"{str(time.time())},None,None,None"
        ):
            human_action_str = rospy.get_param("/human_action")

            print(f"Received human action string {human_action_str}")

            human_action_str = human_action_str.split(",")
            human_action_str = [s.strip() for s in human_action_str]
            _, action, object_name, receptacle_name = human_action_str
            action = action.lower()

            # Process the object name if there are "_" in the name
            if "_" in object_name:
                object_name = object_name.split("_")
                object_name = " ".join(object_name)

            # Determine the msg to return
            if action == "pick":
                msg = f"Human has picked up the {object_name}, you should not intervene human actions and should move to the next object"
            elif action == "place":
                msg = f"Human has placed the {object_name}, you should not intervene human actions and should move to the next object"
            else:
                msg = f"Human has done something to the {object_name}, you should not intervene human actions and should move to the next object"

            succeded = False

            # Reset the human action
            rospy.set_param("/human_action", f"{str(time.time())},None,None,None")

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

    def read_cancel_msg(self):
        while True:
            cancel = rospy.get_param("/cancel", False)
            if cancel:
                print("Cancel!!! Robot returns to dock")
                spotskillmanager = SpotSkillManager(
                    use_mobile_pick=True, use_semantic_place=True
                )
                spotskillmanager.dock()
                self.end = True
                rospy.set_param("/cancel", False)  # Reset
                raise SystemExit

    def check_pick_condition(self):
        """ "Check pick condition in spot-sim2real side"""
        robot_holding = (
            self.spotskillmanager.spot.robot_state_client.get_robot_state().manipulator_state.is_gripper_holding_item
        )
        rospy.set_param("robot_holding", robot_holding)
        if robot_holding:
            return (
                False,
                "The arm is currently grasping the object. Make the agent place the grasped object first.",
            )
        else:
            return True, ""

    # type: ignore
    def execute_skills(self):  # noqa: C901
        """Execute skills."""

        # Get the current skill name
        skill_log = {"success": False, "num_steps": 0}
        skill_name, skill_input = get_skill_name_and_input_from_ros()
        final_success = True
        metric_list = []
        # Power on the robot if the robot was in the dock
        if self._is_robot_on_dock:
            self.spotskillmanager.spot.power_robot()
            self._is_robot_on_dock = False  # reset the flag

        robot_holding = (
            self.spotskillmanager.spot.robot_state_client.get_robot_state().manipulator_state.is_gripper_holding_item
        )
        rospy.set_param("robot_holding", robot_holding)
        # Select the skill from the ros buffer and call the skill
        if skill_name == "nav":
            rospy.set_param("skill_in_execution_lock", True)
            print(f"current skill_name {skill_name} skill_input {skill_input}")
            if not robot_holding:
                self.spotskillmanager.spot.open_gripper()
                flag = self._use_continuos_dwg_or_stop_add == "continous"
                rospy.set_param(
                    "/enable_dwg_object_addition", f"{str(time.time())},{flag}"
                )
            else:
                rospy.set_param(
                    "/enable_dwg_object_addition", f"{str(time.time())},False"
                )

            # Reset the skill message
            self.reset_skill_msg()
            # For navigation target
            nav_target_xyz = rospy.get_param("nav_target_xyz", "None,None,None|")
            is_exploring = rospy.get_param("nav_velocity_scaling", 1.0) != 1.0
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

                    # Nav -> scan behavior
                    if is_exploring and ENABLE_ARM_SCAN:
                        scan_arm(
                            self.spotskillmanager.spot,
                            publisher=self.detection_publisher,
                            enable_object_detector_during_movement=False,
                        )

                    if not succeded and not is_exploring:
                        break
                succeded = True if is_exploring else succeded
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
                time.sleep(1)
                print("Scanning area with arm")
                # we need to keep it on for both cases continous & stopnadd
                rospy.set_param(
                    "/enable_dwg_object_addition", f"{str(time.time())},True"
                )
                # if ENABLE_ARM_SCAN:
                # scan_arm(
                #     self.spotskillmanager.spot,
                #     publisher=self.detection_publisher,
                #     enable_object_detector_during_movement=False,
                # )
                flag = self._use_continuos_dwg_or_stop_add == "continous"
                rospy.set_param(
                    "/enable_dwg_object_addition", f"{str(time.time())},{flag}"
                )
            else:
                print("Will not scan arm")

            # Reset skill name and input and publish message
            skill_log = self.spotskillmanager.nav_controller.skill_result_log
            if "num_steps" not in skill_log:
                skill_log["num_steps"] = 0
            self.episode_log["actions"].append({"nav": skill_log})
            self.reset_skill_name_input(skill_name, succeded, msg)
            # Reset the navigation target
            rospy.set_param("nav_target_xyz", "None,None,None|")
            rospy.set_param("/viz_pick", "None")
            rospy.set_param("/viz_place", "None")
            rospy.set_param("skill_in_execution_lock", False)
        elif skill_name == "nav_path_planning_with_view_poses":
            rospy.set_param("skill_in_execution_lock", True)
            print(f"current skill_name {skill_name} skill_input {skill_input}")
            if not robot_holding:
                self.spotskillmanager.spot.open_gripper()
                flag = self._use_continuos_dwg_or_stop_add == "continous"
                rospy.set_param(
                    "/enable_dwg_object_addition", f"{str(time.time())},{flag}"
                )
            else:
                rospy.set_param(
                    "/enable_dwg_object_addition", f"{str(time.time())},False"
                )

            skill_input_per_nav = skill_input.split("|")
            for skill_input in skill_input_per_nav[:-1]:
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
                    bbox_center, bbox_extent, query_class_names, True
                )

                # Get the robot x, y, yaw
                x, y, _ = self.spotskillmanager.spot.get_xy_yaw()
                # Get the navigation points
                nav_pts = get_navigation_points(
                    view_poses,
                    bbox_center,
                    bbox_extent,
                    [x, y],
                    True,
                    "pathplanning.png",
                )

                # Do not follow path planning if the distance to target is small
                if np.linalg.norm(np.array([x, y]) - np.array(nav_pts[-1][0:2])) < 3:
                    nav_pts = [nav_pts[-1]]

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
                                navconfig = (
                                    self.spotskillmanager.nav_controller.env.config
                                )
                                backup_success_distance, backup_success_angle = (
                                    navconfig.SUCCESS_DISTANCE,
                                    navconfig.SUCCESS_ANGLE_DIST,
                                )
                                (
                                    navconfig.SUCCESS_DISTANCE,
                                    navconfig.SUCCESS_ANGLE_DIST,
                                ) = (
                                    backup_success_distance * 2.0,
                                    backup_success_angle * 2.0,
                                )
                                succeded, msg = self.spotskillmanager.nav(
                                    x, y, yaw, False
                                )
                                (
                                    navconfig.SUCCESS_DISTANCE,
                                    navconfig.SUCCESS_ANGLE_DIST,
                                ) = (
                                    backup_success_distance,
                                    backup_success_angle,
                                )
                            else:
                                succeded, msg = self.spotskillmanager.nav(
                                    x, y, yaw, False
                                )
                        else:
                            # Do dynamic point yaw here for the intermediate points
                            succeded, msg = self.spotskillmanager.nav(x, y)
                        skill_log = (
                            self.spotskillmanager.nav_controller.skill_result_log
                        )
                        if "num_steps" not in skill_log:
                            skill_log["num_steps"] = 0
                        self.episode_log["actions"].append({"nav_viewpose": skill_log})

                else:
                    succeded = False
                    msg = "Cannot navigate to the point"

                # Run scan arm if gripper is NOT holding any item
                print(
                    f"Navigation finished, succeded={succeded} , robot_holding={robot_holding}"
                )
                is_exploring = rospy.get_param("nav_velocity_scaling", 1.0) != 1.0
                if not robot_holding and is_exploring:
                    time.sleep(1)
                    print("Scanning area with arm")
                    rospy.set_param(
                        "/enable_dwg_object_addition", f"{str(time.time())},True"
                    )
                    if ENABLE_ARM_SCAN:
                        scan_arm(
                            self.spotskillmanager.spot,
                            publisher=self.detection_publisher,
                            enable_object_detector_during_movement=False,
                        )
                    flag = self._use_continuos_dwg_or_stop_add == "continous"
                    rospy.set_param(
                        "/enable_dwg_object_addition", f"{str(time.time())},{flag}"
                    )
                else:
                    print("Will not scan arm")

            # Reset skill name and input and publish message
            if is_exploring:
                succeded, msg = True, "Successful execution!"
                rospy.set_param(
                    "/enable_dwg_object_addition", f"{str(time.time())},False"
                )
            self.reset_skill_name_input(skill_name, succeded, msg)
            rospy.set_param("/viz_pick", "None")
            rospy.set_param("/viz_place", "None")
            rospy.set_param("skill_in_execution_lock", False)

        elif skill_name == "pick":
            rospy.set_param("skill_in_execution_lock", True)
            print(f"current skill_name {skill_name} skill_input {skill_input}")
            rospy.set_param("/enable_dwg_object_addition", f"{str(time.time())},False")
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
            self.check_pick_condition()
            skill_log = self.spotskillmanager.gaze_controller.skill_result_log
            if "num_steps" not in skill_log:
                skill_log["num_steps"] = 0
            self.episode_log["actions"].append({"pick": skill_log})
            self.reset_skill_name_input(skill_name, succeded, msg)
            # (
            #     rospy.set_param(
            #         "/enable_dwg_object_addition", f"{str(time.time())},True"
            #     )
            #     if not succeded and self._use_continuos_dwg_or_stop_add == "continous"
            #     else None
            # )
            rospy.set_param("/viz_object", "None")
            rospy.set_param("skill_in_execution_lock", False)
        elif skill_name == "place":
            rospy.set_param("skill_in_execution_lock", True)
            print(f"current skill_name {skill_name} skill_input {skill_input}")
            rospy.set_param("is_gripper_blocked", 0)
            rospy.set_param("/enable_dwg_object_addition", f"{str(time.time())},False")
            self.reset_skill_msg()
            if self.spotskillmanager.allow_semantic_place:
                # Call semantic place skills
                succeded, msg = self.spotskillmanager.place(
                    skill_input,
                    is_local=True,
                    visualize=False,
                    enable_waypoint_estimation=True,
                )
                if succeded:
                    rospy.set_param("is_gripper_blocked", 0)
            else:
                # Use the following for the hardcode waypoint for static place
                succeded, msg = self.spotskillmanager.place(
                    0.6, 0.0, 0.4, is_local=True
                )
            self.check_pick_condition()
            skill_log = self.spotskillmanager.place_controller.skill_result_log
            if "num_steps" not in skill_log:
                skill_log["num_steps"] = 0
            self.episode_log["actions"].append({"place": skill_log})

            self.reset_skill_name_input(skill_name, succeded, msg)
            # (
            #     rospy.set_param(
            #         "/enable_dwg_object_addition", f"{str(time.time())},True"
            #     )
            #     if self._use_continuos_dwg_or_stop_add == "continous"
            #     else None
            # )
            rospy.set_param("skill_in_execution_lock", False)
        elif skill_name == "opendrawer":
            rospy.set_param("skill_in_execution_lock", True)
            print(f"current skill_name {skill_name} skill_input {skill_input}")
            self.reset_skill_msg()
            succeded, msg = self.spotskillmanager.opendrawer()
            self.reset_skill_name_input(skill_name, succeded, msg)
            rospy.set_param("skill_in_execution_lock", False)
        elif skill_name == "closedrawer":
            rospy.set_param("skill_in_execution_lock", True)
            print(f"current skill_name {skill_name} skill_input {skill_input}")
            self.reset_skill_msg()
            succeded, msg = self.spotskillmanager.closedrawer()
            self.reset_skill_name_input(skill_name, succeded, msg)
            rospy.set_param("skill_in_execution_lock", False)
        elif skill_name == "findagentaction":
            rospy.set_param("skill_in_execution_lock", True)
            print(f"current skill_name {skill_name} skill_input {skill_input}")
            self.reset_skill_msg()
            succeded, msg = True, rospy.get_param("human_state", "standing")
            self.reset_skill_name_input(skill_name, succeded, msg)
            rospy.set_param("skill_in_execution_lock", False)
        elif skill_name == "dock":
            print(f"current skill_name {skill_name} skill_input {skill_input}")
            self.reset_skill_msg()
            self.spotskillmanager.dock()
            self._is_robot_on_dock = True
            rospy.set_param("/skill_name_input", f"{str(time.time())},None,None")

        self.total_steps += len(metric_list)
        self.total_time += sum(
            metric["time_taken"] for metric in metric_list if "time_taken" in metric
        )
        self.episode_log["total_time"] = self.total_time
        self.episode_log["final_success"] = final_success and skill_log["success"]
        self.episode_log["total_steps"] = self.total_steps

    def save_logs_as_json(self):
        file_path = osp.join(LOG_PATH, "test.json")
        with open(file_path, "w") as file:
            json.dump(self.episode_log, file, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--useful_parameters", action="store_true")
    _ = parser.parse_args()

    # Clean up the ros parameters
    rospy.set_param("/skill_name_input", f"{str(time.time())},None,None")
    rospy.set_param("/skill_name_suc_msg", f"{str(time.time())},None,None,None")
    rospy.set_param("/cancel", False)
    rospy.set_param("/enable_dwg_object_addition", f"{str(time.time())},True")

    while True:
        # Call the skill manager
        spotskillmanager = SpotSkillManager(
            use_mobile_pick=True, use_semantic_place=True, use_place_ee=True
        )
        executor = None
        try:
            rospy.set_param("skill_in_execution_lock", False)
            executor = SpotRosSkillExecutor(spotskillmanager)
            # While loop to run in the background
            while not rospy.is_shutdown() and not executor.end:
                executor.execute_skills()
        except Exception as e:
            print(f"Ending script: {e}\n Full exception : {traceback.print_exc()}")


if __name__ == "__main__":
    main()
