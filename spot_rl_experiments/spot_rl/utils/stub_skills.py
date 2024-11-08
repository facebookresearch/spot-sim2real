import time

import numpy as np
import rospy
from spot_rl.utils.retrieve_robot_poses_from_cg import get_view_poses
from spot_rl.utils.waypoint_estimation_based_on_robot_poses_from_cg import (
    get_navigation_points,
)


def get_skill_name_and_input_from_ros():
    """
    Get the ros parameters to get the current skill name and its input
    """
    skill_name_input = rospy.get_param(
        "/skill_name_input", f"{str(time.time())},None,None"
    )
    skill_name_input = skill_name_input.split(",")
    if len(skill_name_input) == 3:
        # We get the correct format to execute the skill
        skill_name = skill_name_input[1]
        skill_input = skill_name_input[2]
    else:
        # We do not get the skill name and input
        skill_name = "None"
        skill_input = "None"

    return skill_name, skill_input


def reset_ros():
    rospy.set_param("/skill_name_input", f"{str(time.time())},None,None")
    rospy.set_param("/skill_name_suc_msg", f"{str(time.time())},None,None,None")
    rospy.set_param("skill_in_execution_lock", False)
    rospy.set_param("/enable_dwg_object_addition", f"{str(time.time())},True")
    rospy.set_param("/human_action", f"{0},None,None,None")


def reset_skill_msg():
    """Reset the skill message. The format is skill name, success flag, and message string.
    This is useful for returning the message (e.g., if skill fails or not) from spot-sim2real to high-level planner.
    """
    rospy.set_param("/skill_name_suc_msg", f"{str(time.time())},None,None,None")


def reset_skill_name_input(skill_name, succeded, msg):
    """Reset skill name and input, and publish the message"""
    rospy.set_param("/skill_name_input", f"{str(time.time())},None,None")
    rospy.set_param(
        "/skill_name_suc_msg", f"{str(time.time())},{skill_name},{succeded},{msg}"
    )


def execute_skills():
    """Execute skills."""

    # Get the current skill name
    skill_name, skill_input = get_skill_name_and_input_from_ros()
    # Select the skill from the ros buffer and call the skill
    if skill_name in [
        "nav",
        "nav_path_planning_with_view_poses",
        "pick",
        "place",
        "opendrawer",
        "explore",
    ]:
        rospy.set_param("skill_in_execution_lock", True)
        print("=======================")
        print(f"Current skill_name {skill_name} skill_input {skill_input}")
        if skill_name == "nav_path_planning_with_view_poses" or skill_name == "explore":
            # Get the bbox center and bbox extent
            bbox_info = skill_input.split(";")  # in the format of x,y,z
            assert (
                len(bbox_info) >= 6
            ), f"Wrong size of the bbox info, it should be 6, but got {len(bbox_info)}"
            bbox_center = np.array([float(v) for v in bbox_info[0:3]])
            bbox_extent = np.array([float(v) for v in bbox_info[3:6]])
            query_class_names = bbox_info[6:]
            query_class_names[0] = query_class_names[0].replace("_", " ")

            # Get the view poses
            view_poses, category_tag = get_view_poses(
                bbox_center, bbox_extent, query_class_names, False
            )

            # Get the navigation points
            _ = get_navigation_points(
                view_poses,
                bbox_center,
                bbox_extent,
                [1, 0],
                True,
                "simpathplanning.png",
            )
        reset_skill_msg()
        time.sleep(5)
        msg = input("Successful execution! or something" + ": ").strip()
        if "Successful" in msg:
            succeded = True
        else:
            succeded = False
        print(f"Msg: {msg}")
        print("=======================")
        reset_skill_name_input(skill_name, succeded, msg)
        rospy.set_param("/human_action", f"{0},None,None,None")
        rospy.set_param("skill_in_execution_lock", False)

        # The candidate response
        # Can't explore while robot is holding the object, please place first.
        # Unexpected failure! - Failed to pick! Not close enough to the object.
        # Unexpected failure! - Failed to place! Not close enough to {target_object_receptacle} or occluded.


print("Listen to skills...")
reset_ros()
while True:
    execute_skills()
