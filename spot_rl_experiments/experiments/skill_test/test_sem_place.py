# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# mypy: ignore-errors
import json
import time
from datetime import datetime

import numpy as np
import rospy
from perception_and_utils.utils.generic_utils import map_user_input_to_boolean
from spot_rl.envs.skill_manager import SpotSkillManager

if __name__ == "__main__":
    # Know which location we are doing experiments
    in_fre_lab = map_user_input_to_boolean("Are you Tushar in FRE? Y/N ")
    enable_estimation_before_place = map_user_input_to_boolean(
        "Enable estimation before place? Y/N "
    )

    if in_fre_lab:
        # at FRE
        place_target = "new_place_waypoint"
    else:
        # at NYC
        place_target = "test_desk"

    spotskillmanager = SpotSkillManager(use_mobile_pick=False, use_semantic_place=True)
    spot_pos1 = spotskillmanager.spot.get_arm_joint_positions(as_array=True)

    is_local = False
    if enable_estimation_before_place:
        place_target = None
        is_local = True

    # Start testing
    contnue = True
    INITIAL_ARM_JOINT_ANGLES = [0, -180, 180, 90, 0, -90]
    episode_ctr = 0
    # Get EE Pose Initial
    spot_pos, spot_ort = spotskillmanager.spot.get_ee_pos_in_body_frame()
    # Set Orientation as Zero
    spot_ort = np.zeros(3)
    while contnue:
        # Open Gripper
        spotskillmanager.spot.open_gripper()
        input("Place an object in Spot's gripper and press Enter to continue...")
        # Place Object and Close Gripper
        rospy.set_param("is_gripper_blocked", 0)
        episode_log = {"actions": []}
        spotskillmanager.spot.close_gripper()
        input("waiting for user to get ready with camera")

        spotskillmanager.place(place_target, is_local=is_local, visualize=False)
        skill_log = spotskillmanager.place_controller.skill_result_log
        if "num_steps" not in skill_log:
            skill_log["num_steps"] = 0
        episode_log["actions"].append({"place": skill_log})
        curr_date = datetime.today().strftime("%m-%d-%y")
        file_path = (
            f"logs/semantic_place/{curr_date}/episode_sem_pl_run2_{episode_ctr}.json"
        )
        with open(file_path, "w") as file:
            json.dump(episode_log, file, indent=4)
            print(f"Saved log: {file_path}")
        episode_ctr += 1
        contnue = map_user_input_to_boolean("Do you want to do it again ? Y/N ")
        # Return the arm to the original position
        spot_pos = spotskillmanager.spot.get_ee_pos_in_body_frame()[0]
        spotskillmanager.spot.move_gripper_to_point(spot_pos, spot_ort)
# The following is a helpful tip to debug the arm
# We get Spot class
# spot = spotskillmanager.spot
# We can move the gripper to a point with x,y,z and roll, pitch, yaw
# spot.move_gripper_to_point((0.55, 0., 0.26), np.deg2rad(np.array([0,0,0])))
# We can also set the robot arm joints
# config = construct_config()
# spot.set_arm_joint_positions(np.deg2rad(config.INITIAL_ARM_JOINT_ANGLES))

# In addition, if you want to use semantic place skill based on the grasping orientation, you can do
# spotskillmanager.nav("black_case")
# spotskillmanager.pick("bottle")
# # Fetch the arm joint at grasping location
# ee_orientation_at_grasping = spotskillmanager.gaze_controller.env.ee_orientation_at_grasping
# spotskillmanager.nav("test_desk")
# spotskillmanager.place("test_desk", orientation_at_grasping) # This controls the arm initial orientation
