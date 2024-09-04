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
    enable_estimation_before_place = map_user_input_to_boolean(
        "Use waypoint estimator? Y/N "
    )

    place_target = "new_place_waypoint"
    spotskillmanager = SpotSkillManager(
        use_mobile_pick=False, use_semantic_place=False, use_place_ee=False
    )
    spot_pos1 = spotskillmanager.spot.get_arm_joint_positions(as_array=True)
    print("CURR POSE", spot_pos1)

    is_local = False
    if enable_estimation_before_place:
        place_target = None
        is_local = True

    # Start testing
    contnue = True
    INITIAL_ARM_JOINT_ANGLES = [0, -180, 180, 90, 0, -90]
    episode_ctr = 1
    spot_pos, spot_ort = spotskillmanager.spot.get_ee_pos_in_body_frame()
    spot_ort = np.zeros(3)
    while contnue:

        spotskillmanager.spot.open_gripper()
        input("Place an object in Spot's gripper and press Enter to continue...")
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
        file_path = f"logs/static_place/{curr_date}/episode_run2_{episode_ctr}.json"
        with open(file_path, "w") as file:
            json.dump(episode_log, file, indent=4)
            print(f"Saved log: {file_path}")
        episode_ctr += 1
        contnue = map_user_input_to_boolean("Do you want to do it again ? Y/N ")
        spotskillmanager.spot.move_gripper_to_point(spot_pos, spot_ort)
