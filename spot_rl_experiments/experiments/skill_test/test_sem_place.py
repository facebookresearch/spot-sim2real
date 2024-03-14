# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import copy

import cv2
import numpy as np
import quaternion
import rospy
from bosdyn.client.frame_helpers import get_a_tform_b
from perception_and_utils.utils.generic_utils import map_user_input_to_boolean
from scipy.spatial.transform import Rotation
from spot_rl.envs.skill_manager import SpotSkillManager
from spot_rl.utils.construct_configs import construct_config
from spot_rl.utils.pose_correction import detect
from spot_wrapper.spot import SpotCamIds, image_response_to_cv2

if __name__ == "__main__":

    # Know which location we are doing experiments
    in_fremont_lab = map_user_input_to_boolean("In Fremont location ? Y/N ")
    if in_fremont_lab:
        # at FRE
        place_target = "place_taget_test_table"
    else:
        # at NYC
        place_target = "test_desk"

    # Initialize the skill manager
    spotskillmanager = SpotSkillManager(use_mobile_pick=False, use_semantic_place=True)

    # Start testing
    contnue = True
    while contnue:
        if in_fremont_lab:
            rospy.set_param("is_gripper_blocked", 0)
            rospy.set_param("pose_correction_success", False)
        else:
            pass

        spotskillmanager.place(place_target)
        contnue = map_user_input_to_boolean("Do you want to do it again ? Y/N ")

# The following is a helpful tip to debug the arm
# We get Spot class
# spot = spotskillmanager.spot
# We can move the gripper to a point with x,y,z and roll, pitch, yaw
# spot.move_gripper_to_point((0.55, 0., 0.26), np.deg2rad(np.array([0,0,0])))
# We can also set the robot arm joints
# config = construct_config()
# spot.set_arm_joint_positions(np.deg2rad(config.INITIAL_ARM_JOINT_ANGLES))
