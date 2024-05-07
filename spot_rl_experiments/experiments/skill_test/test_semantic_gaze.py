# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import time
from copy import deepcopy

import magnum as mn
import numpy as np
import rospy
from spot_rl.envs.skill_manager import SpotSkillManager
from spot_rl.utils.pose_estimation import pose_estimation
from spot_wrapper.spot import Spot, image_response_to_cv2

if __name__ == "__main__":
    from perception_and_utils.utils.generic_utils import map_user_input_to_boolean

    spotskillmanager = SpotSkillManager(use_mobile_pick=True)
    contnue = True
    spot: Spot = spotskillmanager.spot
    pose_estimation_port = 2100
    segmentation_port = 21001
    while contnue:
        spotskillmanager.spot.stand()
        spotskillmanager.spot.open_gripper()
        gaz_arm_angles = deepcopy(spotskillmanager.pick_config.GAZE_ARM_JOINT_ANGLES)
        gaz_arm_angles[-2] = 75
        spotskillmanager.spot.set_arm_joint_positions(np.deg2rad(gaz_arm_angles), 1)
        rospy.set_param("is_gripper_blocked", 0)
        time.sleep(2)
        image_resps_gripper = spot.get_hand_image()
        rospy.set_param("is_gripper_blocked", 1)
        time.sleep(2)
        image_resps = spot.get_hand_image()  # IntelImages
        intrinsics = image_resps[0].source.pinhole.intrinsics
        image_responses = [
            image_response_to_cv2(image_rep) for image_rep in image_resps
        ]
        image_scale = spotskillmanager.pick_config.IMAGE_SCALE

        body_T_gripper = spot.get_magnum_Matrix4_spot_a_T_b(
            "body",
            "hand_color_image_sensor",
            image_resps_gripper[0].shot.transforms_snapshot,
        )
        gripper_T_intel = mn.Matrix4(
            np.load(
                "/home/tushar/Desktop/spot-sim2real/spot_rl_experiments/spot_rl/utils/gripper_T_intel.npy"
            )
        )
        body_T_intel = body_T_gripper @ gripper_T_intel
        graspmode = pose_estimation(
            *image_responses,
            "bottle",
            intrinsics,
            body_T_intel,
            image_scale,
            segmentation_port,
            pose_estimation_port
        )

        run_pick = False  # map_user_input_to_boolean("Do you want to run pick ?")
        if run_pick:
            spotskillmanager.semanticpick("bottle", graspmode)
            spotskillmanager.spot.open_gripper()
        contnue = map_user_input_to_boolean("Do you want to do it again ? Y/N ")

    # Navigate to dock and shutdown
    # spotskillmanager.dock()
