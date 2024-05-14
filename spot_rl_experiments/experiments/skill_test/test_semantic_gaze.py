# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import time
from copy import deepcopy

import cv2
import magnum as mn
import numpy as np
import rospy
from scipy.spatial.transform import Rotation as R
from spot_rl.envs.skill_manager import SpotSkillManager
from spot_rl.utils.pose_estimation import OrientationSolver, pose_estimation
from spot_wrapper.spot import Spot, image_response_to_cv2

if __name__ == "__main__":
    from perception_and_utils.utils.generic_utils import map_user_input_to_boolean

    spotskillmanager = SpotSkillManager(use_mobile_pick=False, use_semantic_place=True)
    contnue = True
    spot: Spot = spotskillmanager.spot
    pose_estimation_port = 2100
    segmentation_port = 21001
    object_name = "bottle" #"penguin plush toy"
    anchor_pose_number = 4
    orientationsolver: OrientationSolver = OrientationSolver()
    image_src = 1 # 1 for intel & 0 for gripper
    while contnue:
        spotskillmanager.spot.stand()
        spotskillmanager.spot.open_gripper()
        gaz_arm_angles = deepcopy(spotskillmanager.pick_config.GAZE_ARM_JOINT_ANGLES)
        gaz_arm_angles[-2] = 75 if image_src > 0 else gaz_arm_angles[-2]
        spotskillmanager.spot.set_arm_joint_positions(np.deg2rad(gaz_arm_angles), 1)
        input("Continue to pick object ?")
        
        rospy.set_param("is_gripper_blocked", 0)
        time.sleep(1)
        image_resps_gripper = spot.get_hand_image()
        rospy.set_param("is_gripper_blocked", image_src)
        #time.sleep(1)
        image_resps = spot.get_hand_image()  # IntelImages
        intrinsics = image_resps[0].source.pinhole.intrinsics
        image_responses = [
            image_response_to_cv2(image_rep) for image_rep in image_resps
        ]
        image_scale = spotskillmanager.pick_config.IMAGE_SCALE

        body_T_hand = spot.get_magnum_Matrix4_spot_a_T_b(
            "body",
            "link_wr1",
        )
        hand_T_gripper = spot.get_magnum_Matrix4_spot_a_T_b(
            "arm0.link_wr1",
            "hand_color_image_sensor",
            image_resps_gripper[0].shot.transforms_snapshot,
        )
        gripper_T_intel = mn.Matrix4(
            np.load(
                "/home/tushar/Desktop/spot-sim2real/spot_rl_experiments/spot_rl/utils/gripper_T_intel.npy"
            ) if image_src > 0 else np.eye(4)
        )
        body_T_intel = body_T_hand @ (hand_T_gripper @ gripper_T_intel)
        
        # cv2.imwrite(f"object_anchor_pose_{anchor_pose_number}.png", image_responses[0])
        graspmode, spinal_axis = pose_estimation(
            *image_responses,
            object_name,
            intrinsics,
            image_src,
            body_T_intel,
            image_scale,
            segmentation_port,
            pose_estimation_port
        )
        
        #graspmode = "topdown"
        rospy.set_param("graspmode", graspmode)
        spotskillmanager.pick(object_name)

        # Simulate pick orientation
        # R_pick = R.from_matrix(R.from_euler("zyx", [0, -90, 0], True).as_matrix()@R.from_euler("zyx", [0, 0, 0], True).as_matrix()).as_quat()
        # current_point, _ = spot.get_ee_pos_in_body_frame()
        # print(spot.move_gripper_to_point((0.55, 0.0, 0.5), R_pick, 20, 60))
        # status = spot.move_gripper_to_points((0.55, 0.0, 0.5), [R.from_euler("zyx", [0, 90, 0], True).as_quat()], 20, 20)

        current_orientation_at_grasp_in_quat = (
            spot.get_ee_quaternion_in_body_frame().view((np.double, 4))
        )
        current_point, _ = spot.get_ee_pos_in_body_frame()
        correction_angles = orientationsolver.get_correction_angle(current_orientation_at_grasp_in_quat, spinal_axis)
        
        input(f"Should I correct the orientation ?, current ee pos point in body {current_point}, correction_angles {correction_angles}")
        current_point_orig = current_point.copy()
        #Correct the orientation 0.2 m above the current position
        current_point[-1] += 0.2
        correction_status = False
        correction_status = spot.move_gripper_to_points(current_point, [current_orientation_at_grasp_in_quat, np.deg2rad(correction_angles)] )
        #input("Run semantic Place ?")
        #Put back the object
        current_orientation_at_grasp_in_quat = (
            spot.get_ee_quaternion_in_body_frame().view((np.double, 4))
        )
        current_point_orig[-1] += 0.08
        put_back_object_status = spot.move_gripper_to_point(current_point_orig, current_orientation_at_grasp_in_quat, 10, 20 )
        #rospy.set_param("is_gripper_blocked", 0)
        #put_back_object_status, _ = spotskillmanager.place(is_local=True, visualize=False, ee_orientation_at_grasping=np.deg2rad(correction_angles))
        print(f"Correction Status {correction_status}, Putback status {put_back_object_status}")
        #input("Open gripper ?")
        spotskillmanager.spot.open_gripper()
        time.sleep(1)
        spotskillmanager.get_env().reset_arm()
        run_pick = False  # map_user_input_to_boolean("Do you want to run pick ?")
        # if run_pick:
        #     spotskillmanager.semanticpick(object_name, graspmode)
        #     spotskillmanager.spot.open_gripper()
        contnue = map_user_input_to_boolean("Do you want to do it again ? Y/N ")

    # Navigate to dock and shutdown
    spotskillmanager.get_env().reset_arm()
    spotskillmanager.sit()
