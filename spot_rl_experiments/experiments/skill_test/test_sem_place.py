# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import copy
import time

import cv2
import magnum as mn
import numpy as np
import quaternion
import rospy
import sophus as sp
from bosdyn.client.frame_helpers import get_a_tform_b
from scipy.spatial.transform import Rotation
from spot_rl.envs.skill_manager import SpotSkillManager
from spot_rl.utils.construct_configs import construct_config
from spot_rl.utils.search_table_location import (
    detect_place_point_by_pcd_method,
    plot_intel_point_in_gripper_image,
    search_table,
)
from spot_wrapper.spot import SpotCamIds, image_response_to_cv2

if __name__ == "__main__":
    from perception_and_utils.utils.generic_utils import map_user_input_to_boolean

    # Know which location we are doing experiments
    in_fre_lab = True  # map_user_input_to_boolean("Are you Tushar in FRE? Y/N ")
    if in_fre_lab:
        # at FRE
        place_target = "place_taget_test_table"
        place_target_before = "place_taget_test_table_before"
    else:
        # at NYC
        place_target = "high_shelf"  # "high_shelf", "test_desk" #"shelf" #"drawer_pretend_to_be_dish_washer"  # "test_semantic_place_table"

    spotskillmanager = SpotSkillManager(use_mobile_pick=False, use_semantic_place=False)
    # spotskillmanager.gaze_controller.reset_skill("bottle")

    spot = spotskillmanager.spot

    # spotskillmanager.pick("bottle")
    contnue = True  # map_user_input_to_boolean("Should we continue ?")

    config = construct_config()
    owlvitmodel, processor, sammodel = None, None, None
    while contnue:
        if in_fre_lab:
            # pass
            # spotskillmanager.nav(place_target)
            rospy.set_param("is_gripper_blocked", 0)
            # rospy.set_param("pose_correction", [90, 0, 0.0])
            rospy.set_param("pose_correction_success", False)
            # spotskillmanager.pick("bottle")
            # Make it stand in Gaze position
            spot.open_gripper()
            gaze_arm_angles = copy.deepcopy(
                spotskillmanager.pick_config.GAZE_ARM_JOINT_ANGLES
            )
            spot.set_arm_joint_positions(np.deg2rad(gaze_arm_angles), 1)
            time.sleep(1.5)

            # 1 for intel & 0 for gripper which camera to use to generate the place waypoint
            place_point_generation_src: int = 1

            # Garther gripper image response -> snapshot tree
            gripper_resps = spot.get_hand_image()
            snapshot_tree = gripper_resps[0].shot.transforms_snapshot

            # Switch to intel/gripper depending on place_point_generation_src
            rospy.set_param("is_gripper_blocked", place_point_generation_src)

            # Gather image & depth from Intel
            image_resps = (
                spot.get_hand_image()
            )  # assume gripper source, if intel source use caliberation gripper_T_intel.npy to multiply with vision_T_hand
            inrinsics = image_resps[0].source.pinhole.intrinsics

            hand_T_intel = mn.Matrix4(
                np.load("gripper_T_intel.npy").T.tolist()
            )  # Load hand_T_intel from caliberation
            hand_T_intel = (
                mn.Matrix4(np.identity(4).T.tolist())
                if place_point_generation_src == 0
                else hand_T_intel
            )
            image_resps = [
                image_response_to_cv2(image_resp) for image_resp in image_resps
            ]
            body_T_hand = spot.get_magnum_Matrix4_spot_a_T_b(
                "body", "hand_color_image_sensor", snapshot_tree
            )  # load body_T_hand
            body_T_hand = body_T_hand.__matmul__(hand_T_intel)  # body_T_intel
            (
                owlvitmodel,
                processor,
                sammodel,
                placexyz,
                placexyzbeforeconv,
            ) = detect_place_point_by_pcd_method(
                image_resps[0],
                image_resps[1],
                inrinsics,
                body_T_hand,
                owlvitmodel=owlvitmodel,
                proceesor=processor,
                sammodel=sammodel,
            )

            if place_point_generation_src == 1:
                hand_T_intel = sp.SE3(np.load("gripper_T_intel.npy"))
                image_in_gripper = plot_intel_point_in_gripper_image(
                    gripper_resps, hand_T_intel, placexyzbeforeconv
                )
                cv2.imshow("Place Point in Gripper Image", image_in_gripper)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            # xyz- intel -> vision _z  -> home - > xy
            # owlvitmodel, processor, sammodel, placexyz, placexyzbeforeconv = search_table(image_resps[0], image_resps[1], inrinsics, vision_T_hand, owlvitmodel=owlvitmodel, proceesor=processor, sammodel=sammodel)
            # TODO: Convert place_xyz in home frame
            print(
                "Place point before offset adjustment",
                placexyz,
                "Place point before any transformation",
                placexyzbeforeconv,
            )
            # x, y, _ = spot.xy_yaw_global_to_home(*placexyz[:2], 0)
            # placexyz[:2] = np.array([x, y])
            placexyz[-1] += 0.1  # static offset in height
            print(
                "Place point after offset adjustment",
                placexyz,
                "Place point before any transformation",
                placexyzbeforeconv,
            )
            print(
                "Ideal 1.1380085945129395, 0.006437085103243589, 0.8053807616233826 in HOME frame"
            )
            # placexyz = [1.0551527738571167, -0.00035699590807780623, 0.8063808679580688]
            breakpoint()

        else:
            # spotskillmanager.nav("nyc_mg_pos1")
            # spotskillmanager.pick("glass bottle")
            # spotskillmanager.nav(place_target)
            pass

        # spotskillmanager.nav(place_target)
        rospy.set_param("is_gripper_blocked", 0)
        spotskillmanager.place_controller.config.RUNNING_AFTER_GRASP_FOR_PLACE = False
        spotskillmanager.place(*placexyz, True)
        # spot.open_gripper()
        contnue = map_user_input_to_boolean("Do you want to do it again ? Y/N ")

    # spotskillmanager.gaze_controller.reset_skill("bottle")
    # Navigate to dock and shutdown
    # spotskillmanager.dock()
