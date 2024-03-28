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
from spot_wrapper.spot import SpotCamIds, image_response_to_cv2, Spot
from spot_rl.envs.base_env import SpotBaseEnv

EE_GRIPPER_OFFSET = [0.2, 0.0, 0.05] #-> 0.2, 0.05, 0.0
def convert_point_in_body_to_place_waypoint(point_in_body:mn.Vector3, spot:Spot):#3d point
    position, rotation = spot.get_base_transform_to("link_wr1") # body_T_linkwr1 center wr1 in body
    position = [point_in_body.x, point_in_body.y, point_in_body.z] #[position.x, position.y, position.z] # + - point_in_gripper hand_T_hand_sensor
    rotation = [rotation.x, rotation.y, rotation.z, rotation.w]
    #breakpoint()
    wrist_T_base = SpotBaseEnv.spot2habitat_transform(position, rotation)
    # gripper_T_base = wrist_T_base @ mn.Matrix4.translation(
    #     mn.Vector3(EE_GRIPPER_OFFSET)
    # )
    base_place_target_habitat = np.array(wrist_T_base.translation)
    base_place_target = base_place_target_habitat[[0, 2, 1]]

    # TODO: Check if we are missing a multiplication with (-1) on y

    x, y, yaw = spot.get_xy_yaw()
    base_T_global = mn.Matrix4.from_(
        mn.Matrix4.rotation_z(mn.Rad(yaw)).rotation(),
        mn.Vector3(mn.Vector3(x, y, 0.5)),
    )
    global_place_target = base_T_global.transform_point(base_place_target)
    global_place_target = np.array([global_place_target.x, global_place_target.y, global_place_target.z])
    return global_place_target

# def convert_point_in_body_to_place_waypoint(point_in_body:mn.Vector3, spot:Spot):#3d point
#     position, rotation = spot.get_base_transform_to("link_wr1") # body_T_linkwr1 center wr1 in body
#     position = [position.x, position.y, position.z] # + - point_in_gripper hand_T_hand_sensor
#     rotation = [rotation.x, rotation.y, rotation.z, rotation.w]
#     #point_in_wr1 += EE_GRIPPER_OFFSET base_T_wr1 - o in wr1 to pt in base 
#     wrist_T_base = SpotBaseEnv.spot2habitat_transform(position, rotation)
#     gripper_T_base = wrist_T_base @ mn.Matrix4.translation(
#        SpotBaseEnv.spot2habitat_translation(point_in_body)
#     )
#     base_place_target_habitat = np.array(gripper_T_base.translation)
#     base_place_target = base_place_target_habitat[[0, 2, 1]]

#     # TODO: Check if we are missing a multiplication with (-1) on y

#     x, y, yaw = spot.get_xy_yaw()
#     base_T_global = mn.Matrix4.from_(
#         mn.Matrix4.rotation_z(mn.Rad(yaw)).rotation(),
#         mn.Vector3(mn.Vector3(x, y, 0.5)),
#     )
#     global_place_target = base_T_global.transform_point(base_place_target)
#     global_place_target = np.array([global_place_target.x, global_place_target.y, global_place_target.z])
#     return global_place_target

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
            gripper_images = [image_response_to_cv2(gripper_resp) for gripper_resp in gripper_resps]
            gripper_depth = gripper_images[-1]
            intrinsics_gripper = gripper_resps[0].source.pinhole.intrinsics
            snapshot_tree = gripper_resps[0].shot.transforms_snapshot
            # Switch to intel/gripper depending on place_point_generation_src
            rospy.set_param("is_gripper_blocked", place_point_generation_src)

            # Gather image & depth from Intel
            image_resps = (
                spot.get_hand_image()
            )  # assume gripper source, if intel source use caliberation gripper_T_intel.npy to multiply with vision_T_hand
            inrinsics_intel = image_resps[0].source.pinhole.intrinsics

            hand_T_intel = np.load("gripper_T_intel.npy") if place_point_generation_src else np.identity(4)
            #hand_T_intel[:3, :3] = np.identity(3)
            hand_T_intel = mn.Matrix4(hand_T_intel.T.tolist())  # Load hand_T_intel from caliberation
            image_resps = [
                image_response_to_cv2(image_resp) for image_resp in image_resps
            ]
            body_T_hand = spot.get_magnum_Matrix4_spot_a_T_b(
                "body", "hand_color_image_sensor", snapshot_tree
            )  # load body_T_hand
            #body_T_hand = body_T_hand.__matmul__(hand_T_intel)  # body_T_intel
            (
                owlvitmodel,
                processor,
                sammodel,
                placexyz,
                placexyzbeforeconv,
                intel_img_with_vis
            ) = detect_place_point_by_pcd_method(
                image_resps[0],
                image_resps[1],
                gripper_depth,
                inrinsics_intel,
                intrinsics_gripper,
                body_T_hand,
                hand_T_intel,
                #object_name="table top",
                owlvitmodel=owlvitmodel,
                proceesor=processor,
                sammodel=sammodel,
            )

            if place_point_generation_src == 1:
                hand_T_intel = mn.Matrix4(np.identity(4).T.tolist())
                image_in_gripper = plot_intel_point_in_gripper_image(
                    gripper_resps, hand_T_intel, placexyzbeforeconv
                )
                cv2.imwrite("table_detection.png", np.hstack([intel_img_with_vis, image_in_gripper]))
                # cv2.imshow("Place Point in Intel Vs Gripper Images", np.hstack([intel_img_with_vis, image_in_gripper]))
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
            # xyz- intel -> vision _z  -> home - > xy
            # owlvitmodel, processor, sammodel, placexyz, placexyzbeforeconv = search_table(image_resps[0], image_resps[1], inrinsics, vision_T_hand, owlvitmodel=owlvitmodel, proceesor=processor, sammodel=sammodel)
            # TODO: Convert place_xyz in home frame
            # x, y, _ = spot.xy_yaw_global_to_home(*placexyz[:2], 0)
            # placexyz[:2] = np.array([x, y])
            
            print(
                "Place point after offset adjustment",
                placexyz,
                "Place point before any transformation",
                placexyzbeforeconv,
            )
            # body to vision frame
            placexyz = convert_point_in_body_to_place_waypoint(mn.Vector3(*placexyz), spot)
            print(f"PlaceXYZ, {placexyz}")
            print(
                "Ideal 1.0143576860427856 , -0.031638436019420624, 0.8091665506362915 in HOME frame"
            )
            placexyz[0] += 0.2
            #placexyz[1] += 0.05
            placexyz[-1] += 0.15  # static offset in height
            
            breakpoint()

        else:
            # spotskillmanager.nav("nyc_mg_pos1")
            # spotskillmanager.pick("glass bottle")
            # spotskillmanager.nav(place_target)
            pass

        # spotskillmanager.nav(place_target)
        rospy.set_param("is_gripper_blocked", 0)
        spotskillmanager.place_controller.config.RUNNING_AFTER_GRASP_FOR_PLACE = False
        spotskillmanager.place(*placexyz, False)
        # spot.open_gripper()
        contnue = map_user_input_to_boolean("Do you want to do it again ? Y/N ")

    # spotskillmanager.gaze_controller.reset_skill("bottle")
    # Navigate to dock and shutdown
    # spotskillmanager.dock()
