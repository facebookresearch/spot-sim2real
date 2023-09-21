# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# mypy: ignore-errors
import curses
import os
import signal
import time

import cv2
import numpy as np
from spot_wrapper.april_tag_pose_estimator import AprilTagPoseEstimator
from spot_wrapper.spot import Spot, SpotCamIds, image_response_to_cv2

MOVE_INCREMENT = 0.02
TILT_INCREMENT = 5.0
BASE_ANGULAR_VEL = np.deg2rad(50)
BASE_LIN_VEL = 0.75
DOCK_ID = int(os.environ.get("SPOT_DOCK_ID", 520))
UPDATE_PERIOD = 0.2

# Where the gripper goes to upon initialization
INITIAL_POINT = np.array([0.5, 0.0, 0.35])
INITIAL_RPY = np.deg2rad([0.0, 45.0, 0.0])
KEY2GRIPPERMOVEMENT = {
    "w": np.array([0.0, 0.0, MOVE_INCREMENT, 0.0, 0.0, 0.0]),  # move up
    "s": np.array([0.0, 0.0, -MOVE_INCREMENT, 0.0, 0.0, 0.0]),  # move down
    "a": np.array([0.0, MOVE_INCREMENT, 0.0, 0.0, 0.0, 0.0]),  # move left
    "d": np.array([0.0, -MOVE_INCREMENT, 0.0, 0.0, 0.0, 0.0]),  # move right
    "q": np.array([MOVE_INCREMENT, 0.0, 0.0, 0.0, 0.0, 0.0]),  # move forward
    "e": np.array([-MOVE_INCREMENT, 0.0, 0.0, 0.0, 0.0, 0.0]),  # move backward
    "i": np.deg2rad([0.0, 0.0, 0.0, 0.0, -TILT_INCREMENT, 0.0]),  # pitch up
    "k": np.deg2rad([0.0, 0.0, 0.0, 0.0, TILT_INCREMENT, 0.0]),  # pitch down
    "j": np.deg2rad([0.0, 0.0, 0.0, 0.0, 0.0, TILT_INCREMENT]),  # pan left
    "l": np.deg2rad([0.0, 0.0, 0.0, 0.0, 0.0, -TILT_INCREMENT]),  # pan right
}
KEY2BASEMOVEMENT = {
    "q": [0.0, 0.0, BASE_ANGULAR_VEL],  # turn left
    "e": [0.0, 0.0, -BASE_ANGULAR_VEL],  # turn right
    "w": [BASE_LIN_VEL, 0.0, 0.0],  # go forward
    "s": [-BASE_LIN_VEL, 0.0, 0.0],  # go backward
    "a": [0.0, BASE_LIN_VEL, 0.0],  # strafe left
    "d": [0.0, -BASE_LIN_VEL, 0.0],  # strafe right
}
INSTRUCTIONS = (
    "Use 'wasdqe' for translating gripper, 'ijkl' for rotating.\n"
    "Use 'g' to grasp whatever is at the center of the gripper image.\n"
    "Press 't' to toggle between controlling the arm or the base\n"
    "('wasdqe' will control base).\n"
    "Press 'z' to quit.\n"
)


def move_to_initial(spot):
    point = INITIAL_POINT
    rpy = INITIAL_RPY
    spot.move_gripper_to_point(point, rpy, timeout_sec=1.5)
    cement_arm_joints(spot)

    return point, rpy


def cement_arm_joints(spot):
    arm_proprioception = spot.get_arm_proprioception()
    current_positions = np.array(
        [v.position.value for v in arm_proprioception.values()]
    )
    spot.set_arm_joint_positions(positions=current_positions, travel_time=UPDATE_PERIOD)


def raise_error(sig, frame):
    raise RuntimeError


def _to_camera_metadata_dict(camera_intrinsics):
    """Converts a camera intrinsics proto to a 3x3 matrix as np.array"""
    intrinsics = {
        "fx": camera_intrinsics.focal_length.x,
        "fy": camera_intrinsics.focal_length.x,
        "ppx": camera_intrinsics.principal_point.x,
        "ppy": camera_intrinsics.principal_point.y,
        "coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    return intrinsics


# DEBUG FUNCTION. REMOVE LATER
def take_snapshot(spot: Spot):
    resp_head = spot.experiment(is_hand=False)
    cv2_image_head_r = image_response_to_cv2(resp_head, reorient=True)

    resp_hand = spot.experiment(is_hand=True)
    cv2_image_hand = image_response_to_cv2(resp_hand, reorient=False)
    cv2.imwrite("test_head_right_rgb1.jpg", cv2_image_head_r)
    cv2.imwrite("test_hand_rgb1.jpg", cv2_image_hand)


def decorate_img_with_text(image, frame: str, translation):
    cv2.putText(
        image,
        f"Frame : {frame}",
        (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        f"x : {translation[0]}",
        (50, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        f"y : {translation[1]}",
        (50, 300),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 2500, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        f"z : {translation[2]}",
        (50, 400),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (2500, 0, 0),
        2,
        cv2.LINE_AA,
    )

    return image


# TODO: DECIDE BEST PLACE FOR THIS FUNCTION
def get_body_T_handcam(spot: Spot, frame_tree_snapshot_hand):
    # print(frame_tree_snapshot_hand)
    hand_bd_wrist_T_handcam_dict = (
        frame_tree_snapshot_hand.child_to_parent_edge_map.get(
            "hand_color_image_sensor"
        ).parent_tform_child
    )
    hand_mn_wrist_T_handcam = spot.convert_transformation_from_BD_to_magnun(
        hand_bd_wrist_T_handcam_dict
    )

    hand_bd_body_T_wrist_dict = frame_tree_snapshot_hand.child_to_parent_edge_map.get(
        "arm0.link_wr1"
    ).parent_tform_child
    hand_mn_body_T_wrist = spot.convert_transformation_from_BD_to_magnun(
        hand_bd_body_T_wrist_dict
    )

    # hand_bd_body_T_odom_dict = frame_tree_snapshot_hand.child_to_parent_edge_map.get(
    #     "odom"
    # ).parent_tform_child
    # hand_mn_body_T_odom = spot.convert_transformation_from_BD_to_magnun(
    #     hand_bd_body_T_odom_dict
    # )
    # hand_mn_odom_T_body = hand_mn_body_T_odom.inverted()

    # print("hand__body_T_odom", hand_mn_body_T_odom)
    # print("hand__odom_T_body", hand_mn_odom_T_body)

    hand_mn_body_T_handcam = hand_mn_body_T_wrist @ hand_mn_wrist_T_handcam
    # hand_mn_odom_T_handcam = hand_mn_odom_T_body @ hand_mn_body_T_handcam
    # hand_mn_odom_T_wrist = hand_mn_odom_T_body @ hand_mn_body_T_wrist

    # print(f"wrist_T_handcam - mn: {hand_mn_wrist_T_handcam}")
    # print(f"body_T_wrist - mn: {hand_mn_body_T_wrist}")
    # print(f"body_T_handcam - mn : {hand_mn_body_T_handcam}")
    # print(f"odom_T_handcam - mn : {hand_mn_odom_T_handcam}")
    # print(f"odom_T_wrist - mn : {hand_mn_odom_T_wrist}")
    return hand_mn_body_T_handcam


# TODO: DECIDE BEST PLACE FOR THIS FUNCTION
def get_body_T_headcam(spot: Spot, frame_tree_snapshot_head):
    # print(frame_tree_snapshot_head)
    head_bd_fr_T_frfe_dict = frame_tree_snapshot_head.child_to_parent_edge_map.get(
        "frontright_fisheye"
    ).parent_tform_child
    head_mn_fr_T_frfe_dict = spot.convert_transformation_from_BD_to_magnun(
        head_bd_fr_T_frfe_dict
    )

    head_bd_head_T_fr_dict = frame_tree_snapshot_head.child_to_parent_edge_map.get(
        "frontright"
    ).parent_tform_child
    head_mn_head_T_fr = spot.convert_transformation_from_BD_to_magnun(
        head_bd_head_T_fr_dict
    )

    head_bd_body_T_head_dict = frame_tree_snapshot_head.child_to_parent_edge_map.get(
        "head"
    ).parent_tform_child
    head_mn_body_T_head = spot.convert_transformation_from_BD_to_magnun(
        head_bd_body_T_head_dict
    )

    # head_bd_body_T_odom_dict = frame_tree_snapshot_head.child_to_parent_edge_map.get(
    #     "odom"
    # ).parent_tform_child
    # head_mn_body_T_odom = spot.convert_transformation_from_BD_to_magnun(
    #     head_bd_body_T_odom_dict
    # )
    # head_mn_odom_T_body = head_mn_body_T_odom.inverted()

    # print("head__body_T_odom", head_mn_body_T_odom)
    # print("head__odom_T_body", head_mn_odom_T_body)

    head_mn_head_T_frfe = head_mn_head_T_fr @ head_mn_fr_T_frfe_dict
    head_mn_body_T_frfe = head_mn_body_T_head @ head_mn_head_T_frfe
    # head_mn_odom_T_frfe = head_mn_odom_T_body @ head_mn_body_T_frfe

    # print(f"head__body_T_frfe - mn: {head_mn_body_T_frfe}")
    # print(f"head__odom_T_frfe - mn : {head_mn_odom_T_frfe}").
    return head_mn_body_T_frfe


def main(spot: Spot):
    """Uses IK to move the arm by setting hand poses"""
    # spot.power_robot()

    # # Open the gripper
    # spot.open_gripper()

    # # Move arm to initial configuration
    # point, rpy = move_to_initial(spot)
    # control_arm = False

    # stdscr = curses.initscr()
    # stdscr.nodelay(True)
    # curses.noecho()
    # signal.signal(signal.SIGINT, raise_error)
    # stdscr.addstr(INSTRUCTIONS)
    # last_execution = time.time()

    cv2.namedWindow("hand_image", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("head_right_image", cv2.WINDOW_AUTOSIZE)
    # Start in-terminal GUI

    # # Get Hand camera intrinsics
    hand_cam_intrinsics = spot.get_camera_intrinsics(SpotCamIds.HAND_COLOR)
    hand_cam_intrinsics = _to_camera_metadata_dict(hand_cam_intrinsics)
    hand_cam_pose_estimator = AprilTagPoseEstimator(spot, hand_cam_intrinsics)

    # Get Head(right) camera intrinsics
    head_cam_intrinsics = spot.get_camera_intrinsics(SpotCamIds.FRONTRIGHT_FISHEYE)
    head_cam_intrinsics = _to_camera_metadata_dict(head_cam_intrinsics)
    head_cam_pose_estimator = AprilTagPoseEstimator(spot, head_cam_intrinsics)

    # Register marker ids
    marker_ids_list = [i for i in range(521, 550)]
    hand_cam_pose_estimator.register_marker_ids(marker_ids_list)
    head_cam_pose_estimator.register_marker_ids(marker_ids_list)

    try:
        while True:
            # point_rpy = np.concatenate([point, rpy])
            # pressed_key = stdscr.getch()

            is_marker_detected_from_hand_cam = False
            is_marker_detected_from_head_cam = False

            img_response_hand = spot.experiment(is_hand=True)
            img_response_head = spot.experiment(is_hand=False, is_grayscale=False)

            # ###### FairO frame detection wrt Spot-Camera - START ######
            img_hand = image_response_to_cv2(img_response_hand)
            img_head = image_response_to_cv2(img_response_head)

            (
                img_rend_hand,
                hand_mn_handcam_T_marker,
            ) = hand_cam_pose_estimator.detect_markers_and_estimate_pose(
                img_hand, should_render=True
            )
            (
                img_rend_head,
                head_mn_frfe_T_marker,
            ) = head_cam_pose_estimator.detect_markers_and_estimate_pose(
                img_head, should_render=True
            )

            if hand_mn_handcam_T_marker is not None:
                is_marker_detected_from_hand_cam = True
            if head_mn_frfe_T_marker is not None:
                is_marker_detected_from_head_cam = True

            # ###### FairO frame detection wrt Spot-Camera - END   ######

            # ###### Spot HandCam to Base Transformation - START ######
            frame_tree_snapshot_hand = img_response_hand.shot.transforms_snapshot
            hand_mn_body_T_handcam = get_body_T_handcam(spot, frame_tree_snapshot_hand)

            if is_marker_detected_from_hand_cam:
                hand_mn_base_T_marker = (
                    hand_mn_body_T_handcam @ hand_mn_handcam_T_marker
                )
                # print(f"Marker Location in Base Frame- @@@: {hand_mn_base_T_marker}")

                img_rend_hand = decorate_img_with_text(
                    img_rend_hand, "Body", hand_mn_base_T_marker.translation
                )
            # ###### Spot HandCam to Base Transformation - END   ######

            # ###### Spot BodyCam to Base Transformation - START ######

            frame_tree_snapshot_head = img_response_head.shot.transforms_snapshot
            head_mn_body_T_frfe = get_body_T_headcam(spot, frame_tree_snapshot_head)

            if is_marker_detected_from_head_cam:
                head_mn_base_T_marker = head_mn_body_T_frfe @ head_mn_frfe_T_marker
                # print(f"Marker Location in Base Frame- @@@: {hand_mn_base_T_marker}")

                img_rend_head = decorate_img_with_text(
                    img_rend_head, "Body", head_mn_base_T_marker.translation
                )
            # ###### Spot HandCam to Base Transformation - END   ######

            cv2.imshow("hand_image", img_rend_hand)
            cv2.imshow("head_right_image", img_rend_head)
            cv2.waitKey(1)

    #         key_not_applicable = False

    #         # Don't update if no key was pressed or we updated too recently
    #         if pressed_key == -1 or time.time() - last_execution < UPDATE_PERIOD:
    #             continue

    #         pressed_key = chr(pressed_key)

    #         if pressed_key == "z":
    #             # Quit
    #             break
    #         elif pressed_key == "t":
    #             # Toggle between controlling arm or base
    #             control_arm = not control_arm
    #             if not control_arm:
    #                 cement_arm_joints(spot)
    #             spot.loginfo(f"control_arm: {control_arm}")
    #             time.sleep(0.2)  # Wait before we starting listening again
    #         elif pressed_key == "g":
    #             # Grab whatever object is at the center of hand RGB camera image
    #             image_responses = spot.get_image_responses([SpotCamIds.HAND_COLOR])
    #             hand_image_response = image_responses[0]  # only expecting one image
    #             spot.grasp_point_in_image(hand_image_response)
    #             # Retract arm back to initial configuration
    #             point, rpy = move_to_initial(spot)
    #         elif pressed_key == "r":
    #             # Open gripper
    #             spot.open_gripper()
    #         elif pressed_key == "}":
    #             take_snapshot(spot)
    #         elif pressed_key == "n":
    #             try:
    #                 spot.dock(DOCK_ID)
    #                 spot.home_robot()
    #             except Exception:
    #                 print("Dock was not found!")
    #         elif pressed_key == "i":
    #             point, rpy = move_to_initial(spot)
    #         else:
    #             # Tele-operate either the gripper pose or the base
    #             if control_arm:
    #                 if pressed_key in KEY2GRIPPERMOVEMENT:
    #                     # Move gripper
    #                     point_rpy += KEY2GRIPPERMOVEMENT[pressed_key]
    #                     point, rpy = point_rpy[:3], point_rpy[3:]
    #                     print("Gripper destination: ", point, rpy)
    #                     spot.move_gripper_to_point(
    #                         point, rpy, timeout_sec=UPDATE_PERIOD * 0.5
    #                     )
    #             elif pressed_key in KEY2BASEMOVEMENT:
    #                 # Move base
    #                 x_vel, y_vel, ang_vel = KEY2BASEMOVEMENT[pressed_key]
    #                 spot.set_base_velocity(
    #                     x_vel=x_vel,
    #                     y_vel=y_vel,
    #                     ang_vel=ang_vel,
    #                     vel_time=UPDATE_PERIOD * 2,
    #                 )
    #             else:
    #                 key_not_applicable = True

    #         if not key_not_applicable:
    #             last_execution = time.time()

    finally:
        print("Finally done ")
    # curses.echo()
    # stdscr.nodelay(False)
    # curses.endwin()


if __name__ == "__main__":
    spot = Spot("ArmKeyboardTeleop")
    # with spot.get_lease(hijack=False) as lease:
    main(spot)
    # spot.power_off()
    # take_snapshot(spot)
