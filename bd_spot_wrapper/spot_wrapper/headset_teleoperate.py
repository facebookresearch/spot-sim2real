# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# mypy: ignore-errors
import curses
import os
import signal
import time

import magnum as mn
import numpy as np
import quaternion
import rospy
from bosdyn.client.math_helpers import quat_to_eulerZYX
from spot_wrapper.spot import Spot, SpotCamIds
from utils_transformation import (
    euler_from_matrix,
    euler_from_quaternion,
    quaternion_from_matrix,
)

MOVE_INCREMENT = 0.02
TILT_INCREMENT = 5.0
BASE_ANGULAR_VEL = np.deg2rad(50)
BASE_LIN_VEL = 0.75
DOCK_ID = int(os.environ.get("SPOT_DOCK_ID", 520))
UPDATE_PERIOD = 0.2

# Where the gripper goes to upon initialization
INITIAL_POINT = np.array([0.5, 0.0, 0.35])
INITIAL_RPY = np.deg2rad([0.0, 0.0, 0.0])
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


def cam_pose_from_opengl_to_opencv(cam_pose: np.ndarray) -> np.ndarray:
    """
    Convert pose matrix from OpenGL (habitat) to OpenCV convention.
    """
    assert cam_pose.shape == (4, 4), f"Invalid pose shape {cam_pose.shape}"
    transform = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    cam_pose = cam_pose @ transform
    return cam_pose


def cam_pose_from_xzy_to_xyz(camera_pose_xzy: np.ndarray) -> np.ndarray:
    """
    Convert from habitat to common convention
    """
    assert camera_pose_xzy.shape == (
        4,
        4,
    ), f"Invalid pose shape {camera_pose_xzy.shape}"
    # Extract rotation matrix and translation vector from the camera pose
    rotation_matrix_xzy = camera_pose_xzy[:3, :3]
    translation_vector_xzy = camera_pose_xzy[:3, 3]

    # Convert rotation matrix from XZ-Y to XYZ convention
    rotation_matrix_xyz = np.array(
        [
            [
                rotation_matrix_xzy[0, 0],
                rotation_matrix_xzy[0, 1],
                rotation_matrix_xzy[0, 2],
            ],
            [
                -rotation_matrix_xzy[2, 0],
                -rotation_matrix_xzy[2, 1],
                -rotation_matrix_xzy[2, 2],
            ],
            [
                rotation_matrix_xzy[1, 0],
                rotation_matrix_xzy[1, 1],
                rotation_matrix_xzy[1, 2],
            ],
        ]
    )

    # Convert translation vector from XZ-Y to XYZ convention
    translation_vector_xyz = np.array(
        [
            translation_vector_xzy[0],
            -translation_vector_xzy[2],
            translation_vector_xzy[1],
        ]
    )

    # Create the new camera pose matrix in XYZ convention
    camera_pose_xyz = np.eye(4)
    camera_pose_xyz[:3, :3] = rotation_matrix_xyz
    camera_pose_xyz[:3, 3] = translation_vector_xyz

    return camera_pose_xyz


def move_to_initial(spot):
    point = INITIAL_POINT
    rpy = INITIAL_RPY
    spot.move_gripper_to_point(point, rpy, timeout_sec=2)
    cement_arm_joints(spot)

    return point, rpy


def get_cur_vr_button():
    """Get the button being pressed"""
    hand_held = rospy.get_param("buttonHeld", None)
    if hand_held is not None and len(hand_held) > 0:
        return True
    return False


def get_cur_vr_pose():
    """Get the VR pose"""
    hand_pos = rospy.get_param("hand_0_pos", None)
    hand_rot = rospy.get_param("hand_0_rot", None)
    return hand_pos, hand_rot


def get_init_transformation_vr():
    """Get the init transformation of the VR"""
    hand_pos = rospy.get_param("hand_0_pos", None)
    hand_rot = rospy.get_param("hand_0_rot", None)
    if hand_pos is None or hand_rot is None:
        return None

    hand_rot_quat = mn.Quaternion(
        mn.Vector3(hand_rot[0], hand_rot[1], hand_rot[2]), hand_rot[3]
    )

    # Get the transformation
    trans = mn.Matrix4.from_(hand_rot_quat.to_matrix(), mn.Vector3(hand_pos))

    return trans


def angle_between(a, b):
    return (b - a + np.pi * 3) % (2 * np.pi) - np.pi


def cement_arm_joints(spot):
    arm_proprioception = spot.get_arm_proprioception()
    current_positions = np.array(
        [v.position.value for v in arm_proprioception.values()]
    )
    spot.set_arm_joint_positions(positions=current_positions, travel_time=UPDATE_PERIOD)


def raise_error(sig, frame):
    raise RuntimeError


def main(spot: Spot):
    """Uses IK to move the arm by setting hand poses"""
    spot.power_robot()

    # Open the gripper
    spot.open_gripper()

    # Move arm to initial configuration
    point, rpy = move_to_initial(spot)

    # Get the trans from hody to hand for robots
    robot_trans = spot.get_magnum_Matrix4_spot_a_T_b("body", "hand")

    # Get the initil transformation of the VR
    vr_trans = get_init_transformation_vr()
    while vr_trans is None:
        vr_trans = get_init_transformation_vr()
    print("Got the VR transformation...")

    # Get the initial VR rotation
    _, init_rot = get_cur_vr_pose()
    while init_rot is None:
        _, init_rot = get_cur_vr_pose()

    cur_trans_xyz = cam_pose_from_xzy_to_xyz(
        cam_pose_from_opengl_to_opencv(np.array(vr_trans))
    )
    r_m, p_m, y_m = euler_from_matrix(cur_trans_xyz)

    # # Start in-terminal GUI
    # stdscr = curses.initscr()
    # stdscr.nodelay(True)
    # curses.noecho()
    # signal.signal(signal.SIGINT, raise_error)
    # stdscr.addstr(INSTRUCTIONS)
    # last_execution = time.time()
    try:
        while True:
            # Don't update  we updated too recently
            # if time.time() - last_execution < UPDATE_PERIOD:
            #     continue

            # Get the current VR hand location
            cur_pos, cur_rot = get_cur_vr_pose()
            cur_pos_relative_to_init = vr_trans.inverted().transform_point(cur_pos)

            # Make the xyz to be correct
            cur_pos_relative_to_init = [
                -cur_pos_relative_to_init[2],
                -cur_pos_relative_to_init[0],
                cur_pos_relative_to_init[1],
            ]

            # Get the current transformation
            cur_trans = get_init_transformation_vr()

            # rpy[0] -= np.pi / 2
            # rpy[2] += np.pi / 2
            # print(rpy)

            cur_trans_xyz = cam_pose_from_xzy_to_xyz(
                cam_pose_from_opengl_to_opencv(np.array(cur_trans))
            )
            r_m_step, p_m_step, y_m_step = euler_from_matrix(cur_trans_xyz)

            delta_r = angle_between(r_m_step, r_m)
            delta_p = angle_between(p_m_step, p_m)
            delta_y = angle_between(y_m_step, y_m)

            rpy = np.array([-delta_p, delta_r, -delta_y])

            # rpy_quaternion = quaternion.from_rotation_matrix(np.array(cur_trans_xyz))
            # rpy_quaternion = np.array([rpy_quaternion.w, rpy_quaternion.x, rpy_quaternion.y, rpy_quaternion.z])
            # r_q,p_q,y_q = euler_from_quaternion(rpy_quaternion)

            print(f"rpy: {rpy}, {np.array([delta_r,delta_p,delta_y])}")

            # Get the point in robot frame
            cur_ee_pos = robot_trans.transform_point(cur_pos_relative_to_init)
            cur_ee_rot = np.array(rpy)
            print(f"target cur_ee_pos/cur_ee_rot: {cur_ee_pos} {cur_ee_rot}")

            # # Move the gripper
            spot.move_gripper_to_point(
                cur_ee_pos, cur_ee_rot, seconds_to_goal=1.0, timeout_sec=0.1
            )
            cement_arm_joints(spot)

            # Get the parameter for the button
            if get_cur_vr_button():
                spot.close_gripper()
            else:
                spot.open_gripper()
            # last_execution = time.time()

    finally:
        # spot.power_off()
        # curses.echo()
        # stdscr.nodelay(False)
        # curses.endwin()
        print("done!")


if __name__ == "__main__":
    spot = Spot("ArmKeyboardTeleop")
    with spot.get_lease(hijack=True) as lease:
        main(spot)
