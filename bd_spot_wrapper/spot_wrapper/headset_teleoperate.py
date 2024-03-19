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
import rospy
from spot_wrapper.spot import Spot, SpotCamIds

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


def move_to_initial(spot):
    point = INITIAL_POINT
    rpy = INITIAL_RPY
    spot.move_gripper_to_point(point, rpy, timeout_sec=2)
    cement_arm_joints(spot)

    return point, rpy


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

            # Get the point in robot frame
            cur_ee_pos = robot_trans.transform_point(cur_pos_relative_to_init)
            cur_ee_rot = np.array(
                [0, 0, 0]
            )  # [cur_rot[3], cur_rot[0], cur_rot[1], cur_rot[2]]
            print(cur_pos_relative_to_init)
            print(f"target cur_ee_pos/cur_ee_rot: {cur_ee_pos} {cur_ee_rot}")

            # Move the gripper
            spot.move_gripper_to_point(cur_ee_pos, cur_ee_rot, timeout_sec=0.1)
            cement_arm_joints(spot)

            # last_execution = time.time()

    finally:
        spot.power_off()
        # curses.echo()
        # stdscr.nodelay(False)
        # curses.endwin()


if __name__ == "__main__":
    spot = Spot("ArmKeyboardTeleop")
    with spot.get_lease(hijack=True) as lease:
        main(spot)
