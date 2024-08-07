# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# mypy: ignore-errors
import curses
import os
import signal
import time
from typing import Any, Dict, List

import click
import numpy as np
from spot_wrapper.data_logger import DataLogger, dump_pkl
from spot_wrapper.spot import Spot, SpotCamIds

MOVE_INCREMENT = 0.02
TILT_INCREMENT = 5.0
BASE_ANGULAR_VEL = np.deg2rad(50)
BASE_LIN_VEL = 0.75
DOCK_ID = int(os.environ.get("SPOT_DOCK_ID", 520))
UPDATE_PERIOD = 0.2

# Where the gripper goes to upon initialization
INITIAL_POINT = np.array([0.5, 0.0, 0.35])
INITIAL_RPY = np.deg2rad([0.0, np.pi / 4, 0.0])

INITIAL_ARM_JOINT_ANGLES_GRIPPERCAM_LOGGER = np.deg2rad([0, -91, 33, 0, 100, 0])
INITIAL_ARM_JOINT_ANGLES_INTELCAM_LOGGER = np.deg2rad([0, -100, 33, 0, 75, 0])  # 89

KEY2GRIPPERMOVEMENT = {
    "w": np.array([0.0, 0.0, MOVE_INCREMENT, 0.0, 0.0, 0.0]),  # move up
    "s": np.array([0.0, 0.0, -MOVE_INCREMENT, 0.0, 0.0, 0.0]),  # move down
    "a": np.array([0.0, MOVE_INCREMENT, 0.0, 0.0, 0.0, 0.0]),  # move left
    "d": np.array([0.0, -MOVE_INCREMENT, 0.0, 0.0, 0.0, 0.0]),  # move right
    "q": np.array([MOVE_INCREMENT, 0.0, 0.0, 0.0, 0.0, 0.0]),  # move forward
    "e": np.array([-MOVE_INCREMENT, 0.0, 0.0, 0.0, 0.0, 0.0]),  # move backward
    "k": np.deg2rad([0.0, 0.0, 0.0, 0.0, -TILT_INCREMENT, 0.0]),  # pitch up
    "m": np.deg2rad([0.0, 0.0, 0.0, 0.0, TILT_INCREMENT, 0.0]),  # pitch down
    "h": np.deg2rad([0.0, 0.0, 0.0, 0.0, 0.0, TILT_INCREMENT]),  # pan left
    "j": np.deg2rad([0.0, 0.0, 0.0, 0.0, 0.0, -TILT_INCREMENT]),  # pan right
    "y": np.deg2rad([0.0, 0.0, 0.0, TILT_INCREMENT, 0.0, 0.0]),  # roll up
    "u": np.deg2rad([0.0, 0.0, 0.0, -TILT_INCREMENT, 0.0, 0.0]),  # roll down
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


def move_to_initial(spot, initial_arm_state=0):
    point, rpy = INITIAL_POINT, INITIAL_RPY
    if initial_arm_state == 0:
        spot.move_gripper_to_point(point, rpy, timeout_sec=2)
        cement_arm_joints(spot)
    elif initial_arm_state == 1:
        spot.set_arm_joint_positions(
            positions=INITIAL_ARM_JOINT_ANGLES_GRIPPERCAM_LOGGER,
            travel_time=UPDATE_PERIOD * 5,
        )
    elif initial_arm_state == 2:
        # IntelConfig is giving bad data
        # spot.set_arm_joint_positions(positions=INITIAL_ARM_JOINT_ANGLES_INTELCAM_LOGGER, travel_time=UPDATE_PERIOD*5)
        spot.set_arm_joint_positions(
            positions=INITIAL_ARM_JOINT_ANGLES_INTELCAM_LOGGER,
            travel_time=UPDATE_PERIOD * 5,
        )
    else:
        raise KeyError(
            f"Invalid initial arm state provided {initial_arm_state}. Provide a value between 0-2. 0 for default, 1 for gripperCam logger, 2 for intel realsense logger"
        )
    return point, rpy


def cement_arm_joints(spot):
    arm_proprioception = spot.get_arm_proprioception()
    current_positions = np.array(
        [v.position.value for v in arm_proprioception.values()]
    )
    spot.set_arm_joint_positions(positions=current_positions, travel_time=UPDATE_PERIOD)


def raise_error(sig, frame):
    raise RuntimeError


def rotate(datalogger: DataLogger, n_intervals: int = 16, n_captures: int = 2) -> List:
    x0, y0, theta0 = spot.get_xy_yaw()
    for i in range(n_intervals):
        spot.set_base_position(
            x_pos=x0,
            y_pos=y0,
            yaw=theta0 + (i + 1) * 2 * np.pi / n_intervals,
            end_time=100,
            blocking=True,
        )
        datalogger.log_data_finite(n_captures)


def main(spot: Spot, initial_arm_state: int = 1):
    """Uses IK to move the arm by setting hand poses"""
    spot.power_robot()

    # Open the gripper
    spot.open_gripper()

    sources = []
    if initial_arm_state == 2:
        sources = [SpotCamIds.INTEL_REALSENSE_COLOR, SpotCamIds.INTEL_REALSENSE_DEPTH]
    else:
        sources = [SpotCamIds.HAND_COLOR, SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME]

    # Init logger for hand cameras
    datalogger = DataLogger(spot=spot)
    datalogger.setup_logging_sources(camera_sources=sources)

    # TODO: Add code for moving arm to higher perspective for better view
    # Move arm to initial configuration
    point, rpy = move_to_initial(spot, initial_arm_state)
    control_arm = False

    enable_logger_during_teleop = False

    # Start in-terminal GUI
    stdscr = curses.initscr()
    stdscr.nodelay(True)
    curses.noecho()
    signal.signal(signal.SIGINT, raise_error)
    stdscr.addstr(INSTRUCTIONS)
    last_execution = time.time()
    try:
        while True:
            point_rpy = np.concatenate([point, rpy])
            pressed_key = stdscr.getch()

            key_not_applicable = False

            # Don't update if no key was pressed or we updated too recently
            if pressed_key == -1 or time.time() - last_execution < UPDATE_PERIOD:
                continue

            pressed_key = chr(pressed_key)

            if pressed_key == "z":
                # Quit
                break

            elif pressed_key == "r":
                rotate(datalogger=datalogger)

            elif pressed_key == "l":
                if enable_logger_during_teleop:
                    spot.loginfo(f"{time.time()} - Disabling Logger during teleop")
                    enable_logger_during_teleop = False
                else:
                    spot.loginfo(f"{time.time()} - Enabling Logger during teleop")
                    enable_logger_during_teleop = True

            elif pressed_key == "t":
                # Toggle between controlling arm or base
                control_arm = not control_arm
                if not control_arm:
                    cement_arm_joints(spot)
                spot.loginfo(f"control_arm: {control_arm}")
                time.sleep(0.2)  # Wait before we starting listening again
            elif pressed_key == "g":
                # Grab whatever object is at the center of hand RGB camera image
                image_responses = spot.get_image_responses([SpotCamIds.HAND_COLOR])
                hand_image_response = image_responses[0]  # only expecting one image
                spot.grasp_point_in_image(hand_image_response)
                # Retract arm back to initial configuration
                point, rpy = move_to_initial(spot, 0)
            elif pressed_key == "o":
                # Open gripper
                spot.open_gripper()
            elif pressed_key == "n":
                try:
                    spot.dock(DOCK_ID, home_robot=True)
                except Exception:
                    print("Dock was not found!")
            elif pressed_key == "i":
                point, rpy = move_to_initial(spot, initial_arm_state)
            else:
                # Tele-operate either the gripper pose or the base
                if control_arm:
                    if pressed_key in KEY2GRIPPERMOVEMENT:
                        # Move gripper
                        point_rpy += KEY2GRIPPERMOVEMENT[pressed_key]
                        point, rpy = point_rpy[:3], point_rpy[3:]
                        print("Gripper destination: ", point, rpy)
                        status = spot.move_gripper_to_point(
                            point, rpy, timeout_sec=UPDATE_PERIOD * 0.5
                        )
                        if status is False:
                            print(
                                "Pose out of reach, please bring gripper within valid bounds"
                            )
                elif pressed_key in KEY2BASEMOVEMENT:
                    # Move base
                    x_vel, y_vel, ang_vel = KEY2BASEMOVEMENT[pressed_key]
                    spot.set_base_velocity(
                        x_vel=x_vel,
                        y_vel=y_vel,
                        ang_vel=ang_vel,
                        vel_time=UPDATE_PERIOD * 2,
                    )
                else:
                    key_not_applicable = True

                # Log data
                if enable_logger_during_teleop:
                    # datalogger.log_data()
                    datalogger.log_data_finite(2)

            if not key_not_applicable:
                last_execution = time.time()

    finally:
        spot.power_off()
        curses.echo()
        stdscr.nodelay(False)
        curses.endwin()

        # Save log data
        dump_pkl(datalogger.log_packet_list)


if __name__ == "__main__":
    spot = Spot("ArmKeyboardTeleop")
    with spot.get_lease(hijack=True) as lease:
        main(spot)
