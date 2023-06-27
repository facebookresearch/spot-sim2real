# mypy: ignore-errors
import curses
import os
import signal
import time

import numpy as np
from spot_wrapper.spot import Spot, SpotCamIds

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
    cmd_id = spot.move_gripper_to_point(point, rpy)
    spot.block_until_arm_arrives(cmd_id, timeout_sec=1.5)
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


def main(spot: Spot):
    """Uses IK to move the arm by setting hand poses"""
    spot.power_on()
    spot.blocking_stand()

    # Open the gripper
    spot.open_gripper()

    # Move arm to initial configuration
    point, rpy = move_to_initial(spot)
    control_arm = False

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
                point, rpy = move_to_initial(spot)
            elif pressed_key == "r":
                # Open gripper
                spot.open_gripper()
            elif pressed_key == "n":
                try:
                    spot.dock(DOCK_ID)
                    spot.home_robot()
                except Exception:
                    print("Dock was not found!")
            elif pressed_key == "i":
                point, rpy = move_to_initial(spot)
            else:
                # Tele-operate either the gripper pose or the base
                if control_arm:
                    if pressed_key in KEY2GRIPPERMOVEMENT:
                        # Move gripper
                        point_rpy += KEY2GRIPPERMOVEMENT[pressed_key]
                        point, rpy = point_rpy[:3], point_rpy[3:]
                        cmd_id = spot.move_gripper_to_point(point, rpy)
                        print("Gripper destination: ", point, rpy)
                        spot.block_until_arm_arrives(
                            cmd_id, timeout_sec=UPDATE_PERIOD * 0.5
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

            if not key_not_applicable:
                last_execution = time.time()

    finally:
        spot.power_off()
        curses.echo()
        stdscr.nodelay(False)
        curses.endwin()


if __name__ == "__main__":
    spot = Spot("ArmKeyboardTeleop")
    with spot.get_lease(hijack=True) as lease:
        main(spot)
