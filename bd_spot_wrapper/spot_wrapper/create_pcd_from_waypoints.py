import json
import math
import os
import time

import numpy as np
from spot_wrapper.data_logger import DataLogger, dump_pkl
from spot_wrapper.spot import Spot, SpotCamIds

FILE_PATH = (
    "/home/tushar/Desktop/spot-sim2real/spot_rl_experiments/configs/data_waypoints.json"
)
INITIAL_ARM_JOINT_ANGLES_GRIPPERCAM_LOGGER = np.deg2rad([0, -91, 33, 0, 100, 0])
INITIAL_ARM_JOINT_ANGLES_INTELCAM_LOGGER = np.deg2rad([0, -100, 33, 0, 75, 0])  # 89
DOCK_ID = int(os.environ.get("SPOT_DOCK_ID", 520))
UPDATE_PERIOD = 0.2
BASE_LIN_VEL = 0.75


def move_to_initial(spot, initial_arm_state=0):
    if initial_arm_state == 1:
        spot.set_arm_joint_positions(
            positions=INITIAL_ARM_JOINT_ANGLES_GRIPPERCAM_LOGGER,
            travel_time=UPDATE_PERIOD * 5,
        )
    elif initial_arm_state == 2:
        spot.set_arm_joint_positions(
            positions=INITIAL_ARM_JOINT_ANGLES_INTELCAM_LOGGER,
            travel_time=UPDATE_PERIOD * 5,
        )
    else:
        raise KeyError(
            f"Invalid initial arm state provided {initial_arm_state}. Provide a value between 0-2. 0 for default, 1 for gripperCam logger, 2 for intel realsense logger"
        )
    return None, None


def rotate(datalogger: DataLogger, n_intervals: int = 16, n_captures: int = 3):
    x0, y0, theta0 = spot.get_xy_yaw()
    for i in range(n_intervals):
        spot.set_base_position(
            x_pos=x0,
            y_pos=y0,
            yaw=theta0 + (i + 1) * np.pi / 8,
            end_time=100,
            blocking=True,
        )
        time.sleep(0.6)  # TODO: Try playing with this param
        # TODO: Add call to log_data_finite
        datalogger.log_data_finite(n_captures)


def get_heading(x_curr, y_curr, x_goal, y_goal):
    # Calculate the angle in radians using atan2
    angle_radians = math.atan2(y_goal - y_curr, x_goal - x_curr)

    # Convert angle from radians to degrees
    angle_degrees = math.degrees(angle_radians)

    return angle_radians, angle_degrees


if __name__ == "__main__":

    initial_arm_state = 2  # set 1 for grippercam, 2 for intelRS

    # Read the JSON file
    with open(FILE_PATH, "r") as file:
        data = json.load(file)

    spot = Spot("PointCloudDataCollection")
    with spot.get_lease(hijack=True) as lease:

        spot.power_robot()

        # Open the gripper
        spot.open_gripper()

        sources = []
        if initial_arm_state == 2:
            sources = [
                SpotCamIds.INTEL_REALSENSE_COLOR,
                SpotCamIds.INTEL_REALSENSE_DEPTH,
            ]
        else:
            sources = [SpotCamIds.HAND_COLOR, SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME]

        # Init logger for hand cameras
        datalogger = DataLogger(spot=spot)
        datalogger.setup_logging_sources(camera_sources=sources)
        move_to_initial(spot, initial_arm_state)

        # Rotate robot after undocking and capture data
        rotate(datalogger=datalogger)

        try:
            # Iterate over the list of dictionaries
            for item in data:
                # Each item is a dictionary with one key-value pair
                key = next(iter(item))
                coordinates = item[key]
                x = coordinates["x"]
                y = coordinates["y"]
                print(f"Navigating to : {key}, x: {x}, y: {y}")

                xc, yc, _ = spot.get_xy_yaw()
                dist = math.sqrt((xc - x) ** 2 + (yc - y) ** 2)
                tim = dist / BASE_LIN_VEL

                heading_rad, heading_deg = get_heading(xc, yc, x, y)

                # Inplace re-oreintation to new goal
                spot.set_base_position(
                    x_pos=xc,
                    y_pos=yc,
                    yaw=heading_rad,
                    end_time=100 * tim,
                    blocking=True,
                )

                spot.set_base_position(
                    x_pos=x,
                    y_pos=y,
                    yaw=heading_rad,
                    end_time=100 * tim * 2,
                    blocking=True,
                )

                # Collect data while rotating
                rotate(datalogger=datalogger)

            x, y = 3.138, 0.186
            xc, yc, _ = spot.get_xy_yaw()
            dist = math.sqrt((xc - x) ** 2 + (yc - y) ** 2)
            tim = dist / BASE_LIN_VEL
            heading_rad, heading_deg = get_heading(xc, yc, x, y)

            # Inplace re-oreintation to new goal
            spot.set_base_position(
                x_pos=xc,
                y_pos=yc,
                yaw=heading_rad,
                end_time=100 * tim,
                blocking=True,
            )
            spot.set_base_position(
                x_pos=3.138,
                y_pos=0.186,
                yaw=heading_rad,
                end_time=100 * tim * 5,
                blocking=True,
            )

            spot.set_base_position(
                x_pos=1.5,
                y_pos=0.0,
                yaw=0,
                end_time=2000,
                blocking=True,
            )
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            spot.dock(DOCK_ID, home_robot=False)
            # Save and poweroff robot
            print("Done recording all data, now saving")
            spot.power_off()
            dump_pkl(datalogger.log_packet_list)
