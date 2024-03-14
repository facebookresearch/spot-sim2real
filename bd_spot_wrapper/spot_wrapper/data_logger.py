import os.path as osp
import pickle as pkl
import time
from typing import Any, Dict, List

import click
import cv2
import numpy as np
from spot_wrapper.spot import Spot, SpotCamIds

PATH_TO_LOGS = osp.join(
    osp.dirname(osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))))),
    "data/data_logs",
)

PICKLE_PROTOCOL_VERSION = 4


##################### Pickle Helpers #####################


def dump_to_pkl(log_packet_list: List[Dict[str, Any]], filename_prefix: str = ""):
    # dump data as pkl
    file_name = filename_prefix + "_" + time.strftime("%Y,%m,%d-%H,%M,%S") + ".pkl"
    print(f"Saving Log file as : {file_name}")
    with open(osp.join(PATH_TO_LOGS, file_name), "wb") as handle:
        pkl.dump(log_packet_list, handle, protocol=PICKLE_PROTOCOL_VERSION)


def read_pkl(logfile_name: str) -> List[Dict[str, Any]]:
    log_packet_list: List[Dict[str, Any]] = None

    # Read pickle file
    with open(
        osp.join(PATH_TO_LOGS, logfile_name), "rb"
    ) as handle:  # should raise an error if invalid file path
        # Load pickle file
        log_packet_list = pkl.load(handle)

        # Verification check
        if len(log_packet_list) == 0:
            raise ValueError("Log File is empty!")

    return log_packet_list


#########################################################

# TODO: Not useful .. remove
def log(
    spot, include_image_data: bool, visualize: bool = True, verbose: bool = False
) -> List[Dict[str, Any]]:
    log_packet_list = []  # type: List[Dict[str, Any]]
    spot.setup_logging_sources(
        [SpotCamIds.HAND_COLOR, SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME]
    )

    st = time.time()
    # while time.time() - st < 30:
    #     time.sleep(5)
    while time.time() - st < 120:  # TODO: Make this keyboard input based
        # Log data packet
        log_packet = spot.update_logging_data(
            include_image_data=include_image_data,
            visualize=visualize,
            verbose=verbose,
        )

        # TODO: Add 2 subs for rgb & d from NUC + Realsense
        # TODO: Realsense Cam INtrinsics on a sepearate topic

        log_packet_list.append(log_packet)

    return log_packet_list


# TODO: Maybe move this to utils?
def convert_depth_to_img(raw_depth):
    depth_image = (raw_depth.copy()).astype(np.float32)
    depth_image /= depth_image.max()
    depth_image *= 255.0
    depth_image = depth_image.astype(np.uint8)
    h, w = depth_image.shape[:2]
    depth_image = np.dstack([depth_image, depth_image, depth_image]).reshape(h, w, 3)
    return depth_image


def log_data(spot, include_image_data: bool):
    # Init data list
    log_packet_list = []  # type: List[Dict[str, Any]]
    spot.setup_logging_sources(
        [SpotCamIds.HAND_COLOR, SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME]
    )

    # Log trajectory until keyboard interrupt
    st = time.time()
    while time.time() - st < 120:  # TODO: Make this keyboard input based
        # Log data
        log_packet = spot.update_logging_data(
            include_image_data=include_image_data,
            visualize=include_image_data,
            verbose=False,
        )
        log_packet_list.append(log_packet)

        # TODO: Add 2 subs for rgb & d from NUC + Realsense
        # TODO: Realsense Cam INtrinsics on a sepearate topic

    # Save to pkl file
    filename_prefix = "completeData" if include_image_data else "trajectory"
    dump_to_pkl(log_packet_list=log_packet_list, filename_prefix=filename_prefix)


def log_data_from_trajectory(spot, trajectory_file: str):

    # Read trajectory data from pkl file
    trajectory_list = read_pkl(trajectory_file)

    # Init data list
    log_packet_list = []  # type: List[Dict[str, Any]]
    spot.setup_logging_sources(
        [SpotCamIds.HAND_COLOR, SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME]
    )

    # TODO: Protect the loop within try-catch

    # Iterate over trajectory, move at each step and log data
    for trajectory_packet in trajectory_list:

        # Move to x,y,theta
        np_xyt = trajectory_packet.get("base_pose_xyt")
        spot.set_base_position(
            x_pos=np_xyt[0], y_pos=np_xyt[1], yaw=np_xyt[2], end_time=10, blocking=True
        )  # TODO: Play with end_time

        # Wait # TODO: Try to adjust this sleep time
        time.sleep(0.5)

        # Log data
        log_packet = spot.update_logging_data(
            include_image_data=True,
            visualize=False,
            verbose=False,
        )
        log_packet_list.append(log_packet)

    dump_to_pkl(log_packet_list=log_packet_list, filename_prefix="completeData")


def log_replay(logfile_name: str):
    log_packet_list = read_pkl(logfile_name)
    cam_srcs = [
        camera_data["src_info"] for camera_data in log_packet_list[0]["camera_data"]
    ]
    print("Log packet includes data for following cameras : ", cam_srcs)

    if log_packet_list[0]["camera_data"] == []:
        print("Missing Camera data in log ... Exiting!")
        return

    # Iterate over log packets and show images from all camera sources
    for i in range(len(log_packet_list)):
        log_packet = log_packet_list[i]
        # Iterate over every camera data and display image
        for camera_data in log_packet["camera_data"]:
            # Show depth data
            if "depth" in camera_data["src_info"]:
                updated_depth = convert_depth_to_img(camera_data["raw_image"] / 1000.0)
                cv2.imshow(
                    "Depth Window",
                    np.hstack([camera_data["raw_image"], updated_depth]),
                )
            # Show RGB data
            else:
                cv2.imshow(camera_data["src_info"], camera_data["raw_image"])
        cv2.waitKey(100)


@click.command
@click.option("--log_trajectory", is_flag=True, type=bool, default=False)
@click.option("--log_complete_data", is_flag=False, type=str, default="")
@click.option(
    "--log_complete_data_from_trajectory", is_flag=False, type=str, default=""
)
@click.option("--replay", is_flag=False, type=str, default=None)
def main(
    log_trajectory: bool,
    log_complete_data: bool,
    log_complete_data_from_trajectory: str,
    replay: str,
):
    if log_trajectory:
        spot = Spot("SpotDataLogger")
        log_data(spot, include_image_data=False)
    elif log_complete_data:
        spot = Spot("SpotDataLogger")
        log_data(spot, include_image_data=True)
    elif log_complete_data_from_trajectory:
        spot = Spot("SpotDataLogger")
        log_data_from_trajectory(spot, log_complete_data_from_trajectory)
    else:
        if replay is None:
            raise ValueError("Pass a valid log file from data_logs folder")
        log_replay(replay)


if __name__ == "__main__":
    main()
