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


def verify_sources(camera_sources: List[str]) -> bool:
    """Verify the image source inputs"""
    for camera_source in camera_sources:
        if (
            "_depth" in camera_source
            and "_depth_in_visual_frame" not in camera_source
            and "_depth_in_hand_color_frame" not in camera_source
        ):
            print(
                f"Invalid source name {camera_source}. Camera source should either be rgb or depth aligned in rgb"
            )
            return False
    return True


def log_data_async(spot):
    source_list = [
        SpotCamIds.HAND_COLOR,
        SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME,
    ]  # type: List[str]

    log_packet_list = []  # type: List[Dict[str, Any]]

    if not verify_sources(source_list):
        raise ValueError(
            "Camera source verification failed please provide valid sources."
        )

    spot.setup_logging_sources(source_list)
    print("Will start logging data now, hit Ctrl+C to end.")
    try:
        while True:
            # Log data packet
            log_packet = spot.update_logging_data(
                include_image_data=True, visualize=True, verbose=False
            )

            # TODO: Future support: Realsense Cam Intrinsics on a sepearate topic
            log_packet_list.append(log_packet)

    except Exception as e:
        print(f"Encountered an exception while logging data async - {e}")
        raise e
    finally:
        # Dump data as pkl
        dump_pkl(log_packet_list=log_packet_list)


def read_pkl(logfile_name: str) -> List[Dict[str, Any]]:
    """Read the file"""
    log_packet_list: List[Dict[str, Any]] = None

    # Check if the path exsits
    path = osp.join(PATH_TO_LOGS, logfile_name)
    if not osp.exists(path):
        raise ValueError(f"Cannot find log file path for {path}!")

    # Read pickle file
    with open(path, "rb") as handle:
        # Load pickle file
        log_packet_list = pkl.load(handle)

        # Verification check
        if len(log_packet_list) == 0:
            raise ValueError("Log File is empty!")

    return log_packet_list


def dump_pkl(log_packet_list: List[Dict[str, Any]], filename_prefix: str = "log"):
    """Dump the file into pkl"""
    file_name = filename_prefix + "_" + time.strftime("%Y,%m,%d-%H,%M,%S") + ".pkl"
    print(f"Saving Log file as : {file_name}")
    with open(osp.join(PATH_TO_LOGS, file_name), "wb") as handle:
        pkl.dump(log_packet_list, handle, protocol=PICKLE_PROTOCOL_VERSION)


def convert_depth_to_img(raw_depth):
    """Helper to convert depth to image (uint8-grayscale)"""
    depth_image = (raw_depth.copy()).astype(np.float32)
    depth_image /= depth_image.max()
    depth_image *= 255.0
    depth_image = depth_image.astype(np.uint8)
    h, w = depth_image.shape[:2]
    depth_image = np.dstack([depth_image, depth_image, depth_image]).reshape(h, w, 3)
    return depth_image


def log_replay(logfile_name: str):
    log_packet_list = read_pkl(logfile_name)
    cam_srcs = [
        camera_data["src_info"] for camera_data in log_packet_list[0]["camera_data"]
    ]
    print(f"Log packet includes data for following cameras : {cam_srcs}")

    # Iterate over log packets and show images from all camera sources
    for log_packet in log_packet_list:
        # Iterate over every camera data and display image
        for camera_data in log_packet["camera_data"]:
            if "hand_depth_in_hand_color_frame" in camera_data["src_info"]:
                updated_depth = convert_depth_to_img(camera_data["raw_image"] / 1000.0)
                cv2.imshow(camera_data["src_info"], updated_depth)
            else:
                cv2.imshow(camera_data["src_info"], camera_data["raw_image"])
        cv2.waitKey(100)

    freq = len(log_packet_list) / (
        log_packet_list[-1]["timestamp"] - log_packet_list[0]["timestamp"]
    )
    print(f"Freq of data : {freq} hz")


@click.command
@click.option("--log_data", is_flag=True, type=bool, default=False)
@click.option("--replay", is_flag=False, type=str, default=None)
def main(log_data: bool, replay: str):
    if log_data:
        spot = Spot("SpotDataLogger")
        log_data_async(spot)
    else:
        if replay is None:
            raise ValueError("Pass a valid log file from data_logs folder")
        log_replay(replay)


if __name__ == "__main__":
    main()
