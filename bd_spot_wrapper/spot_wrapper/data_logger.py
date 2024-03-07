import os.path as osp
import pickle as pkl
import time
from typing import Any, Dict, List

import cv2
from spot_wrapper.spot import Spot, SpotCamIds

PATH_TO_LOGS = osp.join(
    osp.dirname(osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))))),
    "data/data_logs",
)


def log(spot):
    data_log_list = []  # type: List[Dict[str, Any]]
    spot.setup_logging_sources(
        [SpotCamIds.HAND_COLOR, SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME]
    )
    st = time.time()
    while time.time() - st < 30:  # TODO: Make this keyboard input based
        # Log data packet
        log_packet = spot.update_logging_data(include_image_data=True, visualize=False)

        # TODO: Add 2 subs for rgb & d from NUC + Realsense
        # TODO: Realsense Cam INtrinsics on a sepearate topic

        data_log_list.append(log_packet)

    # dump data as pkl
    file_name = time.strftime("%Y,%m,%d-%H,%M,%S") + ".pkl"
    with open(osp.join(PATH_TO_LOGS, file_name), "wb") as handle:
        pkl.dump(data_log_list, handle, protocol=4)


def log_replay(logfile_name: str):
    # Read pickle file
    with open(
        osp.join(PATH_TO_LOGS, logfile_name), "rb"
    ) as handle:  # should raise an error if invalid file path
        # Load pickle file
        log_packet_list = pkl.load(handle)  # type: List[Dict[str, Any]]

        # Verification check
        if len(log_packet_list) == 0:
            raise ValueError("Log File is empty!")

        cam_srcs = [
            camera_data["src_info"] for camera_data in log_packet_list[0]["camera_data"]
        ]
        print("Log packet includes data for following cameras : ", cam_srcs)

        # Iterate over log packets and show images from all camera sources
        for log_packet in log_packet_list:
            # Iterate over every camera data and display image
            for camera_data in log_packet["camera_data"]:
                cv2.imshow(camera_data["src_info"], camera_data["raw_image"])
            cv2.waitKey(100)


if __name__ == "__main__":
    spot = Spot("SpotDataLogger")
    # log(spot)
    log_replay("2024,03,06-18,35,23.pkl")
