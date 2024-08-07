import os
import pickle as pkl
import time
import traceback
from typing import Any, Dict, List

import click
import cv2
import numpy as np
import sophuspy as sp
from spot_wrapper.spot import Spot, SpotCamIds, image_response_to_cv2
from tqdm import tqdm

SPOTDATA_DIR = os.environ.get("SPOTDATA_ROOT", None)

PICKLE_PROTOCOL_VERSION = 4


def get_working_spotdata_dir():
    """Helper method to get os dir to store spot data"""
    spotdata_dir = None
    if SPOTDATA_DIR is None:
        spotdata_dir = os.path.join("~", "Datasets/SpotData/")
        print(
            "Could not find `SPOTDATA_ROOT` environment variable. Please set this variable in bashrc pointing to location where you want to store pointcloud data from Spot."
        )
        print(f"Currently storing the data at {spotdata_dir}")

        if not os.path.exists(spotdata_dir):
            os.makedirs(spotdata_dir)
    else:
        spotdata_dir = SPOTDATA_DIR

    return spotdata_dir


def read_pkl(logfolder_name: str) -> List[Dict[str, Any]]:
    """Read the file"""
    log_packet_list: List[Dict[str, Any]] = None

    spotdata_dir = get_working_spotdata_dir()
    # Check if the path exsits
    path = os.path.join(spotdata_dir, logfolder_name, "data.pkl")
    if not os.path.exists(path):
        raise ValueError(f"Cannot find log file path for {path}!")

    # Read pickle file
    with open(path, "rb") as handle:
        # Load pickle file
        log_packet_list = pkl.load(handle)

        # Verification check
        if len(log_packet_list) == 0:
            raise ValueError("Log File is empty!")

    return log_packet_list


def dump_pkl(log_packet_list: List[Dict[str, Any]], folder_prefix: str = "log"):
    """Dump the data into a new folder with a file called data.pkl. This is guarded with exception handling & triggers a breakpoint on exception"""
    if len(log_packet_list) == 0:
        print("No data to dump into pkl, exiting")
        return
    try:
        spotdata_dir = get_working_spotdata_dir()
        # spotdata_dir = None
        timestamped_folder_name = (
            folder_prefix + "_" + time.strftime("%Y,%m,%d-%H,%M,%S")
        )
        print(f"Folder Name : {timestamped_folder_name}")

        full_folder_path = os.path.join(spotdata_dir, timestamped_folder_name)
        os.makedirs(full_folder_path)

        full_file_path = os.path.join(full_folder_path, "data.pkl")
        print(f"Saving Log file at : {full_file_path}")
        with open(full_file_path, "wb") as handle:
            pkl.dump(log_packet_list, handle, protocol=PICKLE_PROTOCOL_VERSION)
    except Exception:
        print("Failed to create the dump data due to an error")
        print(traceback.format_exc())
        print("***************************************************")
        print("INITIATING BREAKPOINT TO AS SAFETY NET TO SAVE DATA")
        print("***************************************************")
        breakpoint()


def convert_depth_to_img(raw_depth):
    """Helper to convert depth to image (uint8-grayscale)"""
    depth_image = (raw_depth.copy()).astype(np.float32)
    depth_image /= depth_image.max()
    depth_image *= 255.0
    depth_image = depth_image.astype(np.uint8)
    h, w = depth_image.shape[:2]
    depth_image = np.dstack([depth_image, depth_image, depth_image]).reshape(h, w, 3)
    return depth_image


class DataLogger:
    def __init__(self, spot):
        self.spot = spot
        self.log_packet_list = []  # type: List[Dict[str, Any]]
        self.source_list = []  # type: List[str]

    def _verify_sources(self, camera_sources: List[str]) -> bool:
        """Verify the image source inputs"""
        if camera_sources is None or camera_sources == []:
            print("Empty list passed in logger camera sources.")
            return False

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
            elif "intel" in camera_source and not (
                camera_source == SpotCamIds.INTEL_REALSENSE_COLOR
                or camera_source == SpotCamIds.INTEL_REALSENSE_DEPTH
            ):
                print(
                    f"Invalid source name {camera_source}. Please provide correct frame name for intel camera"
                )
                return False
        return True

    def setup_logging_sources(self, camera_sources: List[str]):
        """
        Order .. RGB & then DEPTH_IN_RGB
        DO NOT SUPPORT DEPTH and RGB_IN_DEPTH.
        """
        # By default, always log for hand camera data
        source_list = [
            SpotCamIds.HAND_COLOR,
            SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME,
        ]  # type: List[str]

        # TODO: Add verification for intel source being present or not

        if self._verify_sources(camera_sources=camera_sources):
            for camera_source in camera_sources:
                if camera_source not in source_list:
                    source_list.append(camera_source)
        else:
            print(
                f"Could not verify sources : {camera_sources}. Will default initiate logger for :{source_list}"
            )
            input("Press Enter to continue or Ctrl+C to terminate.")
        self.source_list = source_list

        print(f"Initialized logging for sources : {self.source_list}")

    def update_logging_data(
        self,
        include_image_data: bool = True,
        visualize: bool = False,
        verbose: bool = False,
    ):
        """Log robot data and camera info"""
        log_packet = {
            "timestamp": time.time(),
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "camera_data": [],
            "vision_T_base": None,
            "base_pose_xyt": None,
            "arm_pose": None,
            "is_gripper_holding_item": None,
            "gripper_open_percentage": None,
            "gripper_force_in_hand": None,
        }  # type: Dict[str, Any]

        if include_image_data:
            img_responses_frametree = self.spot.get_image_responses(
                [SpotCamIds.HAND_COLOR, SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME]
            )  # Get Gripper images
            frame_tree_snapshot = img_responses_frametree[
                0
            ].shot.transforms_snapshot  # store transform snapshot
            img_responses = self.spot.get_image_responses(self.source_list)
            base_T_grippercam: sp.SE3 = self.spot.get_sophus_SE3_spot_a_T_b(
                frame_tree_snapshot,
                "body",  # "body",
                "hand_color_image_sensor",
            )
            sp_eye4 = sp.SE3(np.eye(4))
            for i, camera_source in enumerate(self.source_list):
                gripper_T_intel = (
                    self.spot.gripper_T_intel if "intel" in camera_source else sp_eye4
                )
                base_T_camera: sp.SE3 = base_T_grippercam * gripper_T_intel
                log_packet["camera_data"].append(
                    {
                        "src_info": camera_source,
                        "raw_image": image_response_to_cv2(
                            img_responses[i], reorient=True
                        ),  # np.ndarray
                        "camera_intrinsics": self.spot.get_camera_intrinsics_as_3x3(
                            img_responses[i].source.pinhole.intrinsics
                        ),  # np.ndarray
                        "base_T_camera": base_T_camera.matrix(),  # np.ndarray
                    }
                )
                # print("Base_T_camera:", base_T_camera.matrix())
                if visualize:
                    cv2.imshow(camera_source, log_packet["camera_data"][i]["raw_image"])
            if visualize:
                cv2.waitKey(1)

        log_packet["vision_T_base"] = self.spot.get_sophus_SE3_spot_a_T_b(
            frame_tree_snapshot=frame_tree_snapshot, a="vision", b="body"
        ).matrix()  # np.ndarray
        log_packet["base_pose_xyt"] = np.asarray(
            self.spot.get_xy_yaw()
        )  # robot's x,y,yaw w.r.t "home" frame provided spot_wrapper/home.txt exists
        log_packet["arm_pose"] = self.spot.get_arm_joint_positions()
        log_packet["is_gripper_holding_item"] = bool(
            self.spot.robot_state_client.get_robot_state().manipulator_state.is_gripper_holding_item
        )
        log_packet[
            "gripper_open_percentage"
        ] = (
            self.spot.robot_state_client.get_robot_state().manipulator_state.gripper_open_percentage
        )
        log_packet[
            "gripper_force_in_hand"
        ] = 0.0  # self.spot.robot_state_client.get_robot_state().manipulator_state.estimated_end_effector_force_in_hand

        if verbose:
            print(log_packet)
        return log_packet

    def log_data(self):
        # Log data packet
        log_packet = self.update_logging_data(
            include_image_data=True, visualize=True, verbose=False
        )
        self.log_packet_list.append(log_packet)

    def log_data_indefinite(self):
        print("Will start logging data now, hit Ctrl+C to end.")
        try:
            while True:
                self.log_data()
        except Exception as e:
            print(f"Encountered an exception while logging data indefinitely - {e}")
            raise e
        finally:
            # Dump data as pkl
            dump_pkl(log_packet_list=self.log_packet_list)

    def log_data_finite(self, n: int = 10):
        time.sleep(0.8)  # stabilise motion lag

        print(f"Will start logging data now for {n} steps")
        try:
            for _ in tqdm(range(n)):
                self.log_data()
        except Exception as e:
            print(f"Encountered an exception while logging data async - {e}")
            raise e

    @staticmethod
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
                    updated_depth = convert_depth_to_img(
                        camera_data["raw_image"] / 1000.0
                    )
                    cv2.imshow(camera_data["src_info"], updated_depth)
                else:
                    cv2.imshow(camera_data["src_info"], camera_data["raw_image"])
            cv2.waitKey(100)

        freq = len(log_packet_list) / (
            log_packet_list[-1]["timestamp"] - log_packet_list[0]["timestamp"]
        )
        print(f"Freq of data : {freq} hz")


@click.command
@click.option("--log-data", is_flag=True, type=bool, default=False)
@click.option("--replay", is_flag=False, type=str, default=None)
def main(log_data: bool, replay: str):
    if log_data:
        spot = Spot("SpotDataLogger")
        datalogger = DataLogger(spot=spot)
        datalogger.setup_logging_sources(
            [
                SpotCamIds.HAND_COLOR,
                SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME,
            ]
        )
        datalogger.log_data_indefinite()
    else:
        if replay is None:
            raise ValueError("Pass a valid log file from data_logs folder")
        DataLogger.log_replay(replay)


if __name__ == "__main__":
    main()
