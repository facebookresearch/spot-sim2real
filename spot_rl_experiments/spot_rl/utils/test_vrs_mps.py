import os
import time
from typing import Any, Dict

import cv2
import numpy as np
from projectaria_tools.core import calibration, data_provider

# from matplotlib import pyplot as plt
from spot_wrapper.april_tag_pose_estimator import AprilTagPoseEstimator

vrsfile = "/home/kavitsha/fair/aria_data/vrs_files/Stroll1.vrs"
DOCK_ID = int(os.environ.get("SPOT_DOCK_ID", 520))

STREAM1_NAME = "camera-rgb"
STREAM2_NAME = "camera-slam-left"
STREAM3_NAME = "camera-slam-right"


class VRSStreamer:
    def __init__(self, vrs_file_path: str):
        assert vrs_file_path is not None and os.path.exists(
            vrs_file_path
        ), "Incorrect VRS file path"

        self._qr_pose_estimator_dict = dict()  # type: Dict[str, Any]
        self._src_calib_params_dict = dict()  # type: Dict[str, Any]
        self._dst_calib_params_dict = dict()  # type: Dict[str, Any]
        self.provider = data_provider.create_vrs_data_provider(vrsfile)
        assert self.provider is not None, "Cannot open VRS file"

    def _loop_stream(self, idx: int, timestamp: int, name_image_transform_zip):
        for name, image, cam_T_marker in name_image_transform_zip:
            image = np.rot90(image, k=3)
            image = np.ascontiguousarray(
                image
            )  # GOD KNOW WHY THIS IS NEEDED -> https://github.com/clovaai/CRAFT-pytorch/issues/84#issuecomment-574683857

            if cam_T_marker is not None:
                cv2.putText(
                    image,
                    "Detected QR Marker",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow(f"Stream - {name}", image)
        cv2.waitKey(1)

    def create_display_windows(self, stream_name_list: list):
        for stream_name in stream_name_list:
            cv2.namedWindow(f"Stream - {stream_name}", cv2.WINDOW_NORMAL)

    def rectify_image(self, stream_name: str, image):
        src_calib = self._src_calib_params_dict.get(stream_name)
        dst_calib = self._dst_calib_params_dict.get(stream_name)
        rectified_image = calibration.distort_by_calibration(
            image, dst_calib, src_calib
        )
        return rectified_image

    def stream_cameras(self, stream_names: list, should_rectify=True, detect_qr=False):
        start_time = time.time()
        self.create_display_windows(stream_name_list=stream_names)

        # Setup rectification
        device_calib = self.provider.get_device_calibration()
        if should_rectify:
            for stream_name in stream_names:
                if stream_name not in self._src_calib_params_dict.keys():
                    self._src_calib_params_dict[
                        stream_name
                    ] = device_calib.get_camera_calib(stream_name)
                    self._dst_calib_params_dict[
                        stream_name
                    ] = calibration.get_linear_camera_calibration(
                        512, 512, 280, stream_name
                    )

        # Setup April tag detection
        if detect_qr:
            for stream_name in stream_names:
                if stream_name not in self._qr_pose_estimator_dict.keys():
                    focal_lengths = self._dst_calib_params_dict.get(
                        stream_name
                    ).get_focal_lengths()
                    principal_point = self._dst_calib_params_dict.get(
                        stream_name
                    ).get_principal_point()
                    calib_dict = {
                        "fx": focal_lengths[0].item(),
                        "fy": focal_lengths[1].item(),
                        "ppx": principal_point[0].item(),
                        "ppy": principal_point[1].item(),
                        "coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
                    }

                    # LETS ONLY DETECT DOCK-ID RIGHT NOW
                    april_tag_pose_estimator = AprilTagPoseEstimator(
                        camera_intrinsics=calib_dict
                    )
                    april_tag_pose_estimator.register_marker_ids([DOCK_ID])
                    self._qr_pose_estimator_dict[stream_name] = april_tag_pose_estimator

        # Assuming all frames are synced
        stream_id = self.provider.get_stream_id_from_label(stream_names[0])
        # num_frames = self.provider.get_num_data(stream_id)
        # CUSTOM RANGE FOR VIDEO - TODO: REMOVE LATER
        custom_range = range(1400, 1550)
        input("Press Enter to continue")
        for frame_idx in custom_range:
            image_list = []
            camera_T_marker_list = []
            for stream_name in stream_names:
                stream_id = self.provider.get_stream_id_from_label(stream_name)
                frame_data = self.provider.get_image_data_by_index(stream_id, frame_idx)
                frame_data_array = frame_data[0].to_numpy_array()
                if should_rectify:
                    frame_data_array = self.rectify_image(
                        stream_name=stream_name, image=frame_data_array
                    )
                    if detect_qr:
                        (
                            frame_data_array,
                            camera_T_marker,
                        ) = self._qr_pose_estimator_dict[
                            stream_name
                        ].detect_markers_and_estimate_pose(
                            image=frame_data_array, should_render=True
                        )
                image_list.append(frame_data_array)
                camera_T_marker_list.append(camera_T_marker)

            self._loop_stream(
                frame_idx, 0000, zip(stream_names, image_list, camera_T_marker_list)
            )

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total time to complete stream = {total_time} seconds")


if __name__ == "__main__":
    vrs_streamer = VRSStreamer(vrs_file_path=vrsfile)

    ##### TIME results for non-rectified non-rotated no-FAIRO
    # stream_list = [STREAM1_NAME, STREAM2_NAME, STREAM3_NAME]  # takes 22 seconds
    # stream_list = [STREAM1_NAME, STREAM2_NAME]                # takes 17 seconds
    # stream_list = [STREAM1_NAME]                              # takes 15 seconds

    ##### TIME results for rectified but non-rotated no-FAIRO
    # stream_list = [STREAM1_NAME, STREAM2_NAME, STREAM3_NAME]  # takes 177 seconds
    # stream_list = [STREAM1_NAME, STREAM2_NAME]                # takes 79 seconds
    # stream_list = [STREAM1_NAME]                              # takes 47 seconds

    stream_list = [STREAM1_NAME, STREAM2_NAME, STREAM3_NAME]
    vrs_streamer.stream_cameras(stream_list, detect_qr=True)


# TODO: Record raw and rectified camera params for each camera in a config file
