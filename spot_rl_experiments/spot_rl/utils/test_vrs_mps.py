import os
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import sophus as sp
from fairotag.scene import Scene
from fairotag.viz import SceneViz
from matplotlib import pyplot as plt
from projectaria_tools.core import calibration, data_provider, mps
from projectaria_tools.core.sensor_data import TimeDomain
from spot_wrapper.april_tag_pose_estimator import AprilTagPoseEstimator

vrsfile = "/home/kavitsha/fair/aria_data/Stroll1/Stroll1.vrs"
mpspath = "/home/kavitsha/fair/aria_data/Stroll1/"
DOCK_ID = int(os.environ.get("SPOT_DOCK_ID", 520))
AXES_SCALE = 0.9
STREAM1_NAME = "camera-rgb"
STREAM2_NAME = "camera-slam-left"
STREAM3_NAME = "camera-slam-right"

############## Simple Helper Methods to keep code clean ##############


def label_img(
    img: np.ndarray,
    text: str,
    org: tuple,
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.8,
    color: tuple = (0, 0, 255),
    thickness: int = 2,
    line_type: int = cv2.LINE_AA,
):
    cv2.putText(
        img,
        text,
        org,
        font_face,
        font_scale,
        color,
        thickness,
        line_type,
    )

    # def plot_trajectory(self, marker: np.ndarray=None, device: np.ndarray=None):
    #     print(f"PLTO TRAJ : {marker}")
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     if marker is not None:
    #         print("Printing Marker")
    #         print(marker[0], marker[1], marker[2])
    #         ax.plot(marker[0], marker[1], marker[2], 'ro', markersize=12)

    #     if device is not None:
    #         print("Printing Marker2")
    #         print(device[0], device[1], device[2])
    #         ax.plot(device[0], device[1], device[2], 'go', markersize=12)
    #     ax.plot(self.xyz_trajectory[:,0], self.xyz_trajectory[:,1], self.xyz_trajectory[:,2])
    #     plt.ioff()  # Use non-interactive mode.
    #     plt.show()  # Show the figure. Won't return until the figure is closed.


def plot_rgb_and_trajectory(
    marker: sp.SE3, device: sp.SE3, rgb, timestamp_of_interest=None, traj_data=None
):
    fig = plt.figure(figsize=plt.figaspect(2.0))
    fig.suptitle("A tale of 2 subplots")

    _ = fig.add_subplot(1, 2, 1)
    plt.imshow(rgb)

    scene = Scene()
    scene.add_camera("aria", frame="world", pose_in_frame=device, size=AXES_SCALE)
    scene.add_camera("dock", frame="world", pose_in_frame=marker, size=AXES_SCALE)

    for camera_name in scene.get_cameras():
        print(
            f"Pose of camera '{camera_name}':\n {scene.get_camera_info(camera_name)['pose_in_frame']}\n"
        )

    plt_ax = scene.visualize(fig=fig, should_return=True)
    plt_ax.plot(traj_data[:, 0], traj_data[:, 1], traj_data[:, 2])
    plt.show()


######################################################################


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

    def _loop_stream(self, name_image_transform_zip):
        for name, image, cam_T_marker in name_image_transform_zip:
            image = np.rot90(image, k=3)
            image = np.ascontiguousarray(
                image
            )  # GOD KNOW WHY THIS IS NEEDED -> https://github.com/clovaai/CRAFT-pytorch/issues/84#issuecomment-574683857

            # Augment displayed image with marker location information
            if cam_T_marker is not None:
                position = cam_T_marker.translation()
                label_img(image, "Detected QR Marker", (50, 50))
                label_img(image, f"X = {[position[0]]}", (50, 75))
                label_img(image, f"Y : {position[1]}", (50, 100))
                label_img(image, f"Z : {position[2]}", (50, 125))
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

        image_data_ret_list = []
        processed_img_ret_list = []
        transformation_ret_list = []

        # Setup rectification
        device_calib = self.provider.get_device_calibration()
        # sensor_calib_list = [device_calib.get_sensor_calib(label) for label in stream_names][0]
        # print(dir(sensor_calib_list))

        sp_device_T_camera_list = [
            sp.SE3(device_calib.get_transform_device_sensor(label).matrix())
            for label in stream_names
        ]
        assert len(sp_device_T_camera_list) == len(stream_names)

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
        # stream_id = self.provider.get_stream_id_from_label(stream_names[0])
        # stream_id = self.provider.get_stream_id_from_label(STREAM1_NAME)
        # num_frames = self.provider.get_num_data(stream_id)
        # CUSTOM RANGE FOR VIDEO - TODO: REMOVE LATER
        custom_range = range(1400, 1550)
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
                            image=frame_data_array, should_render=True, magnum=False
                        )
                        if camera_T_marker is not None:
                            print(
                                f"Time stamp with Detections- {frame_data[1].capture_timestamp_ns}"
                            )
                            image_data_ret_list.append(frame_data)
                            transformation_ret_list.append(
                                sp_device_T_camera_list[0] * camera_T_marker
                            )
                            processed_img_ret_list.append(frame_data_array)

                image_list.append(frame_data_array)
                camera_T_marker_list.append(camera_T_marker)

            self._loop_stream(zip(stream_names, image_list, camera_T_marker_list))

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total time to complete stream = {total_time} seconds")

        return image_data_ret_list, transformation_ret_list, processed_img_ret_list


class MPSStreamer:
    def __init__(self, mps_file_path: str):
        assert mps_file_path is not None and os.path.exists(
            mps_file_path
        ), "Incorrect MPS dir path"

        # Trajectory and global points
        closed_loop_trajectory_file = os.path.join(
            mps_file_path, "closed_loop_trajectory.csv"
        )
        # global_points_file = os.path.join(mps_file_path, "global_points.csv.gz")
        online_cam_calib_file = os.path.join(mps_file_path, "online_calibration.jsonl")
        self.mps_trajectory = mps.read_closed_loop_trajectory(
            closed_loop_trajectory_file
        )
        self.online_cam_calib = mps.read_online_calibration(online_cam_calib_file)

        self.xyz_trajectory = np.empty([len(self.mps_trajectory), 3])
        # # self.quat_trajectory = np.empty([len(self.mps_trajectory), 4])
        self.trajectory_s = np.empty([len(self.mps_trajectory)])
        self.sp_transform_trajectory = []  # type: List[Any]
        self.initialize_trajectory()

    def initialize_trajectory(self):
        for i in range(len(self.mps_trajectory)):
            # breakpoint()
            self.trajectory_s[i] = self.mps_trajectory[
                i
            ].tracking_timestamp.total_seconds()
            self.sp_transform_trajectory.append(
                sp.SE3(self.mps_trajectory[i].transform_world_device.matrix())
            )
            self.xyz_trajectory[i, :] = self.sp_transform_trajectory[i].translation()
            # self.quat_trajectory[i,:] = self.mps_trajectory[i].transform_world_device.quaternion()
        assert len(self.trajectory_s) == len(self.sp_transform_trajectory)

    def get_closest_mps_idx_to_timestamp_ns(self, timestamp_ns_of_interest: int):
        mps_idx_of_interest = np.argmin(
            np.abs(self.trajectory_s * 1e9 - timestamp_ns_of_interest)
        )
        return mps_idx_of_interest

    def get_closest_world_T_device_to_timestamp(self, timestamp_ns_of_interest: int):
        """
        timestamp_of_interest -> nanoseconds
        """
        mps_idx_of_interest = self.get_closest_mps_idx_to_timestamp_ns(
            timestamp_ns_of_interest
        )
        sp_transform_of_interest = self.sp_transform_trajectory[mps_idx_of_interest]
        return sp_transform_of_interest


if __name__ == "__main__":
    ##### TIME results for non-rectified non-rotated no-FAIRO
    # stream_list = [STREAM1_NAME, STREAM2_NAME, STREAM3_NAME]  # takes 22 seconds
    # stream_list = [STREAM1_NAME, STREAM2_NAME]                # takes 17 seconds
    # stream_list = [STREAM1_NAME]                              # takes 15 seconds

    ##### TIME results for rectified but non-rotated no-FAIRO
    # stream_list = [STREAM1_NAME, STREAM2_NAME, STREAM3_NAME]  # takes 177 seconds
    # stream_list = [STREAM1_NAME, STREAM2_NAME]                # takes 79 seconds
    # stream_list = [STREAM1_NAME]                              # takes 47 seconds

    vrs_streamer = VRSStreamer(vrs_file_path=vrsfile)

    stream_list = [STREAM1_NAME]
    img_data_list, t_list, processed_img_list = vrs_streamer.stream_cameras(
        stream_list, detect_qr=True
    )

    mps_streamer = MPSStreamer(mps_file_path=mpspath)

    for img_data, device_T_marker, processed_img in zip(
        img_data_list, t_list, processed_img_list
    ):
        vrs_timestamp_of_interest_ns = img_data[1].capture_timestamp_ns
        mps_idx_of_intersest = mps_streamer.get_closest_mps_idx_to_timestamp_ns(
            vrs_timestamp_of_interest_ns
        )
        ariaworld_T_device = mps_streamer.get_closest_world_T_device_to_timestamp(
            vrs_timestamp_of_interest_ns
        )
        ariaworld_T_marker = ariaworld_T_device * device_T_marker

        marker_position = ariaworld_T_marker.translation()
        device_position = ariaworld_T_device.translation()
        delta = marker_position - device_position
        dist = np.linalg.norm(delta)

        print(marker_position, device_position, dist)
        plot_rgb_and_trajectory(
            marker=ariaworld_T_marker,
            device=ariaworld_T_device,
            rgb=np.rot90(processed_img, k=3),
            traj_data=mps_streamer.xyz_trajectory,
        )

# TODO: Record raw and rectified camera params for each camera in a config file
"""
1. Downsample GPS data
2.
"""
