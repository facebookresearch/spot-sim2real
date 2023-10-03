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


######################################################################


class VRSMPSStreamer:
    def __init__(self, vrs_file_path: str, mps_file_path: str):
        assert vrs_file_path is not None and os.path.exists(
            vrs_file_path
        ), "Incorrect VRS file path"
        assert mps_file_path is not None and os.path.exists(
            mps_file_path
        ), "Incorrect MPS dir path"

        self.provider = data_provider.create_vrs_data_provider(vrsfile)
        assert self.provider is not None, "Cannot open VRS file"

        self.device_calib = self.provider.get_device_calibration()

        # TODO: Condition check to ensure stream name is valid

        self._qr_pose_estimator = None  # type: ignore
        self._src_calib_params = None  # type: ignore
        self._dst_calib_params = None  # type: ignore

        # Trajectory and global points
        closed_loop_trajectory_file = os.path.join(
            mps_file_path, "closed_loop_trajectory.csv"
        )
        self.mps_trajectory = mps.read_closed_loop_trajectory(
            closed_loop_trajectory_file
        )
        # global_points_file = os.path.join(mps_file_path, "global_points.csv.gz")
        online_cam_calib_file = os.path.join(mps_file_path, "online_calibration.jsonl")
        self.online_cam_calib = mps.read_online_calibration(online_cam_calib_file)

        self.xyz_trajectory = np.empty([len(self.mps_trajectory), 3])
        # # self.quat_trajectory = np.empty([len(self.mps_trajectory), 4])
        self.trajectory_s = np.empty([len(self.mps_trajectory)])

        self.ariaWorld_T_device_trajectory = []  # type: List[Any]
        self.ariaCorrectedWorld_T_device_trajectory = []  # type: List[Any]
        self.ariaCorrectedWorld_T_cpf_trajectory = []  # type: List[Any]

        # Setup some generic transforms
        self.device_T_cpf = sp.SE3(
            self.device_calib.get_transform_device_cpf().matrix()
        )
        # breakpoint()
        self.cpf_T_device = self.device_T_cpf.inverse()

        # Initialize Trajectory after setting up cpf transforms
        self.initialize_trajectory()

        # sensor_calib_list = [device_calib.get_sensor_calib(label) for label in stream_names][0]
        # print(dir(sensor_calib_list))

    def plot_rgb_and_trajectory(
        self,
        marker: sp.SE3,
        device: sp.SE3,
        rgb,
        timestamp_of_interest=None,
        traj_data=None,
    ):
        fig = plt.figure(figsize=plt.figaspect(2.0))
        fig.suptitle("A tale of 2 subplots")

        _ = fig.add_subplot(1, 2, 1)
        plt.imshow(rgb)

        scene = Scene()
        scene.add_frame("correctedWorld", pose=self.device_T_cpf)
        scene.add_camera(
            "aria", frame="correctedWorld", pose_in_frame=device, size=AXES_SCALE
        )
        scene.add_camera(
            "dock", frame="correctedWorld", pose_in_frame=marker, size=AXES_SCALE
        )

        for camera_name in scene.get_cameras():
            print(
                f"Pose of camera '{camera_name}':\n {scene.get_camera_info(camera_name)['pose_in_frame']}\n"
            )

        plt_ax = scene.visualize(fig=fig, should_return=True)
        plt_ax.plot(traj_data[:, 0], traj_data[:, 1], traj_data[:, 2])
        plt.show()

    def _init_april_tag_detector(self):
        focal_lengths = self._dst_calib_params.get_focal_lengths()
        principal_point = self._dst_calib_params.get_principal_point()
        calib_dict = {
            "fx": focal_lengths[0].item(),
            "fy": focal_lengths[1].item(),
            "ppx": principal_point[0].item(),
            "ppy": principal_point[1].item(),
            "coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        }

        # LETS ONLY DETECT DOCK-ID RIGHT NOW
        self._qr_pose_estimator = AprilTagPoseEstimator(camera_intrinsics=calib_dict)
        self._qr_pose_estimator.register_marker_ids([DOCK_ID])

    def _augmeng_img_with_text(
        self, img: np.ndarray, frame_T_marker: sp.SE3 = None, frame: str = "device"
    ):
        img = np.rot90(img, k=3)
        img = np.ascontiguousarray(
            img
        )  # GOD KNOW WHY THIS IS NEEDED -> https://github.com/clovaai/CRAFT-pytorch/issues/84#issuecomment-574683857

        # Augment displayed image with marker location information
        if frame_T_marker is not None:
            position = frame_T_marker.translation()
            label_img(img, "Detected QR Marker", (50, 50))
            label_img(img, f"Frame = {frame}", (50, 75))
            label_img(img, f"X : {position[0]}", (50, 100))
            label_img(img, f"Y : {position[1]}", (50, 125))
            label_img(img, f"Z : {position[2]}", (50, 150))

        return img

    def _display(self, img: np.ndarray, stream_name: str, wait: int = 1):
        cv2.imshow(f"Stream - {stream_name}", img)
        cv2.waitKey(wait)

    def create_display_window(self, stream_name: str):
        cv2.namedWindow(f"Stream - {stream_name}", cv2.WINDOW_NORMAL)

    def rectify_image(self, image):
        rectified_image = calibration.distort_by_calibration(
            image, self._dst_calib_params, self._src_calib_params
        )
        return rectified_image

    def parse_camera_stream(
        self,
        stream_name: str,
        should_rectify=True,
        detect_qr=False,
        should_display=False,
    ):
        stream_id = self.provider.get_stream_id_from_label(stream_name)

        img_list = []
        img_metadata_list = []
        cpf_T_marker_list = []

        device_T_camera = sp.SE3(
            self.device_calib.get_transform_device_sensor(stream_name).matrix()
        )
        assert device_T_camera is not None

        # Setup camera calibration parameters by over-writing self._src_calib_param & self._dst_calib_param
        if should_rectify:
            self._src_calib_params = self.device_calib.get_camera_calib(stream_name)
            self._dst_calib_params = calibration.get_linear_camera_calibration(
                512, 512, 280, stream_name
            )

        # Setup April tag detection by over-writing self._qr_pose_estimator
        if detect_qr:
            self._init_april_tag_detector()

        if should_display:
            self.create_display_window(stream_name)

        # num_frames = self.provider.get_num_data(stream_id)
        # CUSTOM RANGE FOR VIDEO - TODO: REMOVE LATER
        custom_range = range(1400, 1550)
        for frame_idx in custom_range:
            frame_data = self.provider.get_image_data_by_index(stream_id, frame_idx)
            img = frame_data[0].to_numpy_array()
            img_metadata = frame_data[1]

            cpf_T_marker = None

            if should_rectify:
                img = self.rectify_image(image=img)

            if detect_qr:
                (
                    img,
                    camera_T_marker,
                ) = self._qr_pose_estimator.detect_markers_and_estimate_pose(  # type: ignore
                    image=img, should_render=True, magnum=False
                )
                if camera_T_marker is not None:
                    cpf_T_marker = self.cpf_T_device * device_T_camera * camera_T_marker
                    img = self._augmeng_img_with_text(
                        img=img, frame_T_marker=cpf_T_marker, frame="cpf"
                    )
                    print(
                        f"Time stamp with Detections- {img_metadata.capture_timestamp_ns}"
                    )

                    img_list.append(img)
                    img_metadata_list.append(img_metadata)
                    cpf_T_marker_list.append(cpf_T_marker)

            if should_display:
                self._display(img=img, stream_name=stream_name)

        return img_list, img_metadata_list, cpf_T_marker_list

    def initialize_trajectory(self):

        # frame(ariaWorld) is same as frame(device) at the start
        cpf_T_ariaWorld = self.cpf_T_device

        for i in range(len(self.mps_trajectory)):
            self.trajectory_s[i] = self.mps_trajectory[
                i
            ].tracking_timestamp.total_seconds()

            ariaWorld_T_device = sp.SE3(
                self.mps_trajectory[i].transform_world_device.matrix()
            )
            self.ariaWorld_T_device_trajectory.append(ariaWorld_T_device)

            ariaWorld_T_cpf = ariaWorld_T_device * self.device_T_cpf

            ariaCorrectedWorld_T_device = cpf_T_ariaWorld * ariaWorld_T_device
            self.ariaCorrectedWorld_T_device_trajectory.append(
                ariaCorrectedWorld_T_device
            )

            ariaCorrectedWorld_T_cpf = ariaCorrectedWorld_T_device * self.device_T_cpf
            self.ariaCorrectedWorld_T_cpf_trajectory.append(ariaCorrectedWorld_T_cpf)

            self.xyz_trajectory[i, :] = ariaWorld_T_cpf.translation()
            # self.quat_trajectory[i,:] = self.mps_trajectory[i].transform_world_device.quaternion()
        assert len(self.trajectory_s) == len(
            self.ariaCorrectedWorld_T_device_trajectory
        )

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
        sp_transform_of_interest = self.ariaWorld_T_device_trajectory[
            mps_idx_of_interest
        ]
        return sp_transform_of_interest

    def get_closest_ariaCorrectedWorld_T_cpf_to_timestamp(
        self, timestamp_ns_of_interest: int
    ):
        """
        timestamp_of_interest -> nanoseconds
        """
        mps_idx_of_interest = self.get_closest_mps_idx_to_timestamp_ns(
            timestamp_ns_of_interest
        )
        sp_transform_of_interest = self.ariaCorrectedWorld_T_cpf_trajectory[
            mps_idx_of_interest
        ]
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

    vrs_mps_streamer = VRSMPSStreamer(vrs_file_path=vrsfile, mps_file_path=mpspath)

    (
        img_list,
        img_metadata_list,
        cpf_T_marker_list,
    ) = vrs_mps_streamer.parse_camera_stream(stream_name=STREAM1_NAME, detect_qr=True)

    for img, img_metadata, cpf_T_marker in zip(
        img_list, img_metadata_list, cpf_T_marker_list
    ):
        vrs_timestamp_of_interest_ns = img_metadata.capture_timestamp_ns
        mps_idx_of_intersest = vrs_mps_streamer.get_closest_mps_idx_to_timestamp_ns(
            vrs_timestamp_of_interest_ns
        )
        ariaCorrectedWorld_T_cpf = (
            vrs_mps_streamer.get_closest_ariaCorrectedWorld_T_cpf_to_timestamp(
                vrs_timestamp_of_interest_ns
            )
        )
        ariaCorrectedWorld_T_marker = ariaCorrectedWorld_T_cpf * cpf_T_marker

        marker_position = ariaCorrectedWorld_T_marker.translation()
        device_position = ariaCorrectedWorld_T_cpf.translation()
        delta = marker_position - device_position
        dist = np.linalg.norm(delta)

        print(marker_position, device_position, dist)
        vrs_mps_streamer.plot_rgb_and_trajectory(
            marker=ariaCorrectedWorld_T_marker,
            device=ariaCorrectedWorld_T_cpf,
            rgb=img,
            traj_data=vrs_mps_streamer.xyz_trajectory,
        )

# TODO: Record raw and rectified camera params for each camera in a config file
"""
1. Downsample GPS data
2.
"""
