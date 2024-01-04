# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
from typing import Any, Dict, List, Tuple

import click
import cv2
import numpy as np
import rospy
import sophus as sp
from fairotag.scene import Scene
from matplotlib import pyplot as plt
from perception_and_utils.perception.detector_wrappers.april_tag_detector import (
    AprilTagDetectorWrapper,
)
from perception_and_utils.perception.detector_wrappers.object_detector import (
    ObjectDetectorWrapper,
)
from perception_and_utils.utils.image_utils import rotate_img
from projectaria_tools.core import calibration, data_provider, mps
from scipy.spatial.transform import Rotation
from spot_rl.envs.skill_manager import SpotSkillManager
from spot_rl.utils.utils import ros_frames as rf
from spot_wrapper.spot import Spot
from spot_wrapper.spot_qr_detector import SpotCamIds, SpotQRDetector

AXES_SCALE = 0.9
STREAM1_NAME = "camera-rgb"
STREAM2_NAME = "camera-slam-left"
STREAM3_NAME = "camera-slam-right"
FILTER_DIST = 2.4  # in meters (distance for valid detection)

############## Simple Helper Methods to keep code clean ##############


class AriaReader:
    """
    This class is used to read data from Aria VRS and MPS files
    It can parse through a VRS stream and detect April tags and objects of interest too

    For April tag detection, it uses the AprilTagPoseEstimator class (please refer to AprilTagPoseEstimator.py & AprilTagDetectorWrapper.py)
    For object detection, it uses the Owl-VIT model (please refer to OwlVit.py & ObjectDetectorWrapper.py)

    It also has a few helpers for image rectification, image rotation, image display, etc; and a few helpers to get VRS and MPS file streaming

    Args:
        vrs_file_path (str): Path to VRS file
        mps_file_path (str): Path to MPS file
        verbose (bool, optional): Verbosity flag. Defaults to False.

    """

    def __init__(self, vrs_file_path: str, mps_file_path: str, verbose=False):
        assert vrs_file_path is not None and os.path.exists(
            vrs_file_path
        ), "Incorrect VRS file path"
        assert mps_file_path is not None and os.path.exists(
            mps_file_path
        ), "Incorrect MPS dir path"

        # Verbosity flag for updating images when passed through detectors (this is different from config.VERBOSE)
        self.verbose = verbose

        self.provider = data_provider.create_vrs_data_provider(vrs_file_path)
        assert self.provider is not None, "Cannot open VRS file"

        self.device_calib = self.provider.get_device_calibration()

        # April tag detector object
        self.april_tag_detector = AprilTagDetectorWrapper()

        # Object detection object
        self.object_detector = ObjectDetectorWrapper()

        # Aria device camera calibration parameters
        self._src_calib_params = None  # type: ignore
        self._dst_calib_params = None  # type: ignore

        # Closed loop trajectory
        closed_loop_trajectory_file = os.path.join(
            mps_file_path, "closed_loop_trajectory.csv"
        )
        self.mps_trajectory = mps.read_closed_loop_trajectory(
            closed_loop_trajectory_file
        )

        # XYZ trajectory for mps
        self.xyz_trajectory = np.empty([len(self.mps_trajectory), 3])

        # Timestamps for mps in seconds
        self.trajectory_s = np.empty([len(self.mps_trajectory)])

        # Different transformations along the trajectory
        self.ariaWorld_T_device_trajectory = []  # type: List[Any]

        # Setup some generic transforms
        self.device_T_cpf = sp.SE3(
            self.device_calib.get_transform_device_cpf().to_matrix()
        )

        # Initialize Trajectory after setting up device transforms
        self.initialize_trajectory()

        # sensor_calib_list = [device_calib.get_sensor_calib(label) for label in stream_names][0]
        # Record VRS timestamps of interest based upon user input during vrs parsing
        self.vrs_idx_of_interest_list = []  # type: List[Any]

        # Name window suffix (for cv2 windows)
        self.named_window_suffix = "Stream - "

    def plot_rgb_and_trajectory(
        self,
        pose_list: List[sp.SE3],
        rgb: np.ndarray,
        traj_data: np.ndarray = None,
        block: bool = True,
    ):
        """
        Plot RGB image with trajectory

        Args:
            marker_pose (sp.SE3): Pose of marker in frame of reference
            device_pose_list (List[sp.SE3]): List of device poses in frame of reference
            rgb (np.ndarray): RGB image
            traj_data (np.ndarray): Trajectory data
        """
        fig = plt.figure(figsize=plt.figaspect(2.0))
        fig.suptitle("A tale of 2 subplots")

        _ = fig.add_subplot(1, 2, 1)
        plt.imshow(rgb)

        scene = Scene()
        for i in range(len(pose_list)):
            scene.add_camera(
                f"device_{i}",
                pose_in_frame=pose_list[i],
                size=AXES_SCALE,
            )

        plt_ax = scene.visualize(fig=fig, should_return=True)
        plt_ax.plot(traj_data[:, 0], traj_data[:, 1], traj_data[:, 2])
        plt.show(block=block)

    def rectify_aria_image(self, image: np.ndarray) -> np.ndarray:
        """
        Rectify fisheye image based upon camera calibration parameters
        Ensure you have set self._src_calib_param & self._dst_calib_param

        Args:
            image (np.ndarray): Image to be rectified or undistorted

        Returns:
            np.ndarray: Rectified image
        """
        assert self._src_calib_params is not None and self._dst_calib_params is not None
        rectified_image = calibration.distort_by_calibration(
            image, self._dst_calib_params, self._src_calib_params
        )
        return rectified_image

    def get_vrs_timestamp_from_img_idx(
        self, stream_name: str = STREAM1_NAME, idx_of_interest: int = -1
    ) -> int:
        """
        Get VRS timestamp corresponding to VRS image index

        Args:
            stream_name (str, optional): Stream name. Defaults to STREAM1_NAME.
            idx_of_interest (int, optional): Index of interest. Defaults to -1 i.e. the last position of VRS.

        Returns:
            int: Corresponding VRS timestamp in nanoseconds
        """
        stream_id = self.provider.get_stream_id_from_label(stream_name)
        frame_data = self.provider.get_image_data_by_index(stream_id, idx_of_interest)
        return frame_data[1].capture_timestamp_ns

    def initialize_april_tag_detector(self, outputs: dict = {}):
        """
        Initialize the april tag detector

        Args:
            outputs (dict, optional): Dictionary of outputs from the april tag detector. Defaults to {}.

        Updates:
            - self.april_tag_detector: AprilTagDetectorWrapper object

        Returns:
            outputs (dict): Dictionary of outputs from the april tag detector with following keys:
                - "tag_image_list" - List of np.ndarrays of images with detections
                - "tag_image_metadata_list" - List of image metadata
                - "tag_base_T_marker_list" - List of Sophus SE3 transforms from base frame to marker
                                             where base is "device" frame for aria
        """
        focal_length_obj = self._dst_calib_params.get_focal_lengths()  # type:ignore
        focal_lengths = (focal_length_obj[0].item(), focal_length_obj[1].item())

        principal_point_obj = self._dst_calib_params.get_principal_point()  # type: ignore
        principal_point = (
            principal_point_obj["x"].item(),
            principal_point_obj["y"].item(),
        )

        outputs.update(
            self.april_tag_detector._init_april_tag_detector(
                focal_lengths=focal_lengths, principal_point=principal_point
            )
        )
        return outputs

    def initialize_object_detector(
        self, outputs: dict = {}, object_labels: list = [], meta_objects: List[str] = []
    ):
        """
        Initialize the object detector

        Args:
            outputs (dict, optional): Dictionary of outputs from the object detector. Defaults to {}.
            object_labels (list, optional): List of object labels to detect. Defaults to [].

        Updates:
            - self.object_detector: ObjectDetectorWrapper object

        Returns:
            outputs (dict): Dictionary of outputs from the object detector with following keys:
                - "object_image_list" - List of np.ndarrays of images with detections
                - "object_image_metadata_list" - List of image metadata
                - "object_image_segment" - List of Int signifying which segment the image
                    belongs to; smaller number means latter the segment time-wise
                - "object_score_list" - List of Float signifying the detection score
        """
        outputs.update(
            self.object_detector._init_object_detector(
                object_labels + meta_objects,
                verbose=self.verbose,
                version=2,
            )
        )
        self.object_detector._core_objects = object_labels
        self.object_detector._meta_objects = meta_objects

        return outputs

    def parse_camera_stream(
        self,
        stream_name: str,
        detect_qr: bool = False,
        should_display: bool = True,
        detect_objects: bool = False,
        object_labels: List[str] = None,
        iteration_range: Tuple[int, int] = None,
        reverse: bool = False,
        meta_objects: List[str] = ["hand"],
    ) -> Dict[str, Any]:
        """Parse linearly through a camera stream and return a dict of detections
        Detection types supported:
        - April Tag
        - Object Detection

        Args:
            stream_name (str): Stream name
            detect_qr (bool, optional): Boolean to indicate if QR code (Dock ID) should be detected. Defaults to False.
            should_display (bool, optional): Boolean to indicate if image should be displayed. Defaults to True.
            detect_objects (bool, optional): Boolean to indicate if object detection should be performed. Defaults to False.
            object_labels (List[str], optional): List of object labels to be detected. Defaults to None.
            reverse (bool, optional): Boolean to indicate if VRS stream should be parsed in reverse. Defaults to False.
                                      Useful based on the algorithm to detect objects from VRS stream.

        April tag outputs:
        - "tag_image_list" - List of np.ndarrays of images with detections
        - "tag_image_metadata_list" - List of image metadata
        - "tag_base_T_marker_list" - List of Sophus SE3 transforms from Device frame to marker
                                     where base is "device" frame for aria
           CPF is the center frame of Aria with Z pointing out, X pointing up
           and Y pointing left
           Device frame is the base frame of Aria which is aligned with left-slam camera frame

        Object detection outputs:
        - "object_image_list" - List of np.ndarrays of images with detections
        - "object_image_metadata_list" - List of image metadata
        - "object_image_segment" - List of Int signifying which segment the image
            belongs to; smaller number means latter the segment time-wise
        - "object_score_list" - List of Float signifying the detection score
        """
        # Get stream id from stream name
        stream_id = self.provider.get_stream_id_from_label(stream_name)

        # Get device_T_camera i.e. transformation from camera frame of interest to device frame (i.e. left slam camera frame)
        device_T_camera = sp.SE3(
            self.device_calib.get_transform_device_sensor(stream_name).to_matrix()
        )
        assert device_T_camera is not None

        # Setup camera calibration parameters by over-writing self._src_calib_param & self._dst_calib_param
        self._src_calib_params = self.device_calib.get_camera_calib(stream_name)
        self._dst_calib_params = calibration.get_linear_camera_calibration(
            512, 512, 280, stream_name
        )

        outputs: Dict[str, Any] = {}

        # Setup April tag detection by over-writing self._qr_pose_estimator if needed
        if detect_qr:
            outputs = self.initialize_april_tag_detector(outputs)

        # Setup object detection (Owl-ViT) if needed
        if detect_objects:
            self.initialize_object_detector(outputs, object_labels, meta_objects)

        if should_display:
            cv2.namedWindow(
                f"{self.named_window_suffix + stream_name}", cv2.WINDOW_NORMAL
            )

        # Logic for iterating through VRS stream
        num_frames = self.provider.get_num_data(stream_id)
        iteration_delta = -1 if reverse else 1
        if iteration_range is None:
            iteration_range = (0, num_frames)

        if reverse:
            start_frame = iteration_range[1] - 1
            end_frame = iteration_range[0]
        else:
            start_frame = iteration_range[0]
            end_frame = iteration_range[1] - 1
        custom_range = range(start_frame, end_frame, iteration_delta)

        # Iterate through VRS stream
        for frame_idx in custom_range:
            # Get image data for frame
            frame_data = self.provider.get_image_data_by_index(stream_id, frame_idx)
            img = frame_data[0].to_numpy_array()
            img_metadata = frame_data[1]

            # Rectify current image frame
            img = self.rectify_aria_image(image=img)

            # Initialize camera_T_marker to None & object_scores to empty dict for current image frame
            camera_T_marker = None
            object_scores = {}  # type: Dict[str, Any]

            # Detect QR code in current image frame
            if detect_qr:
                (img, camera_T_marker,) = self.april_tag_detector.process_frame(
                    img_frame=img
                )  # type: ignore

            # Rotate current image frame
            img = rotate_img(img=img, num_of_rotation=3)

            if self.object_detector.is_enabled:
                (img, object_scores) = self.object_detector.process_frame(img_frame=img)

            # If april tag is detected, compute the transformation of marker in cpf frame
            if camera_T_marker is not None:
                device_T_marker = device_T_camera * camera_T_marker
                img, outputs = self.april_tag_detector.get_outputs(
                    img_frame=img,
                    outputs=outputs,
                    base_T_marker=device_T_marker,
                    timestamp=rospy.Time.now(),
                    img_metadata=img_metadata,
                )

            # If object is detected, update the outputs
            if object_scores is not {}:
                img, outputs = self.object_detector.get_outputs(
                    img_frame=img,
                    outputs=outputs,
                    object_scores=object_scores,
                    img_metadata=img_metadata,
                )

            # Display current image frame
            if should_display:
                cv2.imshow(
                    f"{self.named_window_suffix + stream_name}",
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                )
                cv2.waitKey(1)

        cv2.destroyAllWindows()
        return outputs

    def initialize_trajectory(self):
        """
        Initialize trajectory data from MPS file for easy access
        """
        # frame(ariaWorld) is same as frame(device) at the start

        for i in range(len(self.mps_trajectory)):
            self.trajectory_s[i] = self.mps_trajectory[
                i
            ].tracking_timestamp.total_seconds()
            ariaWorld_T_device = sp.SE3(
                self.mps_trajectory[i].transform_world_device.to_matrix()
            )
            self.ariaWorld_T_device_trajectory.append(ariaWorld_T_device)
            self.xyz_trajectory[i, :] = ariaWorld_T_device.translation()
            # self.quat_trajectory[i,:] = self.mps_trajectory[i].transform_world_device.quaternion()
        assert len(self.trajectory_s) == len(self.ariaWorld_T_device_trajectory)

    def get_closest_mps_idx_to_timestamp_ns(self, timestamp_ns_of_interest: int) -> int:
        """
        Returns the index of the closest MPS timestamp to the VRS timestamp.
        VRS & MPS timestamps are NOT 100% synced

        Args:
            timestamp_of_interest (int): VRS timestamp in nanoseconds

        Returns:
            mps_idx_of_interest (int): Index of closest MPS timestamp to given VRS timestamp
        """
        # VRS timestamps are NOT 100% synced with MPS timestamps
        # So we find the closest MPS timestamp to the VRS timestamp
        mps_idx_of_interest = np.argmin(
            np.abs(self.trajectory_s * 1e9 - timestamp_ns_of_interest)
        )

        return mps_idx_of_interest

    def get_closest_ariaWorld_T_device_to_timestamp(
        self, timestamp_ns_of_interest: int
    ) -> sp.SE3:
        """
        Returns the transformation of aria device frame to aria world frame at the
        closest MPS timestamp to the given VRS timestamp.
        VRS & MPS timestamps are NOT 100% synced

        Args:
            timestamp_of_interest (int): VRS timestamp in nanoseconds

        Returns:
            sp_transform_of_interest (sp.SE3): Transformation of aria device frame to aria world frame
        """
        mps_idx_of_interest = self.get_closest_mps_idx_to_timestamp_ns(
            timestamp_ns_of_interest
        )
        ariaWorld_T_device_of_insterest = self.ariaWorld_T_device_trajectory[
            mps_idx_of_interest
        ]
        return ariaWorld_T_device_of_insterest

    def get_avg_ariaWorld_T_marker(
        self,
        img_metadata_list: List,
        device_T_marker_list: List,
        filter_dist: float = FILTER_DIST,
    ) -> sp.SE3:
        """
        Returns the average transformation of aria world frame to marker frame

        We get a device_T_marker for each frame in which marker is detected.
        Depending on the frame rate of image capture, multiple frames may have captured the marker.
        Averaging all transforms would be best way to compensate for any noise that may exist in any frame's detections
        camera_T_marker is used to compute device_T_marker[i] and thus ariaWorld_T_marker[i].
        Then we average all ariaWorld_T-marker to find average marker pose wrt ariaWorld.

        NOTE: To compute average of SE3 matrix, we find the average of translation and rotation separately.
              The average rotation is obtained by averaging the quaternions.
        NOTE: Since multiple quaternions can represent the same rotation, we ensure that the 'w' component of the
              quaternion is always positive for effective averaging.

        Args:
            img_metadata_list (List): List of image metadata
            device_T_marker_list (List): List of Sophus SE3 transforms from Device frame to marker
            filter_dist (float, optional): Distance threshold for valid detections. Defaults to FILTER_DIST.
        """
        marker_position_list = []
        marker_quaternion_list = []

        for img_metadata, device_T_marker in zip(
            img_metadata_list, device_T_marker_list
        ):
            vrs_timestamp_of_interest_ns = (
                img_metadata.capture_timestamp_ns
            )  # maybe this can be replaced
            ariaWorld_T_device = self.get_closest_ariaWorld_T_device_to_timestamp(
                vrs_timestamp_of_interest_ns
            )
            ariaWorld_T_marker = ariaWorld_T_device * device_T_marker

            marker_position = ariaWorld_T_marker.translation()
            device_position = ariaWorld_T_device.translation()
            delta = marker_position - device_position
            dist = np.linalg.norm(delta)

            # Consider only those detections where detected marker is within a certain distance of the camera
            if dist < filter_dist:
                marker_position_list.append(marker_position)
                quat = Rotation.from_matrix(
                    ariaWorld_T_marker.rotationMatrix()
                ).as_quat()

                # Ensure quaternion's w is always positive for effective averaging as multiple quaternions can represent the same rotation
                if quat[3] > 0:
                    quat = -1.0 * quat
                marker_quaternion_list.append(quat)

        marker_position_np = np.array(marker_position_list)
        avg_marker_position = np.mean(marker_position_np, axis=0)

        marker_quaternion_np = np.array(marker_quaternion_list)
        avg_marker_quaternion = np.mean(marker_quaternion_np, axis=0)

        avg_ariaWorld_T_marker = sp.SE3(
            Rotation.from_quat(avg_marker_quaternion).as_matrix(), avg_marker_position
        )

        return avg_ariaWorld_T_marker


@click.command()
@click.option("--data-path", help="Path to the data directory", type=str)
@click.option("--vrs-name", help="Name of the vrs file", type=str)
@click.option("--dry-run", type=bool, default=False)
@click.option("--verbose", type=bool, default=True)
@click.option("--use-spot/--no-spot", default=True)
@click.option("--object-names", type=str, multiple=True, default=["smartphone"])
@click.option("--qr/--no-qr", default=True)
def main(
    data_path: str,
    vrs_name: str,
    dry_run: bool,
    verbose: bool,
    use_spot: bool,
    object_names: List[str],
    qr: bool,
):
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("VRS_MPS_STREAMER")
    vrsfile = os.path.join(data_path, vrs_name + ".vrs")
    vrs_mps_streamer = AriaReader(
        vrs_file_path=vrsfile, mps_file_path=data_path, verbose=verbose
    )

    outputs = vrs_mps_streamer.parse_camera_stream(
        stream_name=STREAM1_NAME,
        detect_qr=qr,
        detect_objects=True,
        object_labels=list(object_names),
        reverse=True,
    )
    # tag_img_list = outputs["tag_image_list"]
    if qr:
        tag_img_metadata_list = outputs["tag_image_metadata_list"]
        tag_base_T_marker_list = outputs["tag_base_T_marker_list"]

        avg_ariaWorld_T_marker = vrs_mps_streamer.get_avg_ariaWorld_T_marker(
            tag_img_metadata_list,
            tag_base_T_marker_list,
            filter_dist=FILTER_DIST,
        )
        logger.debug(avg_ariaWorld_T_marker)
        avg_marker_T_ariaWorld = avg_ariaWorld_T_marker.inverse()
    else:
        avg_ariaWorld_T_marker = sp.SE3()

    # find best device location, wrt AriaWorld, for best scored object detection
    # FIXME: extend to multiple objects

    # object detection metadata
    best_object_frame_idx = {}
    best_object_frame_timestamp_ns = {}
    best_object_ariaWorld_T_device = {}
    best_object_img = {}
    for object_name in object_names:
        # TODO: what happens when object is not detected?
        best_object_frame_idx[object_name] = outputs["object_score_list"][
            object_name
        ].index(max(outputs["object_score_list"][object_name]))
        best_object_frame_timestamp_ns[object_name] = outputs[
            "object_image_metadata_list"
        ][object_name][best_object_frame_idx[object_name]].capture_timestamp_ns
        best_object_ariaWorld_T_device[
            object_name
        ] = vrs_mps_streamer.get_closest_ariaWorld_T_device_to_timestamp(
            best_object_frame_timestamp_ns[object_name]
        )
        best_object_img[object_name] = outputs["object_image_list"][object_name][
            best_object_frame_idx[object_name]
        ]

    if use_spot:
        spot = Spot("ArmKeyboardTeleop")
        cam_id = SpotCamIds.HAND_COLOR
        spot_qr = SpotQRDetector(spot=spot, cam_ids=[cam_id])
        (
            avg_spotWorld_T_marker,
            avg_spot_T_marker,
        ) = spot_qr.get_avg_spotWorld_T_marker(cam_id=cam_id)

        logger.debug(avg_spotWorld_T_marker)

        avg_spotWorld_T_ariaWorld = avg_spotWorld_T_marker * avg_marker_T_ariaWorld
        avg_ariaWorld_T_spotWorld = avg_spotWorld_T_ariaWorld.inverse()

        avg_spot_T_ariaWorld = avg_spot_T_marker * avg_marker_T_ariaWorld
        avg_ariaWorld_T_spot = avg_spot_T_ariaWorld.inverse()

        spotWorld_T_device_trajectory = np.array(
            [
                (avg_spotWorld_T_ariaWorld * ariaWorld_T_device).translation()
                for ariaWorld_T_device in vrs_mps_streamer.ariaWorld_T_device_trajectory
            ]
        )

    for i in range(len(object_names)):
        # choose the next object to pick
        next_object = object_names[i]

        next_object_ariaWorld_T_device = best_object_ariaWorld_T_device[next_object]
        next_object_ariaWorld_T_cpf = (
            next_object_ariaWorld_T_device * vrs_mps_streamer.device_T_cpf
        )
        if use_spot:
            # get the best object pose in spotWorld frame
            next_object_spotWorld_T_cpf = (
                avg_spotWorld_T_ariaWorld
                * next_object_ariaWorld_T_device
                * vrs_mps_streamer.device_T_cpf
            )
            vrs_mps_streamer.plot_rgb_and_trajectory(
                pose_list=[
                    avg_spotWorld_T_marker,
                    avg_spotWorld_T_ariaWorld * vrs_mps_streamer.device_T_cpf,
                    spot.get_sophus_SE3_spot_a_T_b(rf.SPOT_WORLD_VISION, rf.SPOT_BODY),
                    next_object_spotWorld_T_cpf,
                ],
                rgb=np.zeros((10, 10, 3), dtype=np.uint8),
                traj_data=spotWorld_T_device_trajectory,
                block=False,
            )
            vrs_mps_streamer.plot_rgb_and_trajectory(
                pose_list=[
                    avg_ariaWorld_T_marker,
                    avg_ariaWorld_T_spotWorld,
                    avg_ariaWorld_T_spot,
                    next_object_spotWorld_T_cpf,
                ],
                rgb=np.zeros((10, 10, 3), dtype=np.uint8),
                traj_data=vrs_mps_streamer.xyz_trajectory,
                block=True,
            )

            # Get the position and orientation of the object of interest in spotWorld frame
            pose_of_interest = next_object_spotWorld_T_cpf

            # Position is obtained from the translation component of the pose
            position = pose_of_interest.translation()

            # Find the angle made by CPF's z axis with spotWorld's x axis
            # as robot should orient to the CPF's z axis. First 3 elements of
            # column 3 from spotWorld_T_cpf represents cpf's z axis in spotWorld frame
            cpf_z_axis_in_spotWorld = pose_of_interest.matrix()[:3, 2]
            # Project cpf's z axis onto xy plane of spotWorld frame by ignoring z component
            xy_plane_projection_array = np.array([1.0, 1.0, 0.0])
            projected_cpf_z_axis_in_spotWorld_xy = np.multiply(
                cpf_z_axis_in_spotWorld, xy_plane_projection_array
            )
            orientation = float(
                np.arctan2(
                    projected_cpf_z_axis_in_spotWorld_xy[1],
                    projected_cpf_z_axis_in_spotWorld_xy[0],
                )
            )  # tan^-1(y/x)

            logger.debug(f" Going to {next_object=} at {position=}, {orientation=}")
            if not dry_run:
                skill_manager = SpotSkillManager()
                skill_manager.nav(position[0], position[1], orientation)
                skill_manager.pick(next_object)
        else:
            logger.debug(f"Showing {next_object=}")
            vrs_mps_streamer.plot_rgb_and_trajectory(
                pose_list=[
                    avg_ariaWorld_T_marker,
                    next_object_ariaWorld_T_cpf,
                ],
                rgb=best_object_img[next_object]
                if best_object_img
                else np.zeros((10, 10, 3), dtype=np.uint8),
                traj_data=vrs_mps_streamer.xyz_trajectory,
                block=True,
            )


if __name__ == "__main__":
    main()
