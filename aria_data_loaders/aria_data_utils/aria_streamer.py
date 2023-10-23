import os
import time
from typing import Any, Dict, List, Optional, Tuple

import click
import cv2
import numpy as np
import sophus as sp
from aria_data_utils.image_utils import centered_heuristic, check_bbox_intersection
from aria_data_utils.utils.april_tag_pose_estimator import AprilTagPoseEstimator
from fairotag.scene import Scene
from matplotlib import pyplot as plt
from projectaria_tools.core import calibration, data_provider, mps
from scipy.spatial.transform import Rotation as R
from spot_rl.envs.skill_manager import SpotSkillManager
from spot_rl.models.owlvit import OwlVit
from spot_wrapper.spot import Spot, SpotCamIds, image_response_to_cv2

"""
dict = {
    "file" : "Start_2m.vrs",
    "2m" : 245,
    "3m" : 334,
    "4m" : 463,
    "5m" : 574,
}

dict = {
    "file" : "Start_1halfm.vrs",
    "2m" : 238,
    "3m" : 348,
    "4m" : 458,
    "5m" : 598,
}

dict = {
    "file" : "Start_2m.vrs",
    "2m" : 237,
    "3m" : 337,
    "4m" : 457,
    "5m" : 581,
}
"""

DOCK_ID = int(os.environ.get("SPOT_DOCK_ID", 520))
# TODO: Clean up the following
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
    """
    Helper method to label image with text

    Args:
        img (np.ndarray): Image to be labeled
        text (str): Text to be labeled
        org (tuple): (x,y) position of text
        font_face (int, optional): Font face. Defaults to cv2.FONT_HERSHEY_SIMPLEX.
        font_scale (float, optional): Font scale. Defaults to 0.8.
        color (tuple, optional): Color of text. Defaults to (0, 0, 255).
        thickness (int, optional): Thickness of text. Defaults to 2.
        line_type (int, optional): Line type. Defaults to cv2.LINE_AA.
    """
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


# @TODO: Maybe make position as Any?
def decorate_img_with_text(img: np.ndarray, frame: str, position: np.ndarray):
    """
    Helper method to label image with text

    Args:
        img (np.ndarray): Image to be labeled
        frame (str): Frame of reference (for labeling)
        position (np.ndarray): Position of object in frame of reference
    """
    label_img(img, "Detected QR Marker", (50, 50), color=(0, 0, 255))
    label_img(img, f"Frame = {frame}", (50, 75), color=(0, 0, 255))
    label_img(img, f"X : {position[0]}", (50, 100), color=(0, 0, 255))
    label_img(img, f"Y : {position[1]}", (50, 125), color=(0, 250, 0))
    label_img(img, f"Z : {position[2]}", (50, 150), color=(250, 0, 0))

    return img


# DEBUG FUNCTION. REMOVE LATER
def take_snapshot(spot: Spot):
    resp_head = spot.experiment(is_hand=False)
    cv2_image_head_r = image_response_to_cv2(resp_head, reorient=True)

    resp_hand = spot.experiment(is_hand=True)
    cv2_image_hand = image_response_to_cv2(resp_hand, reorient=False)
    cv2.imwrite("test_head_right_rgb1.jpg", cv2_image_head_r)
    cv2.imwrite("test_hand_rgb1.jpg", cv2_image_hand)


######################################################################


class AriaReader:
    def __init__(self, vrs_file_path: str, mps_file_path: str, verbose=False):
        assert vrs_file_path is not None and os.path.exists(
            vrs_file_path
        ), "Incorrect VRS file path"
        assert mps_file_path is not None and os.path.exists(
            mps_file_path
        ), "Incorrect MPS dir path"

        # TODO: Maybe rename this flag as we also have a config.VERBOSE flag (for logging) .. to avoid confusion
        self.verbose = verbose

        self.provider = data_provider.create_vrs_data_provider(vrs_file_path)
        assert self.provider is not None, "Cannot open VRS file"

        self.device_calib = self.provider.get_device_calibration()

        # TODO: Condition check to ensure stream name is valid

        # April tag detector object
        self._qr_pose_estimator = None  # type: ignore

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
        self.ariaCorrectedWorld_T_device_trajectory = []  # type: List[Any]
        self.ariaCorrectedWorld_T_cpf_trajectory = []  # type: List[Any]

        # Setup some generic transforms
        self.device_T_cpf = sp.SE3(
            self.device_calib.get_transform_device_cpf().to_matrix()
        )
        self.cpf_T_device = self.device_T_cpf.inverse()

        # Initialize Trajectory after setting up cpf transforms
        self.initialize_trajectory()

        # sensor_calib_list = [device_calib.get_sensor_calib(label) for label in stream_names][0]
        # Record VRS timestamps of interest based upon user input during vrs parsing
        self.vrs_idx_of_interest_list = []  # type: List[Any]

    def plot_rgb_and_trajectory(
        self,
        marker_pose: sp.SE3,
        device_pose_list: List[sp.SE3],
        rgb: np.ndarray,
        traj_data: np.ndarray = None,
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
        scene.add_frame("correctedWorld", pose=self.device_T_cpf)
        scene.add_camera(
            "dock", frame="correctedWorld", pose_in_frame=marker_pose, size=AXES_SCALE
        )
        for i in range(len(device_pose_list)):
            scene.add_camera(
                f"device_{i}",
                frame="correctedWorld",
                pose_in_frame=device_pose_list[i],
                size=AXES_SCALE,
            )

        plt_ax = scene.visualize(fig=fig, should_return=True)
        plt_ax.plot(traj_data[:, 0], traj_data[:, 1], traj_data[:, 2])
        plt.show()

    def _init_april_tag_detector(self) -> Dict[str, Any]:
        """
        Initialize April tag detector object for pose estimation of QR code

        Can only detect dock code
        """
        focal_lengths = self._dst_calib_params.get_focal_lengths()  # type:ignore
        principal_point = self._dst_calib_params.get_principal_point()  # type:ignore
        calib_dict = {
            "fx": focal_lengths[0].item(),
            "fy": focal_lengths[1].item(),
            "ppx": principal_point[0].item(),
            "ppy": principal_point[1].item(),
            "coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        }

        self._qr_pose_estimator = AprilTagPoseEstimator(camera_intrinsics=calib_dict)
        self._qr_pose_estimator.register_marker_ids([DOCK_ID])  # type:ignore
        outputs: Dict[str, Any] = {}
        outputs["tag_cpf_T_marker_list"] = []
        outputs["tag_image_list"] = []
        outputs["tag_image_metadata_list"] = []
        return outputs

    def _init_object_detector(
        self, object_labels: List[str], verbose: bool = None
    ) -> Dict[str, Any]:
        """initialize object_detector by loading it to GPU and setting up the labels"""
        if verbose is None:
            verbose = self.verbose
        self.object_detector = OwlVit(
            [object_labels], score_threshold=0.125, show_img=verbose
        )
        outputs: Dict[str, Any] = {}
        outputs["object_image_list"] = {obj_name: [] for obj_name in object_labels}
        outputs["object_image_metadata_list"] = {
            obj_name: [] for obj_name in object_labels
        }
        outputs["object_image_segment_list"] = {
            obj_name: [] for obj_name in object_labels
        }
        outputs["object_score_list"] = {obj_name: [] for obj_name in object_labels}
        outputs["object_ariaCorrectedWorld_T_cpf_list"] = {
            obj_name: [] for obj_name in object_labels
        }
        return outputs

    def _rotate_img(self, img: np.ndarray, num_of_rotation: int = 3) -> np.ndarray:
        """
        Rotate image in multiples of 90d degrees

        Args:
            img (np.ndarray): Image to be rotated
            k (int, optional): Number of times to rotate by 90 degrees. Defaults to 3.

        Returns:
            np.ndarray: Rotated image
        """
        img = np.ascontiguousarray(
            np.rot90(img, k=num_of_rotation)
        )  # GOD KNOW WHY THIS IS NEEDED -> https://github.com/clovaai/CRAFT-pytorch/issues/84#issuecomment-574683857
        return img

    def _display(self, img: np.ndarray, stream_name: str, wait: int = 1):
        """
        Display image in a cv2 window

        Args:
            img (np.ndarray): Image to be displayed
            stream_name (str): Stream name
            wait (int, optional): Wait time in ms. Defaults to 1 ms
        """
        cv2.imshow(f"Stream - {stream_name}", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cv2.waitKey(wait)

    def _create_display_window(self, stream_name: str):
        """
        Create a display window for image

        Args:
            stream_name (str): Stream name
        """
        cv2.namedWindow(f"Stream - {stream_name}", cv2.WINDOW_NORMAL)

    def _rectify_image(self, image) -> np.ndarray:
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

    def _get_scored_object_detections(
        self, img: np.ndarray, heuristic=None
    ) -> Tuple[bool, Dict[str, float], Dict[str, bool], np.ndarray]:
        """
        Detect object instances in the frame. score them based on heuristics
        and return a tuple of (valid, score, result_image)
        Input:
        - img: np.ndarray: image to run inference on
        - heuristic: function: function to score the detections
        Output:
        - valid: bool: whether the frame is valid or not
        - score: float: score of the frame (based on :heuristic:)
        - result_image: np.ndarray: image with bounding boxes drawn on it
        """
        detections, result_image = self.object_detector.run_inference_and_return_img(
            img
        )
        if heuristic is not None:
            valid, score, stop = heuristic(detections)

        return valid, score, stop, result_image

    def _p1_demo_heuristics(
        self, detections, img_size=(512, 512)
    ) -> Tuple[bool, Dict[str, bool], Dict[str, float]]:
        """*P1 Demo Specific*
        Heuristics to filter out unwanted and bad obejct detections. This heuristic
        scores each frame based on (a) % of pixels occupied by object in the frame
        and (b) proximity of bbox to image center. Further only those images are
        scored which have only the object detection without a hand in the frame.

        Detections are expected to have the format:
        [["object_name", "confidence", "bbox"]]
        """
        valid = False
        score = {}
        stop = {}
        # figure out if this is an interaction frame
        if len(detections) > 1:
            objects_in_frame = [det[0] for det in detections]
            # FIXME: only works for 1 object of interest right now, extend to multiple
            if "hand" in objects_in_frame and "water bottle" in objects_in_frame:
                print(f"checking for intersection b/w: {object}")
                stop["water bottle"] = check_bbox_intersection(
                    detections[objects_in_frame.index("water bottle")][2],
                    detections[objects_in_frame.index("hand")][2],
                )
                print(f"Intersection: {stop}")
        if len(detections) == 1:
            if "water bottle" == detections[0][0]:
                score["water bottle"] = centered_heuristic(detections)[0]
                valid = True

        return valid, score, stop

    def get_vrs_timestamp_from_img_idx(
        self, stream_name: str = STREAM1_NAME, idx_of_interest: int = -1
    ):
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

    def parse_camera_stream(
        self,
        stream_name: str,
        should_rectify: bool = True,
        detect_qr: bool = False,
        should_display: bool = True,
        should_rec_timestamp_from_user: bool = False,
        detect_objects: bool = False,
        object_labels: List[str] = None,
        reverse: bool = False,
    ) -> Dict[str, Any]:
        """Parse linearly through a camera stream and return a dict of detections
        Detection types supported:
        - April Tag
        - Object Detection

        Args:
            stream_name (str): Stream name
            should_rectify (bool, optional): Boolean to indicate if fisheye images should be rectified. Defaults to True.
            detect_qr (bool, optional): Boolean to indicate if QR code (Dock ID) should be detected. Defaults to False.
            should_display (bool, optional): Boolean to indicate if image should be displayed. Defaults to True.
            should_rec_timestamp_from_user (bool, optional): Boolean to iteratively parse through VRS stream and record
                                                             instances of interest. Defaults to False.
            detect_objects (bool, optional): Boolean to indicate if object detection should be performed. Defaults to False.
            object_labels (List[str], optional): List of object labels to be detected. Defaults to None.
            reverse (bool, optional): Boolean to indicate if VRS stream should be parsed in reverse. Defaults to False.
                                      Useful based on the algorithm to detect objects from VRS stream.

        April tag outputs:
        - "tag_image_list" - List of np.ndarrays of images with detections
        - "tag_image_metadata_list" - List of image metadata
        - "tag_cpf_T_marker_list" - List of Sophus SE3 transforms from CPF to marker
           CPF is the center frame of Aria with Z pointing out, X pointing up
           and Y pointing left

        Object detection outputs:
        - "object_image_list" - List of np.ndarrays of images with detections
        - "object_image_metadata_list" - List of image metadata
        - "object_image_segment" - List of Int signifying which segment the image
            belongs to; smaller number means latter the segment time-wise
        - "object_score_list" - List of Float signifying the detection score
        - "object_ariaCorrectedWorld_T_cpf_list" - List of Sophus SE3 transforms from AriaWorld
          to CPF
        """
        # Get stream id from stream name
        stream_id = self.provider.get_stream_id_from_label(stream_name)

        # Get device_T_camera i.e. transformation from camera frame of interest to device frame (i.e. left slam camera frame)
        device_T_camera = sp.SE3(
            self.device_calib.get_transform_device_sensor(stream_name).to_matrix()
        )
        assert device_T_camera is not None

        # Setup camera calibration parameters by over-writing self._src_calib_param & self._dst_calib_param if needed
        if should_rectify:
            self._src_calib_params = self.device_calib.get_camera_calib(stream_name)
            self._dst_calib_params = calibration.get_linear_camera_calibration(
                512, 512, 280, stream_name
            )

        outputs = {}

        # Setup April tag detection by over-writing self._qr_pose_estimator if needed
        if detect_qr:
            outputs.update(self._init_april_tag_detector())

        # Setup object detection (Owl-VIT) if needed
        if detect_objects:
            outputs.update(self._init_object_detector(object_labels))

        if should_display:
            self._create_display_window(stream_name)

        num_frames = self.provider.get_num_data(stream_id)
        # TODO: make this range updatable: min_frame_idx, max_frame_idx as args
        if reverse:
            custom_range = range(num_frames - 1, 0, -1)
        else:
            custom_range = range(0, num_frames)
        for frame_idx in custom_range:
            # Get image data for frame
            frame_data = self.provider.get_image_data_by_index(stream_id, frame_idx)
            img = frame_data[0].to_numpy_array()
            img_metadata = frame_data[1]

            # Initialize camera_T_marker to None for current image frame
            camera_T_marker = None

            # Rectify current image frame
            if should_rectify:
                img = self._rectify_image(image=img)

            # Detect QR code in current image frame
            if detect_qr:
                (
                    img,
                    camera_T_marker,
                ) = self._qr_pose_estimator.detect_markers_and_estimate_pose(  # type: ignore
                    image=img, should_render=True, magnum=False
                )

            # Rotate current image frame
            img = self._rotate_img(img=img)

            # TODO: @Priyam, can you add comment for this block please?
            if detect_objects:
                # FIXME: done for 1 specific object at the moment, extend for multiple
                valid, score, stop, result_img = self._get_scored_object_detections(
                    img, heuristic=self._p1_demo_heuristics
                )
                if stop and stop["water bottle"]:
                    print("Turning off object-detection")
                    detect_objects = False
                if valid:
                    score_string = str(score["water bottle"]).replace(".", "_")
                    plt.imsave(
                        f"./results/{frame_idx}_{valid}_{score_string}.jpg", result_img
                    )
                    print(f"saving valid frame, {score=}")
                    for object_name in score.keys():
                        outputs["object_image_list"][object_name].append(img)
                        outputs["object_score_list"][object_name].append(
                            score[object_name]
                        )
                        outputs["object_ariaCorrectedWorld_T_cpf_list"][
                            object_name
                        ].append(
                            self.get_closest_ariaCorrectedWorld_T_cpf_to_timestamp(
                                img_metadata.capture_timestamp_ns
                            )
                        )
                        # TODO: following is not being used at the moment, clean-up if
                        # multi-object, multi-view logic seems to not require this
                        outputs["object_image_segment_list"][object_name].append(-1)
                        outputs["object_image_metadata_list"][object_name].append(
                            img_metadata
                        )

            # If april tag is detected, compute the transformation of marker in cpf frame
            # TODO: Update cpf to device
            if camera_T_marker is not None:
                cpf_T_marker = self.cpf_T_device * device_T_camera * camera_T_marker

                # Decorate image with text for visualization
                img = decorate_img_with_text(
                    img=img,
                    frame="cpf",
                    position=cpf_T_marker.translation(),
                )
                print(
                    f"Time stamp with Detections- {img_metadata.capture_timestamp_ns}"
                )

                # Append data to lists for return
                outputs["tag_image_list"].append(img)
                outputs["tag_image_metadata_list"].append(img_metadata)
                outputs["tag_cpf_T_marker_list"].append(cpf_T_marker)

            # Display current image frame
            if should_display:
                self._display(img=img, stream_name=stream_name)

            # Record instances of interest (press 'c' to record, 'o' to skip. Will ask input at each frame)
            # TODO: Maybe we can delete this?
            if should_rec_timestamp_from_user:
                val = input("o to skip, c to record")
                if val == "c":
                    self.vrs_idx_of_interest_list.append(frame_idx)

        return outputs

    def initialize_trajectory(self):
        # frame(ariaWorld) is same as frame(device) at the start
        cpf_T_ariaWorld = self.cpf_T_device

        for i in range(len(self.mps_trajectory)):
            self.trajectory_s[i] = self.mps_trajectory[
                i
            ].tracking_timestamp.total_seconds()
            ariaWorld_T_device = sp.SE3(
                self.mps_trajectory[i].transform_world_device.to_matrix()
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

    # TODO: Can be optimized. This is making O(n^2)
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

    def get_avg_ariaCorrectedWorld_T_marker(
        self,
        img_list: List,
        img_metadata_list: List,
        cpf_T_marker_list: List,
        filter_dist: float = 2.4,
        should_plot: bool = False,
    ) -> sp.SE3:
        marker_position_list = []
        marker_quaternion_list = []
        # X = [] # dist(marker_avg, marker[i])
        # Y = [] # dist[i]

        # For viz
        ariaCorrectedWorld_T_cpf_list = []
        ariaCorrectedWorld_T_marker_list = []
        img_viz_list = []
        for img, img_metadata, cpf_T_marker in zip(
            img_list, img_metadata_list, cpf_T_marker_list
        ):
            vrs_timestamp_of_interest_ns = (
                img_metadata.capture_timestamp_ns
            )  # maybe this can be replaced
            ariaCorrectedWorld_T_cpf = (
                self.get_closest_ariaCorrectedWorld_T_cpf_to_timestamp(
                    vrs_timestamp_of_interest_ns
                )
            )
            ariaCorrectedWorld_T_marker = ariaCorrectedWorld_T_cpf * cpf_T_marker

            marker_position = ariaCorrectedWorld_T_marker.translation()
            device_position = ariaCorrectedWorld_T_cpf.translation()
            delta = marker_position - device_position
            dist = np.linalg.norm(delta)
            # Y.append(dist)

            # Consider only those detections where detected marker is within a certain distance of the camera
            if dist < filter_dist:
                marker_position_list.append(marker_position)
                quat = R.from_matrix(
                    ariaCorrectedWorld_T_marker.so3().matrix()
                ).as_quat()
                marker_quaternion_list.append(quat)

                if should_plot:
                    ariaCorrectedWorld_T_cpf_list.append(ariaCorrectedWorld_T_cpf)
                    ariaCorrectedWorld_T_marker_list.append(ariaCorrectedWorld_T_marker)
                    img_viz_list.append(img)

        marker_position_np = np.array(marker_position_list)
        avg_marker_position = np.mean(marker_position_np, axis=0)

        marker_quaternion_np = np.array(marker_quaternion_list)
        avg_marker_quaternion = np.mean(marker_quaternion_np, axis=0)

        avg_ariaCorrectedWorld_T_marker = sp.SE3(
            R.from_quat(avg_marker_quaternion).as_matrix(), avg_marker_position
        )

        if should_plot:
            assert len(marker_position_list) == len(ariaCorrectedWorld_T_marker_list)

            for i in range(len(marker_position_list)):
                self.plot_rgb_and_trajectory(
                    marker_pose=ariaCorrectedWorld_T_marker_list[i],
                    device_pose_list=[
                        ariaCorrectedWorld_T_cpf_list[i],
                        avg_ariaCorrectedWorld_T_marker,
                    ],
                    rgb=img_viz_list[i],
                    traj_data=self.xyz_trajectory,
                )

        # Get a plot for last location of Aria
        # plt_ax = self.plot_rgb_and_trajectory(
        #             marker_pose=ariaCorrectedWorld_T_marker_list[i],
        #             device_pose_list=[ariaCorrectedWorld_T_cpf_list[i],avg_ariaCorrectedWorld_T_marker],
        #             rgb=img_viz_list[i],
        #             traj_data=self.xyz_trajectory,
        #         )
        # X = [np.linalg.norm(avg_marker_position - marker_position) for marker_position in marker_position_l]
        # plt.scatter(X,Y)
        # plt.xlabel("euclid_dist(avg_marker_position, marker_postition[i])")
        # plt.ylabel("dist[i]")
        # plt.show()

        return avg_ariaCorrectedWorld_T_marker


class SpotQRDetector:
    def __init__(self, spot: Spot):
        self.spot = spot
        print("Done init")

    def _to_camera_metadata_dict(self, camera_intrinsics):
        """Converts a camera intrinsics proto to a 3x3 matrix as np.array"""
        intrinsics = {
            "fx": camera_intrinsics.focal_length.x,
            "fy": camera_intrinsics.focal_length.x,
            "ppx": camera_intrinsics.principal_point.x,
            "ppy": camera_intrinsics.principal_point.y,
            "coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
        return intrinsics

    def _get_body_T_handcam(self, frame_tree_snapshot_hand):
        # print(frame_tree_snapshot_hand)
        hand_bd_wrist_T_handcam_dict = (
            frame_tree_snapshot_hand.child_to_parent_edge_map.get(
                "hand_color_image_sensor"
            ).parent_tform_child
        )
        hand_mn_wrist_T_handcam = self.spot.convert_transformation_from_BD_to_magnun(
            hand_bd_wrist_T_handcam_dict
        )

        hand_bd_body_T_wrist_dict = (
            frame_tree_snapshot_hand.child_to_parent_edge_map.get(
                "arm0.link_wr1"
            ).parent_tform_child
        )
        hand_mn_body_T_wrist = self.spot.convert_transformation_from_BD_to_magnun(
            hand_bd_body_T_wrist_dict
        )

        hand_mn_body_T_handcam = hand_mn_body_T_wrist @ hand_mn_wrist_T_handcam

        return hand_mn_body_T_handcam

    def _get_body_T_headcam(self, frame_tree_snapshot_head):
        # print(frame_tree_snapshot_head)
        head_bd_fr_T_frfe_dict = frame_tree_snapshot_head.child_to_parent_edge_map.get(
            "frontright_fisheye"
        ).parent_tform_child
        head_mn_fr_T_frfe_dict = self.spot.convert_transformation_from_BD_to_magnun(
            head_bd_fr_T_frfe_dict
        )

        head_bd_head_T_fr_dict = frame_tree_snapshot_head.child_to_parent_edge_map.get(
            "frontright"
        ).parent_tform_child
        head_mn_head_T_fr = self.spot.convert_transformation_from_BD_to_magnun(
            head_bd_head_T_fr_dict
        )

        head_bd_body_T_head_dict = (
            frame_tree_snapshot_head.child_to_parent_edge_map.get(
                "head"
            ).parent_tform_child
        )
        head_mn_body_T_head = self.spot.convert_transformation_from_BD_to_magnun(
            head_bd_body_T_head_dict
        )

        head_mn_head_T_frfe = head_mn_head_T_fr @ head_mn_fr_T_frfe_dict
        head_mn_body_T_frfe = head_mn_body_T_head @ head_mn_head_T_frfe

        return head_mn_body_T_frfe

    def _get_spotWorld_T_handcam(
        self, frame_tree_snapshot_hand, use_vision_as_world: bool = True
    ):

        spot_world_frame = "vision" if use_vision_as_world else "odom"

        # print(frame_tree_snapshot_hand)
        hand_bd_wrist_T_handcam_dict = (
            frame_tree_snapshot_hand.child_to_parent_edge_map.get(
                "hand_color_image_sensor"
            ).parent_tform_child
        )
        hand_mn_wrist_T_handcam = self.spot.convert_transformation_from_BD_to_magnun(
            hand_bd_wrist_T_handcam_dict
        )

        hand_bd_body_T_wrist_dict = (
            frame_tree_snapshot_hand.child_to_parent_edge_map.get(
                "arm0.link_wr1"
            ).parent_tform_child
        )
        hand_mn_body_T_wrist = self.spot.convert_transformation_from_BD_to_magnun(
            hand_bd_body_T_wrist_dict
        )

        hand_bd_body_T_spotWorld_dict = (
            frame_tree_snapshot_hand.child_to_parent_edge_map.get(
                spot_world_frame
            ).parent_tform_child
        )
        hand_mn_body_T_spotWorld = self.spot.convert_transformation_from_BD_to_magnun(
            hand_bd_body_T_spotWorld_dict
        )
        hand_mn_spotWorld_T_body = hand_mn_body_T_spotWorld.inverted()

        hand_mn_body_T_handcam = hand_mn_body_T_wrist @ hand_mn_wrist_T_handcam
        hand_mn_spotWorld_T_handcam = hand_mn_spotWorld_T_body @ hand_mn_body_T_handcam

        return hand_mn_spotWorld_T_handcam

    def _get_spotWorld_T_headcam(
        self, frame_tree_snapshot_head, use_vision_as_world: bool = True
    ):

        spot_world_frame = "vision" if use_vision_as_world else "odom"

        # print(frame_tree_snapshot_head)
        head_bd_fr_T_frfe_dict = frame_tree_snapshot_head.child_to_parent_edge_map.get(
            "frontright_fisheye"
        ).parent_tform_child
        head_mn_fr_T_frfe_dict = self.spot.convert_transformation_from_BD_to_magnun(
            head_bd_fr_T_frfe_dict
        )

        head_bd_head_T_fr_dict = frame_tree_snapshot_head.child_to_parent_edge_map.get(
            "frontright"
        ).parent_tform_child
        head_mn_head_T_fr = self.spot.convert_transformation_from_BD_to_magnun(
            head_bd_head_T_fr_dict
        )

        head_bd_body_T_head_dict = (
            frame_tree_snapshot_head.child_to_parent_edge_map.get(
                "head"
            ).parent_tform_child
        )
        head_mn_body_T_head = self.spot.convert_transformation_from_BD_to_magnun(
            head_bd_body_T_head_dict
        )

        head_bd_body_T_spotWorld_dict = (
            frame_tree_snapshot_head.child_to_parent_edge_map.get(
                spot_world_frame
            ).parent_tform_child
        )
        head_mn_body_T_spotWorld = self.spot.convert_transformation_from_BD_to_magnun(
            head_bd_body_T_spotWorld_dict
        )
        head_mn_spotWorld_T_body = head_mn_body_T_spotWorld.inverted()

        head_mn_head_T_frfe = head_mn_head_T_fr @ head_mn_fr_T_frfe_dict
        head_mn_body_T_frfe = head_mn_body_T_head @ head_mn_head_T_frfe
        head_mn_spotWorld_T_frfe = head_mn_spotWorld_T_body @ head_mn_body_T_frfe

        return head_mn_spotWorld_T_frfe

    def get_avg_spotWorld_T_marker_HAND(
        self,
        use_vision_as_world: bool = True,
        data_size_for_avg: int = 10,
        filter_dist: float = 2.2,
    ):
        print("Withing function")
        cv2.namedWindow("hand_image", cv2.WINDOW_AUTOSIZE)
        spot_world_frame = "vision" if use_vision_as_world else "odom"

        # Get Hand camera intrinsics
        hand_cam_intrinsics = self.spot.get_camera_intrinsics(SpotCamIds.HAND_COLOR)
        hand_cam_intrinsics = self._to_camera_metadata_dict(hand_cam_intrinsics)
        hand_cam_pose_estimator = AprilTagPoseEstimator(hand_cam_intrinsics)

        # Register marker ids
        marker_ids_list = [DOCK_ID]
        # marker_ids_list = [i for i in range(521, 550)]
        hand_cam_pose_estimator.register_marker_ids(marker_ids_list)

        marker_position_from_dock_list = []  # type: List[Any]
        marker_quaternion_form_dock_list = []  # type: List[Any]

        marker_position_from_robot_list = []  # type: List[Any]
        marker_quaternion_form_robot_list = []  # type: List[Any]

        while len(marker_position_from_dock_list) < data_size_for_avg:
            print(f"Iterating - {len(marker_position_from_dock_list)}")
            is_marker_detected_from_hand_cam = False
            img_response_hand = self.spot.get_hand_image()
            img_hand = image_response_to_cv2(img_response_hand)

            (
                img_rend_hand,
                hand_mn_handcam_T_marker,
            ) = hand_cam_pose_estimator.detect_markers_and_estimate_pose(
                img_hand, should_render=True
            )

            if hand_mn_handcam_T_marker is not None:
                print("Trackedddd")
                is_marker_detected_from_hand_cam = True

            # Spot - spotWorld_T_handcam computation
            frame_tree_snapshot_hand = img_response_hand.shot.transforms_snapshot
            hand_mn_body_T_handcam = self._get_body_T_handcam(frame_tree_snapshot_hand)
            hand_mn_spotWorld_T_handcam = self._get_spotWorld_T_handcam(
                frame_tree_snapshot_hand, use_vision_as_world=use_vision_as_world
            )

            if is_marker_detected_from_hand_cam:
                hand_mn_spotWorld_T_marker = (
                    hand_mn_spotWorld_T_handcam @ hand_mn_handcam_T_marker
                )

                hand_mn_body_T_marker = (
                    hand_mn_body_T_handcam @ hand_mn_handcam_T_marker
                )

                img_rend_hand = decorate_img_with_text(
                    img_rend_hand,
                    spot_world_frame,
                    hand_mn_spotWorld_T_marker.translation,
                )

                dist = hand_mn_handcam_T_marker.translation.length()

                print(
                    f"Dist = {dist}, Recordings - {len(marker_position_from_dock_list)}"
                )
                if dist < filter_dist:
                    marker_position_from_dock_list.append(
                        np.array(hand_mn_spotWorld_T_marker.translation)
                    )
                    marker_quaternion_form_dock_list.append(
                        R.from_matrix(hand_mn_spotWorld_T_marker.rotation()).as_quat()
                    )
                    marker_position_from_robot_list.append(
                        np.array(hand_mn_body_T_marker.translation)
                    )
                    marker_quaternion_form_robot_list.append(
                        R.from_matrix(hand_mn_body_T_marker.rotation()).as_quat()
                    )

            cv2.imshow("hand_image", img_rend_hand)
            cv2.waitKey(1)

        marker_position_from_dock_np = np.array(marker_position_from_dock_list)
        avg_marker_position_from_dock = np.mean(marker_position_from_dock_np, axis=0)

        marker_quaternion_from_dock_np = np.array(marker_quaternion_form_dock_list)
        avg_marker_quaternion_from_dock = np.mean(
            marker_quaternion_from_dock_np, axis=0
        )

        marker_position_from_robot_np = np.array(marker_position_from_robot_list)
        avg_marker_position_from_robot = np.mean(marker_position_from_robot_np, axis=0)

        marker_quaternion_from_robot_np = np.array(marker_quaternion_form_robot_list)
        avg_marker_quaternion_from_robot = np.mean(
            marker_quaternion_from_robot_np, axis=0
        )

        avg_spotWorld_T_marker = sp.SE3(
            R.from_quat(avg_marker_quaternion_from_dock).as_matrix(),
            avg_marker_position_from_dock,
        )
        avg_spot_T_marker = sp.SE3(
            R.from_quat(avg_marker_quaternion_from_robot).as_matrix(),
            avg_marker_position_from_robot,
        )

        return avg_spotWorld_T_marker, avg_spot_T_marker

    def get_avg_spotWorld_T_marker_HEAD(
        self,
        use_vision_as_world: bool = True,
        data_size_for_avg: int = 10,
        filter_dist: float = 2.4,
    ):
        pass

    def get_avg_spotWorld_T_marker(
        self,
        camera: str = "hand",
        use_vision_as_world: bool = True,
        data_size_for_avg: int = 10,
        filter_dist: float = 2.4,
    ):
        pass


@click.command()
@click.option("--data-path", help="Path to the data directory", type=str)
@click.option("--vrs-name", help="Name of the vrs file", type=str)
@click.option("--dry-run", type=bool, default=False)
@click.option("--verbose", type=bool, default=True)
def main(data_path: str, vrs_name: str, dry_run: bool, verbose: bool):
    vrsfile = os.path.join(data_path, vrs_name + ".vrs")
    vrs_mps_streamer = AriaReader(
        vrs_file_path=vrsfile, mps_file_path=data_path, verbose=verbose
    )

    outputs = vrs_mps_streamer.parse_camera_stream(
        stream_name=STREAM1_NAME,
        detect_qr=True,
        should_rec_timestamp_from_user=False,
        detect_objects=True,
        object_labels=["water bottle", "person", "hand"],
        reverse=True,
    )
    tag_img_list = outputs["tag_image_list"]
    tag_img_metadata_list = outputs["tag_image_metadata_list"]
    tag_cpf_T_marker_list = outputs["tag_cpf_T_marker_list"]

    avg_ariaCorrectedWorld_T_marker = (
        vrs_mps_streamer.get_avg_ariaCorrectedWorld_T_marker(
            tag_img_list,
            tag_img_metadata_list,
            tag_cpf_T_marker_list,
            filter_dist=2.4,
            should_plot=False,
        )
    )
    print(avg_ariaCorrectedWorld_T_marker)

    spot = Spot("ArmKeyboardTeleop")
    spot_qr = SpotQRDetector(spot=spot)
    (
        avg_spotWorld_T_marker,
        avg_spot_T_marker,
    ) = spot_qr.get_avg_spotWorld_T_marker_HAND()

    print(avg_spotWorld_T_marker)

    avg_marker_T_ariaCorrectedWorld = avg_ariaCorrectedWorld_T_marker.inverse()
    avg_spotWorld_T_ariaCorrectedWorld = (
        avg_spotWorld_T_marker * avg_marker_T_ariaCorrectedWorld
    )
    avg_ariaCorrectedWorld_T_spotWorld = avg_spotWorld_T_ariaCorrectedWorld.inverse()

    avg_spot_T_ariaCorrectedWorld = avg_spot_T_marker * avg_marker_T_ariaCorrectedWorld
    avg_ariaCorrectedWorld_T_spot = avg_spot_T_ariaCorrectedWorld.inverse()

    # find best CPF location, wrt AriaWorld, for best scored object detection
    # FIXME: extend to multiple objects
    best_idx = outputs["object_score_list"]["water bottle"].index(
        max(outputs["object_score_list"]["water bottle"])
    )
    best_object_ariaCorrectedWorld_T_cpf = outputs[
        "object_ariaCorrectedWorld_T_cpf_list"
    ]["water bottle"][best_idx]

    vrs_mps_streamer.plot_rgb_and_trajectory(
        marker_pose=avg_ariaCorrectedWorld_T_marker,
        device_pose_list=[
            # vrs_mps_streamer.ariaCorrectedWorld_T_cpf_trajectory[-2500],
            avg_ariaCorrectedWorld_T_spotWorld,
            avg_ariaCorrectedWorld_T_spot,
            best_object_ariaCorrectedWorld_T_cpf,
        ],
        rgb=outputs["object_image_list"]["water bottle"][best_idx],
        traj_data=vrs_mps_streamer.xyz_trajectory,
    )

    # TODO: make idx_of_interest as a dynamic variable (will come from object detection)
    # vrs_timestamp_of_interest = outputs["object_image_metadata_list"]["water bottle"][
    # best_idx
    # ].capture_timestamp_ns
    # vrs_timestamp_of_interest = vrs_mps_streamer.get_vrs_timestamp_from_img_idx(
    #     stream_name=STREAM1_NAME, idx_of_interest=574
    # )
    # mps_idx_of_interest = vrs_mps_streamer.get_closest_mps_idx_to_timestamp_ns(
    #     vrs_timestamp_of_interest
    # )
    # mps_idx_of_interest = -2500

    pose_of_interest = (
        avg_spotWorld_T_ariaCorrectedWorld * best_object_ariaCorrectedWorld_T_cpf
    )
    position = pose_of_interest.translation()

    if not dry_run:
        skill_manager = SpotSkillManager()
        skill_manager.nav(position[0], position[1], 0.0)


# TODO: Record raw and rectified camera params for each camera in a config file

if __name__ == "__main__":
    main()
