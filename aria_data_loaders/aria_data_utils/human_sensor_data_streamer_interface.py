# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
from typing import Any, Dict, List, Optional, Tuple

import click
import cv2
import numpy as np
import rospy

try:
    import sophuspy as sp
except Exception as e:
    print(f"Cannot import sophuspy due to {e}. Import sophus instead")
    import sophus as sp

# ROS imports
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Odometry
from perception_and_utils.perception.detector_wrappers.april_tag_detector import (
    AprilTagDetectorWrapper,
)
from perception_and_utils.perception.detector_wrappers.human_motion_detector import (
    HumanMotionDetector,
)
from perception_and_utils.perception.detector_wrappers.object_detector import (
    ObjectDetectorWrapper,
)
from perception_and_utils.utils.data_frame import DataFrame
from spot_rl.utils.utils import ros_frames as rf
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import Marker


class CameraParams:
    """
    CameraParams class to store camera parameters like focal length, principal point etc.
    It stores the image, timestamp and frame number of each received image
    """

    def __init__(
        self, focal_lengths: Tuple[float, float], principal_point: Tuple[float, float]
    ):
        self._focal_lengths = focal_lengths  # type: Optional[Tuple[float, float]]
        self._principal_point = principal_point  # type: Optional[Tuple[float, float]]


class HumanSensorDataStreamerInterface:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

        self._is_connected = False  # TODO : Maybe not needed

        # TODO: Update this later with PerceptionPipeline
        # Create all detectors
        self.april_tag_detector = AprilTagDetectorWrapper()
        self.object_detector = ObjectDetectorWrapper()
        self.human_motion_detector = HumanMotionDetector()

        # ROS publishers & broadcaster
        self.static_tf_broadcaster = StaticTransformBroadcaster()

        self.human_activity_history_pub = rospy.Publisher(
            "/human_activity_history", String, queue_size=10
        )

        # TODO: Define DEVICE FRAME as a visual frame of reference i.e. front facing frame of reference
        self.device_T_camera: Optional[
            sp.SE3
        ] = None  # TODO: Should be initialized by implementations of this class

        # Maintain a list of all poses where qr code is detected (w.r.t deviceWorld)
        self.marker_positions_list: List[
            np.ndarray
        ] = []  # List of  position as np.ndarray (x, y, z)
        self.marker_quaternion_list: List[
            np.ndarray
        ] = []  # List of quaternions as np.ndarray (x, y, z, w)
        self.avg_deviceWorld_T_marker: Optional[sp.SE3] = None

    def connect(self):
        """
        Connect to Device
        """
        raise NotImplementedError

    def disconnect(self):
        """
        Disconnects from Device cleanly
        """
        raise NotImplementedError

    def _setup_device(self) -> Any:
        """
        Setup device with appropriate configurations for live streaming

        Updates:
            - self.device_T_camera - Sophus SE3 transform from device frame (left-slam camera) to rgb-camera frame

        Returns:
            Any : TODO: Decide what to return
        """
        raise NotImplementedError

    def get_latest_data_frame(self) -> Optional[DataFrame]:
        raise NotImplementedError

    def process_frame(
        self,
        frame: dict,
        outputs: dict,
        detect_qr: bool = False,
        detect_objects: bool = False,
        object_label="milk bottle",
    ) -> Tuple[np.ndarray, dict]:
        raise NotImplementedError

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
        raise NotImplementedError

    def initialize_object_detector(
        self, outputs: dict = {}, object_labels: list = [], meta_objects: List[str] = []
    ):
        """
        Initialize the object detector

        Args:
            outputs (dict, optional): Dictionary of outputs from the object detector. Defaults to {}.
            object_labels (list, optional): List of object labels to detect. Defaults to [].
            meta_objects (List[str], optional): List of other objects to be detected in the image
                                                apart from the object_labels; meta_objects are then
                                                used internally to detect intersection of bbox with
                                                objects of interest (eg: hand with bottle)

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
        raise NotImplementedError

    def initialize_human_motion_detector(self, outputs: dict = {}):
        """ """
        raise NotImplementedError

    ##### ROS Publishers #####
    # def publish_human_pose(self, data_frame: DataFrame):
    #     """
    #     Publishes current pose of Device as a pose of interest for Spot

    #     Args:
    #         data_frame (DataFrame): DataFrame object containing data packet with rgb, depth
    #                                 and all necessary transformations

    #     Publishes:
    #         - /human_pose_publisher: ROS PoseStamped message for quest3World_T_rgb # TODO: Update this from rgb to VIZ frame
    #     """
    #     # Publish as pose for detail
    #     pose_stamped = PoseStamped()
    #     pose_stamped.header.stamp = rospy.Time().now()
    #     pose_stamped.header.seq = data_frame._frame_number
    #     pose_stamped.header.frame_id = rf.QUEST3_CAMERA
    #     pose_stamped.pose = (
    #         data_frame._deviceWorld_T_camera_rgb
    #     )  # TODO: Use a VIZ frame which is forward facing
    #     self.human_pose_publisher.publish(pose_stamped)

    # def publish_marker(self, data_frame: DataFrame, marker_scale: float = 0.1):
    #     """
    #     Publishes marker at pose of interest

    #     Args:
    #         data_frame (DataFrame): DataFrame object containing data packet with rgb, depth
    #                                 and all necessary transformations

    #     Publishes:
    #         - /pose_of_interest_marker_publisher: ROS PoseStamped message for quest3World_T_rgb # TODO: Update this from rgb to VIZ frame
    #     """
    #     # Publish as marker for interpretability
    #     marker = Marker()
    #     marker.header.stamp = rospy.Time().now()
    #     marker.header.frame_id = rf.rf.QUEST3_CAMERA
    #     marker.id = self._frame_number

    #     marker.type = Marker.SPHERE
    #     marker.action = Marker.ADD

    #     marker.pose = data_frame._deviceWorld_T_camera_rgb

    #     marker.scale.x = marker_scale
    #     marker.scale.y = marker_scale
    #     marker.scale.z = marker_scale

    #     marker.color.a = 1.0
    #     marker.color.r = 0.45
    #     marker.color.g = 0.95
    #     marker.color.b = 0.2

    #     self.pose_of_interest_marker_publisher.publish(marker)

    def publish_human_activity_history(self, human_history: List[Tuple[float, str]]):
        """
        Publishes human activity history as a string

        Args:
            data_frame (DataFrame): DataFrame object containing data packet with rgb, depth
                                    and all necessary transformations

        Publishes:
            - /human_activity_history: ROS String message for human activity history
        """

        # Iterate over human history and create a string
        human_history_str = ""
        for timestamp, activity in human_history:
            human_history_str += (
                f"{timestamp},{activity}|"  # str = "time1,Standing|time2,Walking"
            )

        # Publish human activity history string
        self.human_activity_history_pub.publish(human_history_str)
