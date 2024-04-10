# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
from typing import Any, Dict, List, Optional, Tuple

import aria.sdk as aria
import click
import cv2
import matplotlib.pyplot as plt
import numpy as np
import projectaria_tools.core as aria_core
import rospy

try:
    import sophuspy as sp
except Exception as e:
    print(f"Cannot import sophuspy due to {e}. Import sophus instead")
    import sophus as sp
from aria_data_utils.aria_sdk_utils import update_iptables
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Odometry
from perception_and_utils.perception.detector_wrappers.april_tag_detector import (
    AprilTagDetectorWrapper,
)
from perception_and_utils.perception.detector_wrappers.object_detector import (
    ObjectDetectorWrapper,
)
from perception_and_utils.utils.conversions import (
    np_matrix3x4_to_sophus_SE3,
    ros_Pose_to_sophus_SE3,
    sophus_SE3_to_ros_Pose,
    sophus_SE3_to_ros_TransformStamped,
)
from perception_and_utils.utils.image_utils import rotate_img
from perception_and_utils.utils.math_utils import get_running_avg_a_T_b
from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    distort_by_calibration,
    get_linear_camera_calibration,
)
from projectaria_tools.core.sensor_data import ImageDataRecord
from spot_rl.utils.utils import ros_frames as rf
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import Marker

# For information on useful reference frame, please check https://github.com/facebookresearch/spot-sim2real?tab=readme-ov-file#eyeglasses-run-spot-aria-project-code


class AriaLiveReader:
    """
    Class to livestream frames and publish different data on ROS topics

    This class reads images from Aria's camera and runs process_frame method on each frame,
    which has following detectors:
        1. AprilTagDetectorWrapper: Detects QR codes in the image.
        2. ObjectDetectorWrapper: Detects objects in the image

    QR Detector
    We get a device_T_marker for each frame in which qr code is detected.
    Depending on the frame rate of image capture, multiple frames may have captured the qr code.
    Averaging all transforms would be best way to compensate for any noise that may exist in any frame's detections
    camera_T_marker is used to compute device_T_marker[i] and thus ariaWorld_T_marker[i].
    Then we average all ariaWorld_T-marker to find average marker pose wrt ariaWorld.

    Object Detector
    We use multi-class object detection models to be able to detect multiple objects in an image.
    Whem we detect an object, we store aria's pose (in cpf frame w.r.t ariaWorld) at the location.


    Args:
        verbose (bool): If true, prints debug messages

    Publishers:
        /aria_current_pose: PoseStamped msg containing current pose of aria (cpf) w.r.t ariaWorld
        /aria_pose_of_interest: PoseStamped msg containing pose of interest (cpf) w.r.t ariaWorld
        /aria_odom: Odometry msg containing odometry info (in cpf) w.r.t ariaWorld
        /aria_pose_of_interest_marker: Marker msg containing pose of interest (cpf) w.r.t ariaWorld

    Subscribers:
        /pose_dynamics: PoseStamped msg containing current pose of aria (device) w.r.t ariaWorld
        image : [NOT ROS] Image msg containing rgb image coming from Aria Streaming Client
    """

    def __init__(self, verbose: bool = False) -> None:
        super().__init__()
        self.verbose = verbose
        self._in_index = 0
        self._out_index = 0
        self._is_connected = False
        self.aria_open_loop_pose_sub = rospy.Subscriber(
            "/pose_dynamics", PoseStamped, self.on_pose_received
        )
        self.pose_of_interest_publisher = rospy.Publisher(
            "/aria_pose_of_interest", PoseStamped, queue_size=10
        )
        self.aria_odom_publisher = rospy.Publisher(
            "/aria_odom", Odometry, queue_size=10
        )
        self.aria_cpf_publisher = rospy.Publisher(
            "/aria_current_pose", PoseStamped, queue_size=10
        )
        self.pose_of_interest_marker_publisher = rospy.Publisher(
            "/aria_pose_of_interest_marker", Marker, queue_size=10
        )

        self.aria_pose = None
        self.aria_ros_pose = None
        self.aria_rgb_frame = None
        aria.set_log_level(aria.Level.Info)

        # Create StreamingClient instance
        self._sensors_calib_json = None
        self._sensors_calib: Optional[aria_core.calibration.SensorCalibration] = None
        self._device_T_cpf = None
        self._rgb_calib_params: Optional[aria_core.calibration.CameraCalibration] = None
        self._dst_calib_params: Optional[aria_core.calibration.CameraCalibration] = None
        self.device_T_camera: Optional[sp.SE3] = None
        self.device, self.device_client = self._setup_aria()

        self.static_tf_broadcaster = StaticTransformBroadcaster()

        # Create all detectors
        self.april_tag_detector = AprilTagDetectorWrapper()
        self.object_detector = ObjectDetectorWrapper()

        # Maintain a list of all poses where qr code is detected (w.r.t ariaWorld)
        self.marker_positions_list = (
            []
        )  # type: List[np.ndarray] # List of  position as np.ndarray (x, y, z)
        self.marker_quaternion_list = (
            []
        )  # type: List[np.ndarray] # List of quaternions as np.ndarray (x, y, z, w)
        self.avg_ariaWorld_T_marker = None  # type: Optional[sp.SE3]

    def connect(self):
        """
        Connect to Aria
        """
        self.device.streaming_manager.streaming_client.subscribe()  # type: ignore
        self._is_connected = True
        rospy.loginfo("Connected to Aria")

    def disconnect(self):
        """
        Unsubscribes from Aria's streaming client & disconnects from Aria
        """
        self._is_connected = False
        self.device.streaming_manager.streaming_client.unsubscribe()  # type: ignore
        self.device_client.disconnect(self.device)  # type: ignore
        rospy.loginfo("Disconnected from Aria")

    def _setup_aria(self) -> Tuple[aria.Device, aria.DeviceClient]:
        """
        Setup aria device and device client with appropriate configurations for live streaming

        Updates:
            - self._sensors_calib_json - JSON string of sensors calibration
            - self._sensors_calib - aria's SensorCalibration object
            - self._rgb_calib_params - aria's CameraCalibration params object for camera-rgb
            - self._dst_calib_params - desired CameraCalibration params with linear distortion model
            - self.device_T_camera - Sophus SE3 transform from device frame (left-slam camera) to rgb-camera frame

        Returns:
            Tuple[aria.Device, aria.DeviceClient]: Tuple of aria's device and device client objects after their proper setup
        """
        # get a device client and configure it
        device_client = aria.DeviceClient()
        client_config = aria.DeviceClientConfig()
        device_client.set_client_config(client_config)

        # create a device
        device = device_client.connect()

        # load the static device_T_cpf matrix from file, more info can be found here - https://facebookresearch.github.io/projectaria_tools/docs/data_formats/coordinate_convention/3d_coordinate_frame_convention
        this_file_path = os.path.dirname(os.path.realpath(__file__))
        self._device_T_cpf = np_matrix3x4_to_sophus_SE3(
            np_matrix=np.load(
                os.path.join(this_file_path, "../device_model/device_T_cpf.npy")
            )
        )

        # create a streaming manager to manage subscriptions
        streaming_manager = device.streaming_manager
        streaming_client: aria.StreamingClient = streaming_manager.streaming_client  # type: ignore

        # Get sensors calibration
        self._sensors_calib_json = streaming_manager.sensors_calibration()
        self._sensors_calib = device_calibration_from_json_string(
            self._sensors_calib_json
        )
        self._rgb_calib_params = self._sensors_calib.get_camera_calib("camera-rgb")  # type: ignore
        self._dst_calib_params = get_linear_camera_calibration(
            512, 512, 280, "camera-rgb"
        )
        self.device_T_camera = sp.SE3(
            self._sensors_calib.get_transform_device_sensor("camera-rgb").to_matrix()
        )  # type: ignore

        # Configure subscription to listen to Aria's RGB stream.
        # streaming configuration
        config = streaming_client.subscription_config
        options = aria.StreamingSecurityOptions()
        options.use_ephemeral_certs = False
        options.local_certs_root_path = os.path.expanduser(
            ("~/.aria/streaming-certs/persistent/")
        )
        config.security_options = options
        config.subscriber_data_type = aria.StreamingDataType.Rgb
        # A shorter queue size may be useful if the processing callback is always slow and you wish to process more recent data
        # For visualizing the images, we only need the most recent frame so set the queue size to 1
        # @TODO: Identify the best queue size for your application
        config.message_queue_size[aria.StreamingDataType.Rgb] = 1  # type: ignore
        config.message_queue_size[aria.StreamingDataType.Slam] = 1  # type: ignore
        streaming_client.subscription_config = config

        streaming_client.set_streaming_client_observer(self)

        return device, device_client

    def on_image_received(self, image: np.ndarray, record: ImageDataRecord):
        """
        Callback for aria's streaming client object.
        Called internally by aria's streaming client everytime a new image is available from aria

        Args:
            image (np.ndarray): Image received from Aria
            record (ImageDataRecord): Image metadata received from Aria

        Updates:
            - self.aria_rgb_frame: aria's raw rgb image without any rectification
        """
        if self._is_connected:
            rospy.logdebug("Received image")
            self.aria_rgb_frame = image

    def on_pose_received(self, msg: PoseStamped):
        """
        ROS callback for aria pose in device frame

        Args:
            msg (PoseStamped): Pose received from Aria

        Updates:
            - self.aria_pose_device - ariaWorld_T_device as Sophus SE3
            - self.aria_pose - ariaWorld_T_cpf as Sophus SE3
            - self.aria_ros_pose - ariaWorld_T_cpf as ROS Pose

        Publishes:
            - /aria_current_pose: ROS PoseStamped message for ariaWorld_T_cpf
            - /aria_odom: ROS Odometry message for ariaWorld_T_cpf (for visualizing aria trajectory in rviz)
        """
        if self._is_connected:
            rospy.logdebug("Received pose")
            self.aria_pose_device = ros_Pose_to_sophus_SE3(msg.pose)
            self.aria_pose = self.aria_pose_device * self._device_T_cpf
            self.aria_ros_pose = sophus_SE3_to_ros_Pose(sp_se3=self.aria_pose)

            # Publish pose as odometry message for visualizing entire aria trajectory in rviz
            odom_msg = Odometry()
            odom_msg.header.frame_id = rf.ARIA_WORLD
            # odom_msg.child_frame_id = "ariaDevice"
            odom_msg.child_frame_id = rf.ARIA
            odom_msg.header.stamp = rospy.Time.now()
            odom_msg.pose.pose = self.aria_ros_pose
            self.aria_odom_publisher.publish(odom_msg)

            # Publish aria pose in cpf frame wrt ariaWorld
            msg.pose = self.aria_ros_pose
            self.aria_cpf_publisher.publish(msg)

    def publish_pose_of_interest(self, pose: Pose, marker_scale: float = 0.1):
        """
        Publishes current pose of Aria as a pose of interest for Spot

        Args:
            pose (Pose): Pose of interest
            marker_scale (float, optional): Scale of marker. Defaults to 0.1

        Updates:
            - self._out_index: Sequence number / ID of the pose of interest published

        Publishes:
            - /aria_pose_of_interest: ROS PoseStamped message for ariaWorld_T_cpf at pose of interest i.e. pose where it detected the object
            - /aria_pose_of_interest_marker: ROS Marker message for ariaWorld_T_cpf at pose of interest i.e. pose where it detected the object
        """
        # Publish as pose for detail
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time().now()
        pose_stamped.header.seq = self._out_index
        pose_stamped.header.frame_id = rf.ARIA_WORLD
        pose_stamped.pose = pose

        self.pose_of_interest_publisher.publish(pose_stamped)

        # Publish as marker for interpretability
        marker = Marker()
        marker.header.stamp = rospy.Time().now()
        marker.header.frame_id = rf.ARIA_WORLD
        marker.id = self._out_index
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = pose
        marker.scale.x = marker_scale
        marker.scale.y = marker_scale
        marker.scale.z = marker_scale
        marker.color.a = 1.0
        marker.color.r = 0.45
        marker.color.g = 0.95
        marker.color.b = 0.2

        self.pose_of_interest_marker_publisher.publish(marker)

        self._out_index += 1

    def get_latest_pose_and_image(self) -> Optional[Dict[str, Any]]:
        """
        Compose a frame dict with latest information from Aria

        A frame is a dict with keys:
            - 'raw_image': aria's raw rgb image with linear rectification
            - 'image': aria's rgb image with linear rectification and necessary rotation
            - 'pose': current pose of the aria in CPF frame i.e. ariaWorld_T_cpf as Sophus SE3
            - 'device_pose': current pose of the aria in device frame i.e. ariaWorld_T_device as Sophus SE3
            - 'ros_pose': current pose of the aria in CPF frame i.e. ariaWorld_T_cpf as ROS Pose
            - 'timestamp': the ros timestamp when frame dict was created (i.e. when this function was called)

        Returns:
            latest_frame (dict): Latest frame dict from Aria containing the above keys
        """
        if self.aria_rgb_frame is None:
            return None
        if self.aria_pose is None:
            return None
        aria_rect_rgb = self.rectify_aria_image(self.aria_rgb_frame)  # type: ignore

        # Rotate current rgb image from aria by 270 degrees in counterclockwise direction as
        # all aria's images are 90 degrees rotated (in anticlockwise direction).
        # Alternatively, you can also rotate the rgb image by 90 degrees in clockwise direction (num_of_rotation=-1)
        aria_rot_rect_rgb = rotate_img(aria_rect_rgb, num_of_rotation=3)

        # Compose frame dict for most recent frame
        latest_frame = {
            "raw_image": aria_rect_rgb,
            "image": aria_rot_rect_rgb,
            "pose": self.aria_pose,
            "device_pose": self.aria_pose_device,
            "ros_pose": self.aria_ros_pose,
            "timestamp": rospy.Time.now(),
        }

        return latest_frame

    def rectify_aria_image(self, image: np.ndarray) -> np.ndarray:
        """
        Rectify fisheye image based upon camera calibration parameters
        Ensure you have set self._src_calib_param & self._dst_calib_param

        Args:
            image (np.ndarray): Image to be rectified or undistorted

        Returns:
            np.ndarray: Rectified image
        """
        assert self._rgb_calib_params is not None and self._dst_calib_params is not None
        aria_rect_rgb = distort_by_calibration(
            image, self._dst_calib_params, self._rgb_calib_params
        )  # type: ignore
        return aria_rect_rgb

    def process_frame(
        self,
        frame: dict,
        outputs: dict,
        detect_qr: bool = False,
        detect_objects: bool = False,
        object_label="milk bottle",
    ) -> Tuple[np.ndarray, dict]:
        """
        Process the frame and return the visualization image  along with a dict of detections

        Detection types supported:
            - April tag
            - Object detection with OwlVIT

        April tag outputs:
            - "tag_base_T_marker_list" - List of Sophus SE3 transforms from base frame to marker
                                         where base is "device" frame for aria
            - "tag_image_list" - List of np.ndarrays of images with detections
            - "tag_image_metadata_list" - List of image metadata

        Object detection outputs:
            - "object_image_list" - List of np.ndarrays of images with detections
            - "object_image_metadata_list" - List of image metadata
            - "object_image_segment" - List of Int signifying which segment the image
                belongs to; smaller number means latter the segment time-wise
            - "object_score_list" - List of Float signifying the detection score

        Args:
            frame (dict): Frame dict from Aria containing the keys:
                - 'raw_image': aria's raw rgb image with linear rectification
                - 'image': aria's rgb image with linear rectification and necessary rotation
                - 'pose': current pose of the aria in CPF frame i.e. ariaWorld_T_cpf as Sophus SE3
                - 'device_pose': current pose of the aria in device frame i.e. ariaWorld_T_device as Sophus SE3
                - 'ros_pose': current pose of the aria in CPF frame i.e. ariaWorld_T_cpf as ROS Pose
                - 'timestamp': the ros timestamp when frame dict was created (i.e. when this function was called)
            outputs (dict): Dictionary of outputs from the april tag detector
            detect_qr (bool, optional): Whether to detect april tag. Defaults to False.
            detect_objects (bool, optional): Whether to detect objects. Defaults to False.
            object_label (str, optional): Object label to detect. Defaults to "milk bottle".

        Publishes:
            - static tf: ROS TransformStamped message for marker frame wrt ariaWorld
            - /aria_pose_of_interest: ROS PoseStamped message for ariaWorld_T_cpf at pose of interest i.e. pose where it detected the object
            - /aria_pose_of_interest_marker: ROS Marker message for ariaWorld_T_cpf at pose of interest i.e. pose where it detected the object

        Returns:
            Tuple[np.ndarray, dict]: Tuple of:
                - viz_img (np.ndarray): Image with april tag and object in hand detections
                - outputs (dict): Dictionary of outputs from the april tag detector
        """
        # Initialize camera_T_marker to None & object_scores to empty dict for current image frame
        camera_T_marker = None
        object_scores = {}  # type: Dict[str, Any]

        viz_img = frame.get("image")
        if detect_qr:
            (viz_img, camera_T_marker) = self.april_tag_detector.process_frame(img_frame=frame.get("raw_image"))  # type: ignore
            # Rotate current image frame
            viz_img = rotate_img(img=viz_img, num_of_rotation=3)

        if camera_T_marker is not None:
            device_T_marker = self.device_T_camera * camera_T_marker
            ariaWorld_T_device = frame.get("device_pose")

            # Frame: a = ariaWorld
            # Frame: intermediate = device (aria's device / left slam camera frame)
            # Frame: b = marker / qr code
            (
                self.marker_positions_list,
                self.marker_quaternion_list,
                self.avg_ariaWorld_T_marker,
            ) = get_running_avg_a_T_b(
                current_avg_a_T_b=self.avg_ariaWorld_T_marker,
                a_T_b_position_list=self.marker_positions_list,
                a_T_b_quaternion_list=self.marker_quaternion_list,
                a_T_intermediate=ariaWorld_T_device,
                intermediate_T_b=device_T_marker,
            )

            if self.avg_ariaWorld_T_marker is not None:
                avg_marker_T_ariaWorld = self.avg_ariaWorld_T_marker.inverse()

                # Publish marker pose in ariaWorld frame
                self.static_tf_broadcaster.sendTransform(
                    sophus_SE3_to_ros_TransformStamped(
                        sp_se3=avg_marker_T_ariaWorld,
                        parent_frame=rf.MARKER,
                        child_frame=rf.ARIA_WORLD,
                    )
                )

            viz_img, outputs = self.april_tag_detector.get_outputs(
                img_frame=viz_img,
                outputs=outputs,
                base_T_marker=device_T_marker,
                timestamp=frame.get("timestamp"),
                img_metadata=None,
            )

        if detect_objects:
            viz_img, object_scores = self.object_detector.process_frame(viz_img)
            if object_label in object_scores.keys():
                if self.verbose:
                    plt.imsave(f"frame_{self._in_index}.jpg", viz_img)
                self.publish_pose_of_interest(frame.get("ros_pose"))
        self._in_index += 1
        return viz_img, outputs

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
            principal_point_obj[0].item(),
            principal_point_obj[1].item(),
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


@click.command()
@click.option("--do-update-iptables", is_flag=True, type=bool, default=False)
@click.option("--read-only", is_flag=True, type=bool, default=False)
@click.option("--debug", is_flag=True, type=bool, default=False)
@click.option("--hz", type=int, default=5)
def main(do_update_iptables: bool, read_only: bool, debug: bool, hz: int):
    """
    Main function initializes the AriaLiveReader object, its detectors and runs detectors on every frame while publishing
    necessary data on different ROS topics

    Detection types supported:
        - April tag
        - Object detection with OwlVIT

    Args:
        do_update_iptables (bool, optional): Whether to update iptables to allow for streaming. Defaults to False.
        read_only (bool, optional): Whether to only read frames and not publish pose of interest. Defaults to False.
        debug (bool, optional): Whether to run in debug mode. Defaults to False.
    """
    if debug:
        _log_level = rospy.DEBUG
    else:
        _log_level = rospy.INFO
    rospy.init_node("aria_live_reader", log_level=_log_level)
    rospy.logwarn("Starting up ROS node")
    if do_update_iptables:
        update_iptables()
    else:
        rospy.logwarn("Not updating iptables")
    object_name = "bottle"

    rate = rospy.Rate(hz)
    outputs: Dict[str, Any] = {}
    if not read_only:
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    outputs = {}
    aria_object = AriaLiveReader(verbose=True)
    outputs = aria_object.initialize_april_tag_detector(outputs=outputs)
    outputs = aria_object.initialize_object_detector(
        outputs=outputs, object_labels=[object_name]
    )
    aria_object.connect()
    while not rospy.is_shutdown():
        frame = aria_object.get_latest_pose_and_image()
        if frame is not None:
            rospy.loginfo(f"Got frame: {frame.keys()}")
            if not read_only:
                image, outputs = aria_object.process_frame(
                    frame,
                    outputs=outputs,
                    detect_qr=True,
                    detect_objects=True,
                    object_label=object_name,
                )

                cv2.imshow("Image", image[:, :, ::-1])
                cv2.waitKey(1)
        rate.sleep()
    aria_object.disconnect()
    rospy.logwarn("Disconnecting and shutting down the node")


if __name__ == "__main__":
    main()
