import os
from typing import Any, Dict, List, Optional, Tuple

import aria.sdk as aria
import click
import cv2
import matplotlib.pyplot as plt
import numpy as np
import projectaria_tools.core as aria_core
import rospy
import sophus as sp
from aria_data_utils.aria_sdk_utils import update_iptables
from aria_data_utils.conversions import (
    compute_avg_spSE3_from_nplist,
    generate_TransformStamped_a_T_b_from_spSE3,
    matrix3x4_to_sophus,
    ros_pose_to_sophus,
    sophus_to_ros_pose,
)
from aria_data_utils.detector_wrappers.april_tag_detector import AprilTagDetectorWrapper
from aria_data_utils.detector_wrappers.object_detector import ObjectDetectorWrapper
from cairo import Device
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Odometry
from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    distort_by_calibration,
    get_linear_camera_calibration,
)
from projectaria_tools.core.sensor_data import ImageDataRecord
from pytest import mark
from scipy.spatial.transform import Rotation
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import Marker

FILTER_DIST = 2.4  # in meters (distance for valid detection)


class AriaLiveReader:
    """
    Class to livestream frames and pose data from Aria
    """

    def __init__(self, verbose: bool = False) -> None:
        super().__init__()
        self.verbose = verbose
        self._in_index = 0
        self._out_index = 0
        self._is_connected = False
        self.aria_pose_sub = rospy.Subscriber(
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

        # 1. Create StreamingClient instance
        self._latest_frame: Dict[str, Any] = None
        self._sensors_calib_json = None
        self._sensors_calib: Optional[aria_core.calibration.SensorCalibration] = None
        self._device_T_cpf = None
        self._rgb_calib_params: Optional[aria_core.calibration.CameraCalibration] = None
        self._dst_calib_params: Optional[aria_core.calibration.CameraCalibration] = None
        self.device_T_camera: Optional[sp.SE3] = None
        self.device, self.device_client = self._setup_aria()

        self.static_tf_broadcaster = StaticTransformBroadcaster()

        # Maintain a list of all poses where qr code is detected (w.r.t ariaWorld)
        self.marker_positions_list = (
            []
        )  # type: List[np.ndarray] # List of  position as np.ndarray (x, y, z)
        self.marker_quaternion_list = (
            []
        )  # type: List[np.ndarray] # List of quaternions as np.ndarray (x, y, z, w)
        self.fix_list_length = 10  # TODO: Add this to config
        self.avg_ariaWorld_T_marker = None  # type: Optional[sp.SE3]

    def connect(self):
        """
        Connect to Aria
        """
        self.device.streaming_manager.streaming_client.subscribe()  # type: ignore
        self._is_connected = True
        rospy.loginfo("Connected to Aria")

    def _setup_aria(self) -> Tuple[aria.Device, aria.DeviceClient]:
        # get a device client and configure it
        device_client = aria.DeviceClient()
        client_config = aria.DeviceClientConfig()
        device_client.set_client_config(client_config)

        # create a device
        device = device_client.connect()

        # get the path to this file
        this_file_path = os.path.dirname(os.path.realpath(__file__))

        self._device_T_cpf = matrix3x4_to_sophus(
            np.load(os.path.join(this_file_path, "../device_model/device_T_cpf.npy"))
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

    def on_image_received(self, image: np.ndarray, record: ImageDataRecord):
        if self._is_connected:
            rospy.logdebug("Received image")
            self.aria_rgb_frame = image

    def on_pose_received(self, msg: PoseStamped):
        if self._is_connected:
            rospy.logdebug("Received pose")
            self.aria_pose_device = ros_pose_to_sophus(msg.pose)
            self.aria_pose = self.aria_pose_device * self._device_T_cpf
            self.aria_ros_pose = sophus_to_ros_pose(self.aria_pose)
            odom_msg = Odometry()
            odom_msg.header.frame_id = "ariaWorld"
            odom_msg.child_frame_id = "ariaDevice"
            odom_msg.header.stamp = rospy.Time.now()
            odom_msg.pose.pose = self.aria_ros_pose
            self.aria_odom_publisher.publish(odom_msg)
            msg.pose = self.aria_ros_pose
            self.aria_cpf_publisher.publish(msg)

    def publish_pose_of_interest(self, pose: Pose, marker_scale: float = 0.1):
        """
        Publishes current pose of Aria as a pose of interest for Spot
        """

        # publish as pose for detail
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time().now()
        pose_stamped.header.seq = self._out_index
        pose_stamped.header.frame_id = "ariaWorld"
        pose_stamped.pose = pose

        self.pose_of_interest_publisher.publish(pose_stamped)

        # publish as marker for interpretability
        marker = Marker()
        marker.header.stamp = rospy.Time().now()
        marker.header.frame_id = "ariaWorld"
        marker.id = self._out_index
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = pose
        marker.scale.x = marker_scale
        marker.scale.y = marker_scale
        marker.scale.z = marker_scale
        marker.color.a = 1.0
        marker.color.r = 0.5
        marker.color.g = 0.5
        marker.color.b = 0.0

        self.pose_of_interest_marker_publisher.publish(marker)

        self._out_index += 1

    def get_frame(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest frame from Aria
        frame is a dict with keys:
            - 'image': the rgb image
            - 'pose': the pose of the aria device in CPF frame
            - 'timestamp': the timestamp of the frame
        """
        self.update_frame()
        return self._latest_frame

    def update_frame(self):
        if self.aria_rgb_frame is None:
            return None
        if self.aria_pose is None:
            return None
        aria_rect_rgb = self._rectify_image(self.aria_rgb_frame)  # type: ignore
        aria_rot_rect_rgb = self._rotate_img(aria_rect_rgb)
        self._latest_frame = {
            "raw_image": aria_rect_rgb,
            "image": aria_rot_rect_rgb,
            "pose": self.aria_pose,
            "device_pose": self.aria_pose_device,
            "ros_pose": self.aria_ros_pose,
            "timestamp": rospy.Time.now(),
        }

    def disconnect(self):
        """
        Disconnect from Aria
        """
        self._is_connected = False
        self.device.streaming_manager.streaming_client.unsubscribe()  # type: ignore
        self.device_client.disconnect(self.device)  # type: ignore

    def _rectify_image(self, image: np.ndarray) -> np.ndarray:
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
        detect_hand_object: bool = False,
        object_label="milk bottle",
    ) -> Tuple[np.ndarray, dict]:
        """
        Process the frame
        """
        # Initialize camera_T_marker to None & object_scores to empty dict for current image frame
        camera_T_marker = None
        object_scores = {}

        viz_img = frame.get("image")
        if detect_qr:
            (viz_img, camera_T_marker) = self.april_tag_detector.process_frame(img_frame=frame.get("raw_image"))  # type: ignore
            # Rotate current image frame
            viz_img = self._rotate_img(img=viz_img)
            # TODO: @kavitshah logic for averaging april tag pose

        if camera_T_marker is not None:
            device_T_marker = self.device_T_camera * camera_T_marker
            ariaWorld_T_device = frame.get("device_pose")

            avg_ariaWorld_T_marker = self.get_running_avg_ariaWorld_T_marker(
                ariaWorld_T_device=ariaWorld_T_device, device_T_marker=device_T_marker
            )

            if avg_ariaWorld_T_marker is not None:
                avg_marker_T_ariaWorld = avg_ariaWorld_T_marker.inverse()

                # Publish marker pose in ariaWorld frame
                self.static_tf_broadcaster.sendTransform(
                    generate_TransformStamped_a_T_b_from_spSE3(
                        avg_marker_T_ariaWorld,
                        parent_frame="marker",
                        child_frame="ariaWorld",
                    )
                )

            viz_img, outputs = self.april_tag_detector.get_outputs(
                img_frame=viz_img,
                outputs=outputs,
                device_T_marker=device_T_marker,
                timestamp=frame.get("timestamp"),
                img_metadata=None,
            )

        if detect_hand_object:
            output, object_scores = self.object_detector.process_frame_online(
                np.copy(frame.get("image"))
            )
            if object_label in object_scores.keys():
                if self.verbose:
                    plt.imsave(f"frame_{self._in_index}.jpg", output)
                self.publish_pose_of_interest(frame.get("ros_pose"))
        self._in_index += 1
        return viz_img, outputs

    def initialize_april_tag_detector(self, outputs: dict = {}):
        """
        Initialize the april tag detector
        """
        self.april_tag_detector = AprilTagDetectorWrapper()
        focal_lengths = self._dst_calib_params.get_focal_lengths()  # type:ignore
        principal_point = self._dst_calib_params.get_principal_point()  # type: ignore
        self.april_tag_detector.enable_detector()

        outputs.update(
            self.april_tag_detector._init_april_tag_detector(
                focal_lengths=focal_lengths, principal_point=principal_point
            )
        )
        return outputs

    def initialize_object_detector(self, outputs: dict = {}, object_labels: list = []):
        """
        Initialize the object in hand detector
        """
        self.object_detector = ObjectDetectorWrapper()
        self.object_detector.enable_detector()
        meta_objects: List[str] = []
        outputs.update(
            self.object_detector._init_object_detector(
                object_labels + meta_objects,
                verbose=self.verbose,
                version=2,
            )
        )
        self.object_detector._core_objects = object_labels
        self.object_detector._meta_objects = meta_objects

    def get_running_avg_ariaWorld_T_marker(
        self,
        ariaWorld_T_device: sp.SE3,
        device_T_marker: sp.SE3,
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
        ariaWorld_T_marker = ariaWorld_T_device * device_T_marker
        marker_position = ariaWorld_T_marker.translation()
        device_position = ariaWorld_T_device.translation()
        delta = marker_position - device_position
        dist = np.linalg.norm(delta)

        # Consider only those detections where detected marker is within a certain distance of the camera
        if dist < filter_dist:
            # If the number of detections exceeds the fix list length, remove the first element
            if len(self.marker_positions_list) >= self.fix_list_length:
                self.marker_positions_list.pop(0)
                self.marker_quaternion_list.pop(0)

            self.marker_positions_list.append(marker_position)
            quat = Rotation.from_matrix(ariaWorld_T_marker.rotationMatrix()).as_quat()

            # Ensure quaternion's w is always positive for effective averaging as multiple quaternions can represent the same rotation
            if quat[3] > 0:
                quat = -1.0 * quat
            self.marker_quaternion_list.append(quat)

            # Compute the average transformation as new data got appended
            self.avg_ariaWorld_T_marker = compute_avg_spSE3_from_nplist(
                a_T_b_position_list=self.marker_positions_list,
                a_T_b_quaternion_list=self.marker_quaternion_list,
            )

        return self.avg_ariaWorld_T_marker


@click.command()
@click.option("--do-update-iptables", is_flag=True, type=bool, default=False)
@click.option("--read-only", is_flag=True, type=bool, default=False)
@click.option("--debug", is_flag=True, type=bool, default=False)
def main(do_update_iptables: bool, read_only: bool, debug: bool):
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

    rate = rospy.Rate(5)
    outputs: Dict[str, Any] = {}
    if not read_only:
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    aria_object = AriaLiveReader(verbose=True)
    aria_object.initialize_april_tag_detector(outputs=outputs)
    aria_object.initialize_object_detector(object_labels=[object_name])
    aria_object.connect()
    while not rospy.is_shutdown():
        frame = aria_object.get_frame()
        if frame is not None:
            rospy.loginfo(f"Got frame: {frame.keys()}")
            if not read_only:
                image, outputs = aria_object.process_frame(
                    frame,
                    outputs=outputs,
                    detect_qr=True,
                    detect_hand_object=True,
                    object_label=object_name,
                )

                cv2.imshow("Image", image[:, :, ::-1])
                cv2.waitKey(1)
        rate.sleep()
    aria_object.disconnect()
    rospy.logwarn("Disconnecting and shutting down the node")


if __name__ == "__main__":
    main()
