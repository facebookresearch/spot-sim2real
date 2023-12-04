import os
from typing import Any, Dict, Optional, Tuple

import aria.sdk as aria
import click
import matplotlib.pyplot as plt
import numpy as np
import projectaria_tools.core as aria_core
import rospy
from aria_data_utils.aria_sdk_utils import update_iptables
from aria_data_utils.conversions import ros_pose_to_sophus
from aria_data_utils.detector_wrappers.april_tag_detector import AprilTagDetectorWrapper
from aria_data_utils.detector_wrappers.object_in_hand_detector import (
    ObjectInHandDetectorWrapper,
)
from cairo import Device
from geometry_msgs.msg import PoseStamped
from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    distort_by_calibration,
    get_linear_camera_calibration,
)
from projectaria_tools.core.sensor_data import ImageDataRecord


class AriaLiveReader(aria.StreamingClientObserver):
    """
    Class to livestream frames and pose data from Aria
    """

    def __init__(self) -> None:
        super().__init__()
        self.aria_pose_sub = rospy.Subscriber(
            "/pose_dynamics", PoseStamped, self.on_pose_received
        )
        self.aria_pose = None
        self.aria_rgb_frame = None
        aria.set_log_level(aria.Level.Info)

        # 1. Create StreamingClient instance
        self._latest_frame: Dict[str, Any] = {}
        self._sensors_calib_json = None
        self._sensors_calib = None
        self._rgb_calib_params: Optional[aria_core.calibration.CameraCalibration] = None
        self._dst_calib_params: Optional[aria_core.calibration.CameraCalibration] = None
        self.device, self.device_client = self._setup_aria()

    def connect(self):
        """
        Connect to Aria
        """
        self.device.streaming_manager.streaming_client.subscribe()  # type: ignore

    def _setup_aria(self) -> Tuple[aria.Device, aria.DeviceClient]:
        # get a device client and configure it
        device_client = aria.DeviceClient()
        client_config = aria.DeviceClientConfig()
        device_client.set_client_config(client_config)

        # create a device
        device = device_client.connect()

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
        rospy.loginfo("Received image")
        self.aria_rgb_frame = image

    def on_pose_received(self, msg: PoseStamped):
        rospy.loginfo("Received pose")
        self.aria_pose = ros_pose_to_sophus(msg.pose)

    def get_frame(self) -> Optional[dict]:
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
        aria_rect_rgb = distort_by_calibration(
            self.aria_rgb_frame, self._dst_calib_params, self._rgb_calib_params
        )  # type: ignore
        aria_rect_rgb = np.rot90(aria_rect_rgb, -1)
        self._latest_frame = {
            "image": aria_rect_rgb,
            "pose": self.aria_pose,
            "timestamp": rospy.Time.now(),
        }

    def disconnect(self):
        """
        Disconnect from Aria
        """
        self.device.streaming_manager.streaming_client.unsubscribe()  # type: ignore
        self.device_client.disconnect(self.device)  # type: ignore

    def process_frame(
        self, frame: dict, detect_qr: bool = True, detect_hand_object: bool = True
    ) -> dict:
        """
        Process the frame
        """
        output = {}
        if detect_qr:
            output.update(self.april_tag_detector.process_frame(frame))

        if detect_hand_object:
            output.update(self.object_in_hand_detector.process_frame(frame))

        return output

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

    def initialize_object_in_hand_detector(self, outputs: dict = {}):
        """
        Initialize the object in hand detector
        """
        self.object_in_hand_detector = ObjectInHandDetectorWrapper()
        # focal_lengths = self._dst_calib_params.get_focal_lengths()
        return outputs


@click.command()
@click.option("--do-update-iptables", is_flag=True, type=bool, default=False)
def main(do_update_iptables: bool):
    rospy.init_node("aria_live_reader")
    if do_update_iptables:
        update_iptables()
    else:
        rospy.logwarn("Not updating iptables")

    rate = rospy.Rate(5)
    aria_object = AriaLiveReader()
    aria_object.connect()
    # aria_object.initialize_april_tag_detector()
    # aria_object.initialize_object_in_hand_detector()
    while not rospy.is_shutdown():
        frame = aria_object.get_frame()
        if frame is not None:
            _ = aria_object.process_frame(frame)
        rate.sleep()
    aria.disconnect()


if __name__ == "__main__":
    main()
