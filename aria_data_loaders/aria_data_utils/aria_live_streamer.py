import aria.sdk as aria
import matplotlib.pyplot as plt
import numpy as np
import rospy
from common import quit_keypress, update_iptables
from geometry_msgs.msg import PoseStamped
from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    distort_by_calibration,
    get_linear_camera_calibration,
)
from projectaria_tools.core.sensor_data import ImageDataRecord


class AriaLiveReader:
    """
    Class to livestream frames and pose data from Aria
    """

    def __init__(self) -> None:
        super().__init__()
        self.aria_pose_sub = rospy.Subscriber(
            "/aria_pose", PoseStamped, self.on_pose_received
        )  # in marker frame
        self.aria_pose = None
        self.aria_rgb_frame = None
        self.streaming_interface = "usb"
        self.profile_name = "profile9"
        aria.set_log_level(aria.Level.Info)
        # 1. Create StreamingClient instance
        self.device, self.device_client = self._setup_aria()
        self.device_client.streaming_client.subscribe()

    def _setup_aria(self) -> aria.DeviceClient:
        device_client = aria.DeviceClient()
        device = device_client.connect()
        streaming_manager = device.streaming_manager
        streaming_client = streaming_manager.streaming_client
        config = streaming_client.subscription_config
        config.subscriber_data_type = aria.StreamingDataType.Rgb
        # A shorter queue size may be useful if the processing callback is always slow and you wish to process more recent data
        # For visualizing the images, we only need the most recent frame so set the queue size to 1
        # @TODO: Identify the best queue size for your application
        config.message_queue_size[aria.StreamingDataType.Rgb] = 1
        config.message_queue_size[aria.StreamingDataType.Slam] = 1

        # Set the security options
        # @note we need to specify the use of ephemeral certs as this sample app assumes
        # aria-cli was started using the --use-ephemeral-certs flag
        # TODO: Identify if we need to use ephemeral certs or permanent certs
        options = aria.StreamingSecurityOptions()
        options.use_ephemeral_certs = True
        config.security_options = options
        streaming_client.subscription_config = config
        streaming_client.set_streaming_client_observer(self)
        streaming_config = aria.StreamingConfig()
        streaming_config.profile_name = self.profile_name
        # Note: by default streaming uses Wifi
        if self.streaming_interface == "usb":
            streaming_config.streaming_interface = aria.StreamingInterface.Usb
        streaming_manager.streaming_config = streaming_config

        # 5. Get sensors calibration
        self._sensors_calib_json = streaming_manager.sensors_calibration()
        self._sensors_calib = device_calibration_from_json_string(
            self._sensors_calib_json
        )
        self._rgb_calib = self._sensors_calib.get_camera_calib("camera-rgb")
        self._dst_calib = get_linear_camera_calibration(512, 512, 150, "camera-rgb")

        # 6. Start streaming
        streaming_manager.start_streaming()

        # 7. Configure subscription to listen to Aria's RGB stream.
        config = streaming_client.subscription_config
        config.subscriber_data_type = aria.StreamingDataType.Rgb
        streaming_client.subscription_config = config
        return device, device_client

    def on_image_received(self, image: np.array, record: ImageDataRecord):
        self.aria_rgb_frame = image

    def on_pose_received(self, pose: np.array, record: ImageDataRecord):
        pass

    def get_frame(self):
        """
        Get the latest frame from Aria
        frame is a dict with keys:
            - 'img': the rgb image
            - 'pose': the pose of the aria device in CPF frame
            - 'timestamp': the timestamp of the frame
        """
        pass

    def disconnect(self):
        """
        Disconnect from Aria
        """
        self.device_client.streaming_client.unsubscribe()
        self.device_client.streaming_manager.stop_streaming()
        self.device_client.disconnect(self.device)


def main():
    rospy.init_node("aria_live_reader")
    rate = rospy.Rate(5)
    aria_object = AriaLiveReader()
    while not rospy.is_shutdown():
        aria_object.get_frame()
        if aria_object.aria_rgb_frame is not None:
            plt.imshow(aria_object.aria_rgb_frame.astype(np.float32) / 255.0)
            plt.show()
        print(aria_object.aria_rgb_frame)
        rate.sleep()

    aria_object.disconnect()


if __name__ == "__main__":
    main()
