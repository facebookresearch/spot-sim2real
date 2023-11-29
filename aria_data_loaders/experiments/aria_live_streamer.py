import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import rospy
from geometry_msgs.msg import PoseStamped
import click
import cv2
import numpy as np
import sophus as sp
from aria_data_utils.detector_wrappers.april_tag_detector import AprilTagDetectorWrapper
from aria_data_utils.detector_wrappers.object_detector import ObjectDetectorWrapper
from aria_data_utils.image_utils import decorate_img_with_text
from aria_data_utils.perception.april_tag_pose_estimator import AprilTagPoseEstimator
from bosdyn.client.frame_helpers import get_a_tform_b
from fairotag.scene import Scene
from matplotlib import pyplot as plt
from projectaria_tools.core import calibration, data_provider, mps
from scipy.spatial.transform import Rotation as R
from spot_rl.envs.skill_manager import SpotSkillManager
from spot_rl.models.owlvit import OwlVit
from spot_wrapper.spot import Spot, SpotCamIds, image_response_to_cv2

DOCK_ID = int(os.environ.get("SPOT_DOCK_ID", 520))
AXES_SCALE = 0.9
STREAM1_NAME = "camera-rgb"
STREAM2_NAME = "camera-slam-left"
STREAM3_NAME = "camera-slam-right"
FILTER_DIST = 2.4  # in meters (distance for valid detection)

# from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image
from std_msgs.msg import Int64
from cv_bridge import CvBridge, CvBridgeError
############## Simple Helper Methods to keep code clean ##############

TOPICS = ["/pose", "/image_raw"]
TYPES = [PoseStamped, Image]

class AriaLiveStreamerTEMP:
    """
    This class is used to read data from rostopics
    It can detect April tags and objects of interest too

    For April tag detection, it uses the AprilTagPoseEstimator class (please refer to AprilTagPoseEstimator.py & AprilTagDetectorWrapper.py)
    For object detection, it uses the Owl-VIT model (please refer to OwlVit.py & ObjectDetectorWrapper.py)

    It also has a few helpers for image rectification, image rotation, image display, etc; and a few helpers to get VRS and MPS file streaming

    Args:
        qr (bool, optional): Boolean to indicate if QR code (Dock ID) should be detected. Defaults to False.
        verbose (bool, optional): Verbosity flag. Defaults to False.

    """

    def __init__(self, qr = False, verbose=False):

        # Verbosity flag for updating images when passed through detectors (this is different from config.VERBOSE)
        self.verbose = verbose

        self.parse_live_stream()

    ##############################################################
    # LIVE STREAM PARSER --- ARIA

    def parse_live_stream(self, 
        # device_T_camera: Any,
        # outputs: Dict, 
        # detect_qr: bool=True,
        # should_display: bool = True,
        # detect_objects: bool = False,
        # object_labels: List[str] = None,
        # # iteration_range: Tuple[int, int] = None,
        # # reverse: bool = False,
        # meta_objects: List[str] = ["hand"],
    ):
        """
        Parse live stream from Aria
        """
        import aria.sdk as aria
        from common import quit_keypress, update_iptables
        from projectaria_tools.core.sensor_data import ImageDataRecord

        #if args.update_iptables and sys.platform.startswith("linux"):
        if True:
            update_iptables()

        #  Optional: Set SDK's log level to Trace or Debug for more verbose logs. Defaults to Info
        aria.set_log_level(aria.Level.Info)

        # 1. Create StreamingClient instance
        streaming_client = aria.StreamingClient()

        #  2. Configure subscription to listen to Aria's RGB stream.
        # @see StreamingDataType for the other data types
        config = streaming_client.subscription_config
        config.subscriber_data_type = (
            aria.StreamingDataType.Rgb
        )

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


        # 3. Create and attach observer
        class StreamingClientObserver:
            def __init__(self):
                self.images = {}
                self.record = {}

            def on_image_received(self, image: np.array, record: ImageDataRecord):
                self.images[record.camera_id] = image
                self.record[record.camera_id] = record

        observer = StreamingClientObserver()
        streaming_client.set_streaming_client_observer(observer)

        # 4. Start listening
        print("Start listening to image data")
        streaming_client.subscribe()

        # 5. Visualize the streaming data until we close the window
        rgb_window = "Aria RGB"

        cv2.namedWindow(rgb_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(rgb_window, 1024, 1024)
        cv2.setWindowProperty(rgb_window, cv2.WND_PROP_TOPMOST, 1)
        cv2.moveWindow(rgb_window, 50, 50)

        outputs: Dict[str, Any] = {}
        while not quit_keypress():
            # Render the RGB image
            if aria.CameraId.Rgb in observer.images:
                raw_rgb_image = np.rot90(observer.images[aria.CameraId.Rgb], -1)
                cv2.imshow(rgb_window, raw_rgb_image)
                del observer.images[aria.CameraId.Rgb]

    ##############################################################


@click.command()
@click.option("--verbose", type=bool, default=True)
def main(
    verbose: bool,
):
    rospy.init_node('aria_live_streamer_temp', anonymous=False)
    aria_live_streamer = AriaLiveStreamerTEMP(
       verbose=verbose
    )


if __name__ == "__main__":
    main()
