import os

from aria_data_utils.aria_sdk_utils import update_iptables_quest3
from aria_data_utils.human_sensor_data_streamer_interface import (
    CameraParams,
    DataFrame,
    HumanSensorDataStreamerInterface,
)
from perception_and_utils.utils.frame_rate_counter import (  # Local Frame rate counter
    FrameRateCounter,
)
from perception_and_utils.utils.math_utils import get_running_avg_a_T_b

try:
    from quest3_streamer.unified_quest_camera import (  # From internal repo
        UnifiedQuestCamera,
    )
except ImportError:
    print("Could not import Quest3 camera wrapper")


import json
import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import click
import cv2
import matplotlib.pyplot as plt
import numpy as np
import rospy

try:
    import sophuspy as sp
except Exception as e:
    print(f"Cannot import sophuspy due to {e}. Import sophus instead")
    import sophus as sp

from perception_and_utils.utils.conversions import (
    sophus_SE3_to_ros_PoseStamped,
    sophus_SE3_to_ros_TransformStamped,
    xyt_to_sophus_SE3,
)
from spot_rl.utils.utils import ros_frames as rf

FILTER_DIST = 0.6  # in metres, the QR registration on Quest3 is REALLY bad beyond 0.6m
NUM_OF_FRAME_OBJECT_DETECTED = 2


class Quest3DataStreamer(HumanSensorDataStreamerInterface):
    def __init__(self, verbose: bool = False, *args, **kwargs) -> None:
        super().__init__(verbose, *args, **kwargs)

        # Init Logging
        self.logger = logging.getLogger("Quest3DataStreamer")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.verbose = verbose

        self.frc_all = FrameRateCounter()
        self.frc_qrd = FrameRateCounter()
        self.frc_hmd = FrameRateCounter()
        self.frc_od = FrameRateCounter()

        # Backward compatibility
        try:
            self.unified_quest3_camera = UnifiedQuestCamera(verbose=self.verbose)
        except Exception:
            self.unified_quest3_camera = UnifiedQuestCamera()

        self.rgb_cam_params: CameraParams = None
        self.depth_cam_params: CameraParams = None
        self._frame_number = -1
        self._setup_device()

        # state-machine for HAR
        self.finding_object_stage = False
        self._partial_action: dict = {}

        # check the consecutive frame that is the correct detection
        self._object_detected = []  # type: ignore

    def connect(self):
        """
        Connect to Device
        """
        self._is_connected = True
        self.logger.info("Quest3DataStreamer :: Connected")

    def disconnect(self):
        """
        Disconnects from Device cleanly
        """
        self._is_connected = False
        self.logger.info("Quest3DataStreamer :: Disconnected")

    def is_connected(self) -> bool:
        return (
            self._is_connected
        )  # TODO: Return device connection status from Bruno's APIs

    def _setup_device(self) -> Any:
        # Setup RGB camera params
        while self.rgb_cam_params is None:
            rgb_fl = self.unified_quest3_camera.get_rbg_focal_lengths()
            rgb_pp = self.unified_quest3_camera.get_rgb_principal_point()

            if rgb_fl is not None and rgb_pp is not None:
                rospy.logwarn(
                    "RGB Camera Params: focal lengths = {}, principal point = {}".format(
                        rgb_fl, rgb_pp
                    )
                )
                self.rgb_cam_params = CameraParams(
                    focal_lengths=rgb_fl, principal_point=rgb_pp
                )
        else:
            self.logger.info("Waiting for RGB camera params to be set...")
            rospy.logwarn("Waiting for RGB camera params to be set...")
            time.sleep(0.1)
        rospy.loginfo(
            "RGB Camera Params: focal lengths = {}, principal point = {}".format(
                self.rgb_cam_params._focal_lengths, self.rgb_cam_params._principal_point
            )
        )

        # Setup Depth camera params
        while self.depth_cam_params is None:
            depth_fl = self.unified_quest3_camera.get_depth_focal_lengths()
            depth_pp = self.unified_quest3_camera.get_depth_principal_point()

            if depth_fl is not None and depth_pp is not None:
                rospy.logwarn(
                    "Depth Camera Params: focal lengths = {}, principal point = {}".format(
                        depth_fl, depth_pp
                    )
                )
                self.depth_cam_params = CameraParams(
                    focal_lengths=depth_fl, principal_point=depth_pp
                )
            else:
                self.logger.info("Waiting for Depth camera params to be set...")
                rospy.logwarn("Waiting for Depth camera params to be set...")
                time.sleep(0.1)
        print(
            "Depth Camera Params: focal lengths = {}, principal point = {}".format(
                self.depth_cam_params._focal_lengths,
                self.depth_cam_params._principal_point,
            )
        )

    def get_latest_data_frame(self) -> Optional[DataFrame]:
        # Create DataFrame object
        data_frame = DataFrame()

        # Populate rgb frame data iff all the required data is available; else return None
        rgb_frame = self.unified_quest3_camera.get_rgb()
        deviceWorld_T_rgbCam = self.unified_quest3_camera.get_deviceWorld_T_rgbCamera()
        device_T_rgbCam = self.unified_quest3_camera.get_device_T_rgbCamera()
        if rgb_frame is None or deviceWorld_T_rgbCam is None or device_T_rgbCam is None:

            if self.verbose:
                rospy.logwarn(
                    "Returning None, as rgb_frame or deviceWorld_T_rgbCam or device_T_rgbCam is None."
                )
            return None
        else:
            data_frame._rgb_frame = rgb_frame
            data_frame._deviceWorld_T_camera_rgb = deviceWorld_T_rgbCam
            data_frame._device_T_camera_rgb = device_T_rgbCam

        # Populate depth frame data iff all the required data is available; else return None
        depth_frame = self.unified_quest3_camera.get_depth()
        deviceWorld_T_depthCam = (
            self.unified_quest3_camera.get_deviceWorld_T_depthCamera()
        )
        device_T_depthCam = self.unified_quest3_camera.get_device_T_depthCamera()
        if depth_frame is None or deviceWorld_T_depthCam is None:
            if self.verbose:
                rospy.logwarn(
                    "Returning None as depth_frame or deviceWorld_T_depthCam or device_T_depthCam is None."
                )
            return None
        else:
            data_frame._depth_frame = depth_frame
            data_frame._deviceWorld_T_camera_depth = deviceWorld_T_depthCam
            data_frame._device_T_camera_depth = device_T_depthCam

        # Update frame number and timestamp of DataFrame
        self._frame_number += 1
        data_frame._frame_number = self._frame_number
        data_frame._timestamp_s = time.time()

        # Align depth frame with rgb frame
        data_frame._aligned_depth_frame = None  # TODO: Add this logic

        # Update frame rate counter
        # self.frame_rate_counter.update()
        data_frame._avg_rgb_fps = self.unified_quest3_camera.get_avg_fps_rgb()
        data_frame._avg_depth_fps = self.unified_quest3_camera.get_avg_fps_depth()
        # data_frame._avg_data_frame_fps = self.frame_rate_counter.avg_value()
        data_frame._avg_data_frame_fps = 0.0

        return data_frame

    def _get_iou(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou

    def process_frame(
        self,
        data_frame: DataFrame,
        outputs: dict,
        detect_qr: bool = False,
        detect_objects: bool = False,
        detect_human_motion: bool = False,
        detect_human_action: bool = False,
        object_label="milk bottle",
        test_without_registration: bool = False,
    ) -> Tuple[np.ndarray, dict]:
        rate_all = 0.0
        rate_qrd = 0.0
        rate_hmd = 0.0
        rate_od = 0.0

        self.frc_all.start()

        # Initialize object_scores to empty dict for current image frame
        # object_scores = {}  # type: Dict[str, Any]

        # Reset the state machine if needed
        if rospy.get_param("reset_har", False):
            self.finding_object_stage = False
            self._partial_action = {}
            self.har_model.current_state = "not_holding"
            rospy.set_param("reset_har", False)

        # Get rgb image from frame
        viz_img = data_frame._rgb_frame
        if viz_img is None:
            rospy.logwarn("No image found in frame")  # This gets over-written.. WASTE!

        if detect_human_action:
            action = {}
            har_output_img, har_output_dict = self.har_model.process_frame(data_frame)
            har_instances = har_output_dict["instances"]
            har_object_bbox = (
                har_instances.pred_boxes[
                    har_instances.pred_classes == self.har_model.OBJECT_CATEGORY
                ]
                .tensor.cpu()
                .numpy()
                .tolist()
            )
            viz_img, detections = self.object_detector.process_frame(data_frame)

            if har_output_dict.get("action_trigger", None) is not None:
                action_string = har_output_dict["action_trigger"]

                action = {
                    "action": action_string,
                    "location": [
                        0.0,
                        0.0,
                        0.0,
                    ],
                }
                if not test_without_registration:
                    human_place_xyz = None
                    try:
                        human_place_xyz = self.get_nav_xyz_to_wearer(
                            self.get_handoff_to_human_pose(
                                source=rf.QUEST3_CAMERA, target=rf.SPOT_WORLD
                            ),
                            shift_offset=0.3,
                        )
                    except RuntimeError:
                        rospy.logwarn("Error in finding place point in front of human")
                    action["location"] = [
                        human_place_xyz[0],
                        human_place_xyz[1],
                        human_place_xyz[2],
                    ]

                if action_string == "pick":
                    self.finding_object_stage = True
                    print(
                        "\n\n******************** YAY PICKED OBJECT ********************\n\n"
                    )
                    self._partial_action = action

            if self.finding_object_stage:
                max_iou = 0.0
                arg_max_indx = -1
                for det_indx, det in enumerate(detections):
                    print("Object-name:", det[0], "; Score:", det[1], "; Bbox:", det[2])
                    try:
                        iou = self._get_iou(har_object_bbox[0], det[-1])
                        print("IOU = ", iou)
                        if iou > max_iou:
                            max_iou = iou
                            arg_max_indx = det_indx
                    except Exception:
                        print("skip due to har_object_bbox is empty")
                if arg_max_indx != -1 and max_iou > 0.5:

                    # Check the consecutive detection
                    obj_class = detections[arg_max_indx][0]
                    if self._object_detected == []:
                        self._object_detected.append(obj_class)
                    elif obj_class not in self._object_detected:
                        self._object_detected = []  # reset
                    else:
                        self._object_detected.append(obj_class)

                    print(f"number of times being held: {len(self._object_detected)}")
                    if len(self._object_detected) >= NUM_OF_FRAME_OBJECT_DETECTED:
                        self._partial_action["object"] = obj_class
                        print("Object being held: ", obj_class)
                        self.finding_object_stage = False
                        action = self._partial_action
                        self._partial_action = {}
                        self._object_detected = []
                    else:
                        action = {}
                else:
                    action = {}

            if action:
                # Broadcast action string data
                action_json_string = json.dumps(action)
                print(f"Action to publish {action}")
                if action.get("action", "None") == "pick":
                    setattr(self, "picked_object", action.get("object", "None"))
                location_str = "unknown"
                rospy.set_param(
                    "/human_action",
                    f"{str(time.time())},{action.get('action', 'None')},{action.get('object', getattr(self, 'picked_object', 'None'))},[{location_str}]",
                )
                self.human_activity_current_pub.publish(action_json_string)

        # if detect_objects:
        #     self.frc_od.start()
        #     viz_img, object_scores = self.object_detector.process_frame(data_frame)
        #     # if object_label in object_scores.keys():
        #     #     if self.verbose:
        #     #         plt.imsave(f"frame_{self.frame_number}.jpg", viz_img)
        #     #     self.publish_pose_of_interest(frame.get("ros_pose"))
        #     rate_od = self.frc_od.stop()

        if detect_qr:
            self.frc_qrd.start()
            (viz_img, camera_T_marker) = self.april_tag_detector.process_frame(
                frame=data_frame
            )  # type: ignore

            deviceWorld_T_camera = data_frame._deviceWorld_T_camera_rgb
            # Broadcast camera frame wrt deviceWorld
            self.static_tf_broadcaster.sendTransform(
                sophus_SE3_to_ros_TransformStamped(
                    sp_se3=deviceWorld_T_camera,
                    parent_frame=rf.QUEST3_WORLD,
                    child_frame=rf.QUEST3_CAMERA,
                )
            )

            if camera_T_marker is not None:
                # Frame: a = deviceWorld (quest3World for Quest3 & ariaWorld for Aria)
                # Frame: intermediate = camera (quest3's rgb camera frame & cpf frame for Aria)
                # Frame: b = marker / qr code
                (
                    self.marker_positions_list,
                    self.marker_quaternion_list,
                    self.avg_deviceWorld_T_marker,
                ) = get_running_avg_a_T_b(
                    current_avg_a_T_b=self.avg_deviceWorld_T_marker,  # type: ignore
                    a_T_b_position_list=self.marker_positions_list,  # type: ignore
                    a_T_b_quaternion_list=self.marker_quaternion_list,  # type: ignore
                    a_T_intermediate=deviceWorld_T_camera,
                    intermediate_T_b=camera_T_marker,
                    filter_dist=FILTER_DIST,
                )

                # Publish marker pose in ariaWorld frame
                if self.avg_deviceWorld_T_marker is not None:
                    avg_marker_T_deviceWorld = self.avg_deviceWorld_T_marker.inverse()
                    self.static_tf_broadcaster.sendTransform(
                        sophus_SE3_to_ros_TransformStamped(
                            sp_se3=avg_marker_T_deviceWorld,
                            parent_frame=rf.MARKER,
                            child_frame=rf.QUEST3_WORLD,
                        )
                    )

                viz_img, outputs = self.april_tag_detector.get_outputs(
                    frame=data_frame,
                    outputs=outputs,
                    base_T_marker=camera_T_marker,
                    timestamp=data_frame._timestamp_s,
                    img_metadata=None,
                    viz_img=viz_img,
                )

            rate_qrd = self.frc_qrd.stop()

        if detect_human_motion:
            self.frc_hmd.start()
            activity_str, avg_velocity = self.human_motion_detector.process_frame(
                frame=data_frame
            )
            ops = {
                "activity": activity_str,
                "velocity": avg_velocity,
                "data_frame_fps": data_frame._avg_data_frame_fps,
            }
            viz_img, _ = self.human_motion_detector.get_outputs(viz_img, ops)
            self.publish_human_activity_history(
                self.human_motion_detector.get_human_motion_history()
            )
            rate_hmd = self.frc_hmd.stop()

        # if detect_objects:
        #     self.frc_od.start()
        #     viz_img, object_scores = self.object_detector.process_frame(viz_img)
        #     if object_label in object_scores.keys():
        #         if self.verbose:
        #             plt.imsave(f"frame_{self.frame_number}.jpg", viz_img)
        #         # self.publish_pose_of_interest(frame.get("ros_pose"))

        #     rate_od = self.frc_od.stop()

        rate_all = self.frc_all.stop()

        outputs["process_all"] = rate_all
        outputs["process_qr"] = rate_qrd
        outputs["process_od"] = rate_od
        outputs["process_hmd"] = rate_hmd
        return [har_output_img, viz_img], outputs

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

        focal_lengths = self.rgb_cam_params._focal_lengths  # type: ignore
        principal_point = self.rgb_cam_params._principal_point  # type: ignore

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
                score_threshold=0.4,
            )
        )
        self.object_detector._core_objects = object_labels
        self.object_detector._meta_objects = meta_objects

        return outputs

    def initialize_human_motion_detector(self):
        self.human_motion_detector._init_human_motion_detector()


@click.command()
@click.option("--do-update-iptables", is_flag=True, type=bool, default=False)
@click.option("--debug", is_flag=True, default=False)
@click.option("--hz", type=int, default=100)
@click.option("--har-model-path")
@click.option("--har-config-path")
@click.option("--test-without-registration", default=False, is_flag=True)
def main(
    do_update_iptables: bool,
    debug: bool,
    hz: int,
    har_model_path: str,
    har_config_path: str,
    test_without_registration: bool,
):
    if debug:
        _log_level = rospy.DEBUG
    else:
        _log_level = rospy.INFO

    rospy.init_node("quest3_data_streamer", log_level=_log_level)
    rospy.loginfo("Starting quest3_data_streamer node")

    # Logger for profiling
    loggerp = logging.getLogger("Profiler-Quest3ProcessFrame")
    loggerp.setLevel(logging.INFO)
    csv_handlerp = logging.FileHandler("quest3_time_profiles_sec.txt")
    formatterp = logging.Formatter("%(asctime)s, %(message)s")
    csv_handlerp.setFormatter(formatterp)
    loggerp.addHandler(csv_handlerp)

    if do_update_iptables:
        update_iptables_quest3()
    else:
        rospy.logwarn("Not updating iptables")

    outer_frc = FrameRateCounter()
    # rate = rospy.Rate(hz)
    outputs: Dict[str, Any] = {}
    data_streamer = None
    cv2.namedWindow("HAR", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("OWL-ViT", cv2.WINDOW_NORMAL)
    try:
        data_streamer = Quest3DataStreamer(
            har_model_path=har_model_path, har_config_path=har_config_path
        )
        time.sleep(5.0)

        outputs = data_streamer.initialize_april_tag_detector(outputs=outputs)
        outputs = data_streamer.initialize_object_detector(
            outputs=outputs,
            object_labels=[
                "pineapple plush toy",
                "pink donut plush toy",
                "avocado plush toy",
                "cup",
                "bottle",
                "can",
            ],
        )
        data_streamer.initialize_human_motion_detector()
        # data_streamer.connect()
        while not rospy.is_shutdown():
            outer_frc.start()
            data_frame = data_streamer.get_latest_data_frame()
            if data_frame is not None:
                if debug:
                    rospy.loginfo("Received data frame")
                viz_img, outputs = data_streamer.process_frame(
                    data_frame=data_frame,
                    outputs=outputs,
                    detect_qr=True,
                    detect_objects=True,
                    detect_human_motion=False,
                    detect_human_action=True,
                    test_without_registration=test_without_registration,
                )
                # data_streamer.publish_human_pose(data_frame=data_frame)
                viz_img[0] = cv2.resize(viz_img[0], viz_img[1].shape[:2][::-1])
                vis_image = np.hstack(viz_img[:2])
                human_action = rospy.get_param("human_action", "0,None,None,None")
                human_action = (
                    "None"
                    if "None" in human_action
                    else ",".join(human_action.split(",")[1:])
                )
                human_action_display = f"Human Action : {human_action}"
                vis_image = cv2.putText(
                    vis_image,
                    human_action_display,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("HAR", vis_image)
                # cv2.imshow("OWL-ViT", viz_img[1])
                cv2.waitKey(1)
            else:
                rospy.logdebug("No data frame received.")

            outer_rate = outer_frc.stop()
            msg_str = f"Fetch+AllDetection:{outer_rate},All_detectors={outputs.get('process_all',0.0)},QR={outputs.get('process_qr',0.0)},OD={outputs.get('process_od',0.0)},HMD={outputs.get('process_hmd',0.0)}"
            loggerp.debug(msg_str)
            # print(msg_str)

    except Exception:
        print("Ending script.")
        rospy.logwarn(f"Exception: {traceback.format_exc()}")
        if data_streamer is not None:
            data_streamer.disconnect()

    cv2.destroyAllWindows()
    rospy.logwarn("Disconnecting and shutting down the node")
    os._exit(1)  # Exits disgracefully


if __name__ == "__main__":
    main()
