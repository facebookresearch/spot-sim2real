import os

from aria_data_utils.aria_sdk_utils import update_iptables_quest3
from perception_and_utils.utils.data_frame import DataFrame
from perception_and_utils.utils.frame_rate_counter import (  # Local Frame rate counter
    FrameRateCounter,
)
from perception_and_utils.perception.detector_wrappers.object_detector import (
    ObjectDetectorWrapper,
)
from perception_and_utils.perception.human_action_recognition_state_machine import (
    HARStateMachine,
)

try:
    from quest3_streamer.unified_quest_camera import UnifiedQuestCamera
except ImportError:
    print("Could not import Quest3 camera wrapper")


import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import click
import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    import sophuspy as sp
except Exception as e:
    print(f"Cannot import sophuspy due to {e}. Import sophus instead")
    import sophus as sp


FILTER_DIST = 0.6  # in metres, the QR registration on Quest3 is REALLY bad beyond 0.6m
NUM_OF_FRAME_OBJECT_DETECTED = 1


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


class HumanObjectInteractions():
    def __init__(self, verbose: bool = False, *args, **kwargs) -> None:

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

        model_path = kwargs.get("har_model_path", None)
        model_config_path = kwargs.get("har_config_path", None)
        if model_path is None or model_config_path is None:
            raise ValueError(
                "Expected HAR model details to be passed as har_model_path and har_config_path kwargs"
            )
        self.har_model = HARStateMachine(model_path, model_config_path, verbose=verbose)
        self.object_detector = ObjectDetectorWrapper()

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
                self.logger.info(
                    "RGB Camera Params: focal lengths = {}, principal point = {}".format(
                        rgb_fl, rgb_pp
                    )
                )
                self.rgb_cam_params = CameraParams(
                    focal_lengths=rgb_fl, principal_point=rgb_pp
                )
            else:
                self.logger.warning("Waiting for RGB camera params to be set...")
                time.sleep(0.1)
        self.logger.info(
            "RGB Camera Params: focal lengths = {}, principal point = {}".format(
                self.rgb_cam_params._focal_lengths, self.rgb_cam_params._principal_point
            )
        )

        # Setup Depth camera params
        while self.depth_cam_params is None:
            depth_fl = self.unified_quest3_camera.get_depth_focal_lengths()
            depth_pp = self.unified_quest3_camera.get_depth_principal_point()

            if depth_fl is not None and depth_pp is not None:
                self.logger.info(
                    "Depth Camera Params: focal lengths = {}, principal point = {}".format(
                        depth_fl, depth_pp
                    )
                )
                self.depth_cam_params = CameraParams(
                    focal_lengths=depth_fl, principal_point=depth_pp
                )
            else:
                self.logger.warning("Waiting for Depth camera params to be set...")
                time.sleep(0.1)
        self.logger.info(
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
                self.logger.warning(
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
                self.logger.warning(
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
        #         if rospy.get_param("reset_har", False):
        #             self.finding_object_stage = False
        #             self._partial_action = {}
        #             self.har_model.current_state = "not_holding"
        # \
        # Get rgb image from frame
        viz_img = data_frame._rgb_frame
        if viz_img is None:
            self.logger.warning("No image found in frame")  # This gets over-written.. WASTE!

        if detect_objects:
            viz_img, detections = self.object_detector.process_frame(data_frame)

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

            if har_output_dict.get("action_trigger", None) is not None:

                action = {
                    "action": har_output_dict["action_trigger"],
                }

                if action.get("action") == "pick":
                    self.finding_object_stage = True
                    print(
                        "\n\n******************** YAY PICKED OBJECT ********************\n\n"
                    )
                    self._partial_action = action

            print(self.finding_object_stage)
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

            outputs["human_action"] = action

        rate_all = self.frc_all.stop()

        outputs["process_all"] = rate_all
        outputs["process_qr"] = rate_qrd
        outputs["process_od"] = rate_od
        outputs["process_hmd"] = rate_hmd
        return [har_output_img, viz_img], outputs

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


@click.command()
@click.option("--do-update-iptables", is_flag=True, type=bool, default=False)
@click.option("--debug", is_flag=True, default=False)
@click.option("--har-model-path")
@click.option("--har-config-path")
def main(
    do_update_iptables: bool,
    debug: bool,
    har_model_path: str,
    har_config_path: str,
):

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
        print("Not updating iptables")

    outer_frc = FrameRateCounter()
    outputs: Dict[str, Any] = {}
    data_streamer = None
    cv2.namedWindow("HAR", cv2.WINDOW_NORMAL)
    try:
        data_streamer = HumanObjectInteractions(
            har_model_path=har_model_path, har_config_path=har_config_path
        )
        time.sleep(5.0)

        outputs = data_streamer.initialize_object_detector(
            outputs=outputs,
            object_labels=[
                # Remove the following objects from the list since they confuse the detection
                # "pineapple plush toy",
                # "pink donut plush toy",
                # "avocado plush toy",
                "cup",
                "bottle",
                "can",
            ],
        )
        # data_streamer.connect()
        while True:
            outer_frc.start()
            data_frame = data_streamer.get_latest_data_frame()
            if data_frame is not None:
                if debug:
                    print("Received data frame")
                viz_img, outputs = data_streamer.process_frame(
                    data_frame=data_frame,
                    outputs=outputs,
                    detect_objects=True,
                    detect_human_action=True,
                )
                viz_img[0] = cv2.resize(viz_img[0], viz_img[1].shape[:2][::-1])
                vis_image = np.hstack(viz_img[:2])
                human_action = outputs.get("human_action", "None")
                human_action = (
                    "None"
                    if "None" in human_action
                    else human_action.get("action", "None")
                )
                current_state = data_streamer.har_model.current_state
                human_action_display = (
                    f"Human action: {human_action} , State: {current_state}"
                )

                vis_image = cv2.putText(
                    vis_image,
                    human_action_display,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("HAR", vis_image)
                cv2.waitKey(1)
            else:
                print("No data frame received.")

            outer_rate = outer_frc.stop()
            msg_str = f"Fetch+AllDetection : {outer_rate}, All_detectors= {outputs.get('process_all', 0.0)}, QR = {outputs.get('process_qr', 0.0)} , OD = {outputs.get('process_od', 0.0)} , HMD = {outputs.get('process_hmd', 0.0)}"
            loggerp.debug(msg_str)

    except Exception:
        print("Ending script.")
        print(f"Exception: {traceback.format_exc()}")
        if data_streamer is not None:
            data_streamer.disconnect()

    cv2.destroyAllWindows()
    print("Disconnecting and shutting down the node")
    os._exit(1)  # Exits disgracefully


if __name__ == "__main__":
    main()