# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from perception_and_utils.perception.detector_wrappers.generic_detector_interface import (
    GenericDetector,
)
from perception_and_utils.utils.data_frame import DataFrame
from perception_and_utils.utils.image_utils import (
    centered_object_detection_heuristic,
    check_bbox_intersection,
)
from spot_rl.models.owlvit import OwlVit


class ObjectDetectorWrapper(GenericDetector):
    """
    Wrapper over OwlVit class to detect object instances and score them based on heuristics

    How to use:
        1. Create an instance of this class
        2. Call `process_frame` method with image as input & outputs dict as input

    Example:
        # Create an instance of ObjectDetectorWrapper
        odw = ObjectDetectorWrapper()

        # Initialize detector
        outputs = odw._init_object_detector(object_labels, verbose)

        # Process image frame
        updated_img_frame, outputs = odw.process_frame(img_frame, outputs, img_metadata)
    """

    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger("ObjectDetectorWrapper")

    def _init_object_detector(
        self,
        object_labels: List[str],
        verbose: bool = True,
        version: int = 1,
        score_threshold: float = 0.125,
    ) -> Dict[str, Any]:
        """
        Initialize object_detector by loading it to GPU and setting up the labels

        Args:
            object_labels (List[str]) : List of object labels
            verbose (bool) : If True, modifies image frame to render detected objects
            version (int) : Version of the object detector to use (supports 1 and 2)
            score_threshold (float) : Confidence threshold for object detection

        Returns:
            outputs (Dict[str, Any]) : Dictionary of outputs
            Manipulates following keys in the outputs dictionary:
                - object_image_list (List[np.ndarray]) : List of decorated image frames
                - object_image_metadata_list (List[Any]) : List of image metadata
                - object_image_segment_list (List[int]) : List of image segment ids
                - object_score_list (List[float]) : List of scores for each image frame
        """
        self.enable_detector()
        self.verbose = verbose

        self.object_detector = OwlVit(
            [object_labels],
            score_threshold=score_threshold,
            show_img=self.verbose,
            version=version,
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
        self._core_objects: list = None
        self._meta_objects: list = None
        return outputs

    def _get_scored_object_detections(
        self, img: np.ndarray, heuristic=None
    ) -> Tuple[bool, Dict[str, Any], Dict[str, bool], np.ndarray]:
        """
        Detect object instances in the img frame. score them based on heuristics
        and return a tuple of (valid, score, result_image)
        Input:
        - img: np.ndarray: image to run inference on
        - heuristic: function: function to score the detections
        Output:
        - valid: bool: whether the img frame is valid or not, False if no heuristic
        - score: Dict: score of the img frame (based on :heuristic:), empty dict if no heuristic
        - stop: Dict: condition representing whether to stop the detector or not, empty dict if no heuristic
        - result_image: np.ndarray: image with bounding boxes drawn on it
        """
        valid = False
        score = {}
        stop = {}
        detections, result_image = self.object_detector.run_inference_and_return_img(
            img
        )

        if heuristic is not None:
            valid, score, stop = heuristic(detections)
            detections["_valid"] = valid
            detections["_score"] = score
            detections["_stop"] = stop

        return result_image, detections

    def _aria_online_heuristic(
        self, detections, img_size=(512, 512), score_thresh=0.35
    ) -> Tuple[bool, Dict[str, float], Dict[str, bool]]:
        """*Online EMRoF Specific*
        Simple heuristic where we detect only the objects of interest and score current
        frame depending on the % of pixels occupied by the object in the image frame and
        proximity of bbox to image center. This is then passed to the EMRoF algorithm.

        Args:
            detections (List[List[str, float, List[int]]]): List of detections
            img_size (Tuple[int, int]): Size of the image frame
            score_thresh (float): Confidence threshold for object detection

        Returns:
            valid (bool): Whether the img frame is valid or not
            score (Dict[str, float]): Score of the img frame (based on :heuristic:)
            stop (Dict[str, bool]): For backward compatibility, always empty
        """
        valid = False
        stop: Dict[str, Any] = {}  # does not apply to this method
        scores = {}
        core_detections = [
            det
            for det in detections
            if det[0] in self._core_objects and det[1] > score_thresh
        ]
        if core_detections:
            scores = centered_object_detection_heuristic(
                core_detections, image_size=img_size
            )
            valid = True
        return valid, scores, stop

    def _aria_fetch_demo_heuristics(
        self, detections, img_size=(512, 512)
    ) -> Tuple[bool, Dict[str, float], Dict[str, bool]]:
        """*P1 Demo Specific*
        Heuristics to filter out unwanted and bad object detections. This heuristic
        scores each img frame based on (a) % of pixels occupied by object in the img frame
        and (b) proximity of bbox to image center. Further only those images are
        scored which have only the object detection without a hand in the img frame.

        Detections are expected to have the format:
        [["object_name", "confidence", "bbox"]]

        Args:
            detections (List[List[str, float, List[int]]]): List of detections
            img_size (Tuple[int, int]): Size of the image frame

        Returns:
            valid (bool): Whether the img frame is valid or not
            score (Dict[str, float]): Score of the img frame (based on :heuristic:)
            stop (Dict[str, bool]): Whether to stop the detector or not
        """
        valid = True
        score = {}
        stop = {}
        # figure out if this is an interaction img frame
        if len(detections) > 1:
            objects_in_frame = [det[0] for det in detections]
            # find intersection b/w object_in_frame and self._meta_objects and
            # self._core_objects
            core_intersection = list(
                set(self._core_objects).intersection(set(objects_in_frame))
            )
            if "hand" in objects_in_frame and core_intersection != []:
                for object_name in core_intersection:
                    self._logger.debug(
                        f"checking for intersection b/w: {object_name} and hand"
                    )
                    stop[object_name] = check_bbox_intersection(
                        detections[objects_in_frame.index(object_name)][2],
                        detections[objects_in_frame.index("hand")][2],
                    )
                    self._logger.debug(f"Intersection: {stop[object_name]}")
        core_detections = [det for det in detections if det[0] in self._core_objects]
        score = centered_object_detection_heuristic(
            core_detections, image_size=img_size
        )

        if len(detections) == 1 and detections[0][0] == "hand" or len(detections) == 0:
            valid = False

        return valid, score, stop

    def process_frame(
        self, frame: DataFrame, heuristic=None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process image frame to detect object instances and score them based on heuristics

        Args:
            img_frame (np.ndarray) : Image frame to process
            outputs (Dict) : Dictionary of outputs (to be updated)
            img_metadata (Any) : Image metadata

        Returns:
            updated_img_frame (np.ndarray) : Image frame with detections and text for visualization
            object_scores (Dict[str, float]) : Dictionary of scores for each object in the image frame
        """
        img_frame = frame._rgb_frame

        # Do nothing if detector is not enabled
        if self.is_enabled is False:
            self._logger.warning(
                "Object detector is disabled... Skipping processing of current frame."
            )
            return img_frame, {}

        updated_img_frame, detections = self._get_scored_object_detections(
            img_frame, heuristic=heuristic
        )
        return updated_img_frame, detections

    def process_frame_offline(
        self, frame: DataFrame
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process image frame to detect object instances and score them based on heuristics

        Args:
            img_frame (np.ndarray) : Image frame to process
            outputs (Dict) : Dictionary of outputs (to be updated)
            img_metadata (Any) : Image metadata

        Returns:
            updated_img_frame (np.ndarray) : Image frame with detections and text for visualization
            object_scores (Dict[str, float]) : Dictionary of scores for each object in the image frame
        """
        img_frame = frame._rgb_frame

        # Do nothing if detector is not enabled
        if self.is_enabled is False:
            self._logger.warning(
                "Object detector is disabled... Skipping processing of current frame."
            )
            return img_frame, {}

        (
            valid,
            object_scores,
            stop,
            updated_img_frame,
        ) = self._get_scored_object_detections(
            img_frame, heuristic=self._aria_fetch_demo_heuristics
        )
        if stop != {}:
            # object-detection + checking for intersection b/w all objects and hand
            # is a computationally expensive operation (happens inside
            # _get_scored_object_detections()).
            # For efficiency, in this code, we turn off object detection for the objects
            # that have already been detected and their place location has been recorded
            # This is done by removing the object_name from the labels and regenerating
            # the prompts for the object detector.

            # check which object we need to stop detection for
            for object_name in stop.keys():
                if stop[object_name]:
                    self._logger.debug(
                        f"Turning off object-detection for {object_name}"
                    )
                    # delete object_name from labels and regenerate prompts
                    self._core_objects.remove(object_name)
                    self._logger.debug(f"Remaining objects: {self._core_objects}")
                    if not self._core_objects:
                        self.disable_detector()
                    else:
                        self.object_detector.update_label(
                            [self._core_objects + self._meta_objects]
                        )
                        self._logger.debug(
                            f"Updated labels: {self.object_detector.labels}"
                        )

        # Ignore the score if the detections are not valid
        if not valid:
            object_scores = {}

        return updated_img_frame, object_scores

    def get_outputs(
        self,
        frame: DataFrame,
        outputs: Dict,
        object_scores: Dict,
        img_metadata: Any,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Update the outputs dictionary with the processed image frame and object score data

        Args:
            img_frame (np.ndarray) : Image frame to process
            outputs (Dict) : Dictionary of outputs (to be updated)
            score (Dict[str, float]) : Dictionary of scores for each object in the image frame
            img_metadata (Any) : Image metadata

        Returns:
            img_frame (np.ndarray) : Image frame with detections and text for visualization
            outputs (Dict) : Updated dictionary of outputs
            Manipulates following keys in the outputs dictionary:
                - object_image_list (List[np.ndarray]) : List of decorated image frames
                - object_image_metadata_list (List[Any]) : List of image metadata
                - object_image_segment_list (List[int]) : List of image segment ids
                - object_score_list (List[float]) : List of scores for each image frame
        """
        img_frame = frame._rgb_frame

        if object_scores is not {}:
            for object_name in object_scores.keys():
                outputs["object_image_list"][object_name].append(img_frame)
                outputs["object_score_list"][object_name].append(
                    object_scores[object_name]
                )
                # TODO: following is not being used at the moment, clean-up if
                # multi-object, multi-view logic seems to not require this
                outputs["object_image_segment_list"][object_name].append(-1)
                outputs["object_image_metadata_list"][object_name].append(img_metadata)

        return img_frame, outputs
