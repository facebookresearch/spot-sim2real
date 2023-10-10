from typing import Any, Dict, List, Tuple

import numpy as np
from aria_data_utils.detector_wrappers.generic_detector_interface import GenericDetector
from aria_data_utils.image_utils import centered_heuristic, check_bbox_intersection
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

        # Enable detector
        odw.enable_detector() # base class method

        # Initialize detector
        outputs = odw._init_object_detector(object_labels, verbose)

        # Process image frame
        updated_img_frame, outputs = odw.process_frame(img_frame, outputs, img_metadata)
    """

    def __init__(self):
        super().__init__()

    def _init_object_detector(
        self, object_labels: List[str], verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Initialize object_detector by loading it to GPU and setting up the labels

        Args:
            object_labels (List[str]) : List of object labels
            verbose (bool) : If True, modifies image frame to render detected objects

        Returns:
            outputs (Dict[str, Any]) : Dictionary of outputs
            Manipulates following keys in the outputs dictionary:
                - object_image_list (List[np.ndarray]) : List of decorated image frames
                - object_image_metadata_list (List[Any]) : List of image metadata
                - object_image_segment_list (List[int]) : List of image segment ids
                - object_score_list (List[float]) : List of scores for each image frame
        """
        assert self.is_enabled is True
        self.verbose = verbose

        self.object_detector = OwlVit(
            [object_labels], score_threshold=0.125, show_img=self.verbose
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
        return outputs

    def _get_scored_object_detections(
        self, img: np.ndarray, heuristic=None
    ) -> Tuple[bool, Dict[str, float], Dict[str, bool], np.ndarray]:
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

        return valid, score, stop, result_image

    def _p1_demo_heuristics(
        self, detections, img_size=(512, 512)
    ) -> Tuple[bool, Dict[str, bool], Dict[str, float]]:
        """*P1 Demo Specific*
        Heuristics to filter out unwanted and bad obejct detections. This heuristic
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
        valid = False
        score = {}
        stop = {}
        # figure out if this is an interaction img frame
        if len(detections) > 1:
            objects_in_frame = [det[0] for det in detections]
            # FIXME: only works for 1 object of interest right now, extend to multiple
            if "hand" in objects_in_frame and "penguin_plush" in objects_in_frame:
                print(f"checking for intersection b/w: {object}")
                stop["penguin_plush"] = check_bbox_intersection(
                    detections[objects_in_frame.index("penguin_plush")][2],
                    detections[objects_in_frame.index("hand")][2],
                )
                print(f"Intersection: {stop}")
        if len(detections) == 1:
            if "penguin_plush" == detections[0][0]:
                score["penguin_plush"] = centered_heuristic(detections)[0]
                valid = True

        return valid, score, stop

    def process_frame(self, img_frame: np.ndarray):
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
        # Do nothing if detector is not enabled
        if self.is_enabled is False:
            return img_frame, {}

        # FIXME: done for 1 specific object at the moment, extend for multiple
        (
            valid,
            object_scores,
            stop,
            updated_img_frame,
        ) = self._get_scored_object_detections(
            img_frame, heuristic=self._p1_demo_heuristics
        )
        if stop and stop["penguin_plush"]:
            print("Turning off object-detection")
            self.disable_detector()

        # Ignore the score if the detections are not valid
        if not valid:
            object_scores = {}

        return updated_img_frame, object_scores

    def get_outputs(
        self,
        img_frame: np.ndarray,
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
