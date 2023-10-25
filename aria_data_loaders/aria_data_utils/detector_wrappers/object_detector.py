from typing import Any, Dict, List, Tuple

import numpy as np
from aria_data_utils.detector_wrappers.generic_detector_interface import GenericDetector
from aria_data_utils.image_utils import centered_heuristic, check_bbox_intersection
from spot_rl.models.owlvit import OwlVit


class ObjectDetectorWrapper(GenericDetector):
    def __init__(self):
        super().__init__()

    def _init_object_detector(
        self, object_labels: List[str], verbose: bool = None
    ) -> Dict[str, Any]:
        """initialize object_detector by loading it to GPU and setting up the labels"""
        assert self.is_enabled is True
        # TODO: Decide VERBOSE
        # if verbose is None:
        #     verbose = self.verbose
        self.object_detector = OwlVit(
            [object_labels], score_threshold=0.125, show_img=verbose
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
        - valid: bool: whether the img frame is valid or not
        - score: float: score of the img frame (based on :heuristic:)
        - result_image: np.ndarray: image with bounding boxes drawn on it
        """
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

    def process_frame(self, img_frame: np.ndarray, outputs: Dict, img_metadata: Any):
        assert self.is_enabled is True

        # FIXME: done for 1 specific object at the moment, extend for multiple
        valid, score, stop, result_img = self._get_scored_object_detections(
            img_frame, heuristic=self._p1_demo_heuristics
        )
        if stop and stop["penguin_plush"]:
            print("Turning off object-detection")
            self.disable_detector()
        if valid:
            # score_string = str(score["penguin_plush"]).replace(".", "_")
            # plt.imsave(
            #     f"./results/{frame_idx}_{valid}_{score_string}.jpg", result_img
            # )
            # print(f"saving valid img_frame, {score=}")
            print(f"valid img_frame, {score}")
            for object_name in score.keys():
                outputs["object_image_list"][object_name].append(img_frame)
                outputs["object_score_list"][object_name].append(score[object_name])
                # TODO: following is not being used at the moment, clean-up if
                # multi-object, multi-view logic seems to not require this
                outputs["object_image_segment_list"][object_name].append(-1)
                outputs["object_image_metadata_list"][object_name].append(img_metadata)

        return result_img, outputs
