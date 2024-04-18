# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
from typing import Any, Dict, Tuple

import numpy as np
import sophuspy as sp
from perception_and_utils.perception.april_tag_pose_estimator import (
    AprilTagPoseEstimator,
)
from perception_and_utils.perception.detector_wrappers.generic_detector_interface import (
    GenericDetector,
)
from perception_and_utils.utils.image_utils import decorate_img_with_text_for_qr

DOCK_ID = int(os.environ.get("SPOT_DOCK_ID", 520))


class AprilTagDetectorWrapper(GenericDetector):
    """
    Wrapper over AprilTagPoseEstimator class to detect QR code and estimate marker pose wrt camera

    NOTE: Can only detect april tag from spot's dock

    How to use:
        1. Create an instance of this class
        2. Call `process_frame` method with image as input
        3. Call `get_outputs` method to get the processed image and pose data

    Example:
        # Create an instance of AprilTagDetectorWrapper
        atdw = AprilTagDetectorWrapper()

        # Initialize detector
        outputs = atdw._init_april_tag_detector(focal_lengths, principal_point)

        # Process image frame
        updated_img_frame, camera_T_marker = atdw.process_frame(img_frame)

        # Get outputs
        updated_img_frame, outputs = atdw.get_outputs(img_frame, outputs, base_T_marker, timestamp, img_metadata)
    """

    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger("AprilTagDetectorWrapper")

    def _init_april_tag_detector(
        self,
        focal_lengths: Tuple[float, float],
        principal_point: Tuple[float, float],
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Initialize April tag detector object for pose estimation of QR code

        NOTE: Can only detect april tag from spot's dock

        Args:
            focal_lengths (Tuple[float, float]) : Focal lengths of camera
            principal_point (Tuple[float, float]) : Principal point of camera
            verbose (bool) : If True, modifies image frame to render detected QR code

        Returns:
            Dict[str, Any] : Dictionary of outputs
            Manipulates following keys in the outputs dictionary:
                - tag_base_T_marker_list (List[sp.SE3]) : List of base_T_marker (Sophus SE3) poses where
                                                          base can be "spot" for Spot or "device" for aria
                - tag_image_list (List[np.ndarray]) : List of decorated image frames
                - tag_image_metadata_list (List[Any]) : List of image metadata
        """
        self.enable_detector()
        self.verbose = verbose

        calib_dict = {
            "fx": focal_lengths[0],
            "fy": focal_lengths[1],
            "ppx": principal_point[0],
            "ppy": principal_point[1],
            "coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        }

        self._qr_pose_estimator = AprilTagPoseEstimator(
            camera_intrinsics=calib_dict, verbose=verbose
        )
        self._qr_pose_estimator.register_marker_ids([DOCK_ID])  # type:ignore

        # Initialize output dictionary
        outputs: Dict[str, Any] = {}
        outputs["tag_base_T_marker_list"] = []
        outputs["tag_image_list"] = []
        outputs["tag_image_metadata_list"] = []
        return outputs

    def process_frame(
        self,
        img_frame: np.ndarray,
    ) -> Tuple[np.ndarray, sp.SE3]:
        """
        Process image frame to detect QR code and estimate marker pose wrt camera

        Args:
            img_frame (np.ndarray) : Image frame to process

        Returns:
            updated_img_frame (np.ndarray) : Image frame with detections and text for visualization
            camera_T_marker (sp.SE3) : SE3 matrix representing marker frame as detected in camera frame
        """
        # Do nothing if detector is not enabled
        if self.is_enabled is False:
            self._logger.warning(
                "AprilTag detector is disabled... Skipping processing of current frame."
            )
            return img_frame, None

        # Detect QR code and estimate marker pose wrt camera
        updated_img_frame, camera_T_marker = self._qr_pose_estimator.detect_markers_and_estimate_pose(  # type: ignore
            image=img_frame
        )

        return updated_img_frame, camera_T_marker

    def get_outputs(
        self,
        img_frame: np.ndarray,
        outputs: Dict,
        base_T_marker: sp.SE3,
        timestamp: int,
        img_metadata: Any,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Update the outputs dictionary with the processed image frame and pose data

        Args:
            img_frame (np.ndarray) : Image frame to process
            outputs (Dict) : Dictionary of outputs (to be updated)
            base_T_marker (sp.SE3) : SE3 matrix representing marker frame as detected in device frame
            img_metadata (Any) : Image metadata

        Returns:
            img_frame (np.ndarray) : Image frame with detections and text for visualization
            outputs (Dict) : Updated dictionary of outputs
            Manipulates following keys in the outputs dictionary:
                - tag_base_T_marker_list (List[sp.SE3]) : List of base_T_marker (Sophus SE3) poses where
                                                          base can be "spot" for Spot or "device" for aria
                - tag_image_list (List[np.ndarray]) : List of decorated image frames
                - tag_image_metadata_list (List[Any]) : List of image metadata

        """
        # Decorate image with text for visualization
        img = decorate_img_with_text_for_qr(
            img=img_frame,
            frame_name_str="device",
            qr_position=base_T_marker.translation(),
        )
        self._logger.debug(f"Time stamp with AprilTag Detections- {timestamp}")

        # Append data to lists for return
        outputs["tag_image_list"].append(img)
        outputs["tag_base_T_marker_list"].append(base_T_marker)
        if img_metadata is not None:
            outputs["tag_image_metadata_list"].append(img_metadata)

        return img, outputs
