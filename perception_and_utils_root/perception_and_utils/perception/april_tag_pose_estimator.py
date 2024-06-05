# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Tuple

import cv2.aruco as aruco
import fairotag as frt
import numpy as np

try:
    import sophuspy as sp
except Exception as e:
    print(f"Cannot import sophuspy due to {e}. Import sophus instead")
    import sophus as sp
from perception_and_utils.utils.generic_utils import conditional_print

MARKER_LENGTH = 0.146


class AprilTagPoseEstimator:
    """
    Used for AprilTag detection and pose estimation
    IMP: PLEASE Install fairotag as `pip install -e fairotag` as there needs to be some changes in its source code

    Args:
        camera_intrinsics: Camera intrinsics dictionary
        marker_length: Length of the marker

    How to use:
        1. Create an instance of this class
        2. Register marker IDs using `register_marker_ids` method
        3. Call `detect_markers_and_estimate_pose` method with image as input

    Example:
        # Create an instance of AprilTagPoseEstimator
        atpo = AprilTagPoseEstimator(camera_intrinsics)

        # Register marker IDs
        atpo.register_marker_ids([1, 2, 3])

        # Detect markers and estimate pose
        viz_image, mn_camera_T_marker = atpo.detect_markers_and_estimate_pose(image)
    """

    def __init__(
        self,
        camera_intrinsics: dict,
        marker_length: float = MARKER_LENGTH,
        verbose: bool = True,
    ):
        apriltag_dict = aruco.Dictionary_get(aruco.DICT_APRILTAG_36h11)
        self._cam_module = frt.CameraModule(dictionary=apriltag_dict)

        assert AprilTagPoseEstimator._validate_camera_intrinsics(
            camera_intrinsics
        ), "Invalid camera intrinsics"

        # Set camera intrinsics
        self._cam_module.set_intrinsics(frt.utils.dict2intrinsics(camera_intrinsics))

        # Set marker length
        self._marker_length = marker_length

        # Registere marker IDs
        self._registered_marker_ids = []  # type: List[int]

        # Verbose
        self._verbose = verbose

    @staticmethod
    def _validate_camera_intrinsics(camera_intrinsics: dict) -> bool:
        """
        Ensures the camera instrinsics being passed contains all required fields
        Must contain - ["fx", "fy", "ppx", "ppy", "coeffs"]

        Arg:
            camera_intrinsics (Dict) : Camera intrinsics dictionary

        Returns:
            bool : True only if all fields are present, False otherwise
        """
        _required_keys = ["fx", "fy", "ppx", "ppy", "coeffs"]
        return all(keys in camera_intrinsics for keys in _required_keys)

    def register_marker_ids(self, marker_ids: List[int]):
        """
        Register the markers for detector to detect.
        Detector will only recognise registered markers.

        Args:
            marker_ids (List[int]): List of markers ids (int) to register
        """
        assert all(
            isinstance(marker_id, int) for marker_id in marker_ids
        ), "Marker ID must be an integer"

        for marker_id in marker_ids:
            if marker_id not in self._registered_marker_ids:
                self._registered_marker_ids.append(marker_id)
                self._cam_module.register_marker_size(marker_id, self._marker_length)
            else:
                conditional_print(
                    f"Marker ID {marker_id} is already registered .. skipping",
                    self._verbose,
                )

    def detect_markers_and_estimate_pose(
        self, image: np.ndarray
    ) -> Tuple[object, sp.SE3]:
        """
        Detect all registered markers in the given image frame
        ONLY supports 1 marker at this time, only considers first
        detection and ignores rest

        Args:
            image (np.ndarray) : Image on which markers need to be detected

        Returns:
            viz_image (np.ndarray) : Updated image after marking detection if marker is detected
            camera_T_marker (sp.SE3) : SE3 matrix representing marker frame as detected in camera frame
        """
        markers = self._cam_module.detect_markers(image)

        if len(markers) == 0:
            return image, None

        # Currently only one marker is supported
        if len(markers) >= 1:
            conditional_print("More than one marker detected", self._verbose)

        marker = markers[0]
        if marker.pose is None:
            return image, None

        # Pose of marker in camera frame expressed as SE3 (in Sophus Library)
        camera_T_marker = marker.pose

        # Render marker on image
        viz_image = self._cam_module.render_markers(image, markers=[marker])

        return viz_image, camera_T_marker
