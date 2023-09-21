from typing import List, Tuple

import cv2.aruco as aruco
import fairotag as frt
import magnum as mn
import sophus as sp

# MAYBE: we dont need spot
from spot_wrapper.spot import Spot

MARKER_LENGTH = 0.146


# TODO: MAYBE This file should be moved to a different package?????
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
        image, mn_camera_T_marker = atpo.detect_markers_and_estimate_pose(image, should_render=True)
    """

    def __init__(self, camera_intrinsics: dict, marker_length: float = MARKER_LENGTH):
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

    @staticmethod
    def _validate_camera_intrinsics(camera_intrinsics: dict):
        _required_keys = ["fx", "fy", "ppx", "ppy", "coeffs"]
        return all(keys in camera_intrinsics for keys in _required_keys)

    def register_marker_ids(self, marker_ids: List[int]):
        assert all(
            isinstance(marker_id, int) for marker_id in marker_ids
        ), "Marker ID must be an integer"

        for marker_id in marker_ids:
            if marker_id not in self._registered_marker_ids:
                self._registered_marker_ids.append(marker_id)
                self._cam_module.register_marker_size(marker_id, self._marker_length)
            else:
                print(f"Marker ID {marker_id} is already registered .. skipping")

    # TODO: If arguments are passed by reference in python, then we can avoid returning image????
    def detect_markers_and_estimate_pose(
        self, image, should_render=False, magnum: bool = True
    ) -> Tuple[object, mn.Matrix4]:
        markers = self._cam_module.detect_markers(image)

        if len(markers) == 0:
            return image, None

        # Currently only one marker is supported @TODO: Make it support multiple markers????
        # assert len(markers) <= 1, "More than one marker detected"
        if len(markers) >= 1:
            print("More than one marker detected")

        marker = markers[0]
        if marker.pose is None:
            return image, None

        # Pose of marker in camera frame expressed as SE3 (in Sophus Library)
        sp_camera_T_marker = marker.pose
        mn_camera_T_marker = Spot.convert_transformation_from_sophus_to_magnum(
            sp_camera_T_marker
        )

        if should_render:
            image = self._cam_module.render_markers(image, markers=[marker])

        if magnum:
            return image, mn_camera_T_marker
        return image, sp_camera_T_marker
