# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rospy
import sophus as sp
from bosdyn.client.frame_helpers import get_a_tform_b
from perception_and_utils.perception.detector_wrappers.april_tag_detector import (
    AprilTagDetectorWrapper,
)
from perception_and_utils.utils.image_utils import decorate_img_with_text_for_qr
from perception_and_utils.utils.math_utils import get_running_avg_a_T_b
from scipy.spatial.transform import Rotation
from spot_rl.utils.utils import ros_frames as rf
from spot_wrapper.spot import (
    Spot,
    SpotCamIds,
    SpotCamIdToFrameNameMap,
    image_response_to_cv2,
)

DOCK_ID = int(os.environ.get("SPOT_DOCK_ID", 520))
FILTER_DIST = 2.4  # in meters (distance for valid detection)


class SpotQRDetector:
    """
    Used for  Spot's cameras' pose estimation using external QR marker (Spot's dock's QR)
    IMP: PLEASE Install fairotag as `pip install -e fairotag` as there needs to be some changes in its source code

    This class will only detect the QR code of spot's base as set in bashrc (SPOT_DOCK_ID)

    Args:
        spot: Spot object
        cam_ids: List of SpotCamIds for which QR detector should be initialized

    How to use:
        1. Create an instance of this class with appropriate cam ids
        2. Call get_avg_spotWorld_T_marker with a single cam_id (blocking call)
        3. Call `detect_markers_and_estimate_pose` method with image as input

    Example:
        # Create an instance of AprilTagPoseEstimator
        sqrd = SpotQRDetector(spot, cam_ids=[cam_id])

        # Compute average world_T_marker
        avg_spotWorld_T_marker = sqrd.get_avg_spotWorld_T_marker(cam_id= SpotCamIds.cam_id)

        # Compute instantaneous transformation between 2 cameras via QR marker
        camera1_T_camera2 = sqrd.get_camera1_T_camera2(cam_id_1=SpotCamIds.cam_id1,cam_id_2=SpotCamIds.cam_id2)

    """

    def __init__(self, spot: Spot, cam_ids: List[SpotCamIds] = [SpotCamIds.HAND_COLOR]):
        self.spot = spot
        print("...Spot initialized...")

        # Verify camera sources
        for cam_id in cam_ids:
            if cam_id == SpotCamIds.HAND or "_depth" in cam_id:
                raise ValueError(
                    f"SpotQRDetector cannot work for depth cameras. Invalid CamId - {cam_id}"
                )

        # Insert HAND_COLOR source if not present
        if SpotCamIds.HAND_COLOR not in cam_ids:
            cam_ids.append(SpotCamIds.HAND_COLOR)

        # Create detector for each camera
        self.april_tag_detector_dict = {
            cam_id: AprilTagDetectorWrapper() for cam_id in cam_ids
        }

        # Get camera intrinsics
        cam_intrinsics_dict = {
            cam_id: self.spot.get_camera_intrinsics([cam_id])[0] for cam_id in cam_ids
        }

        # Initialize output dictionary
        self.outputs_dict: Dict[SpotCamIds, Dict[str, Any]] = {
            cam_id: {} for cam_id in cam_ids
        }

        # Initialize April Tag Pose Estimator for all requested cameras
        self.outputs_dict = self.initialize_april_tag_detectors(
            cam_intrinsics_dict, self.outputs_dict
        )

    # TODO: Move to spot.py as `get_camera_intrinsics_as_3x3` as a part of PR #143
    def _to_camera_metadata_dict(self, camera_intrinsics):
        """Converts a camera intrinsics proto to a 3x3 matrix as np.array"""
        fx = (camera_intrinsics.focal_length.x,)
        fy = (camera_intrinsics.focal_length.y,)
        ppx = (camera_intrinsics.principal_point.x,)
        ppy = (camera_intrinsics.principal_point.y,)
        intrinsics = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]])
        return intrinsics

    def initialize_april_tag_detectors(
        self,
        cam_intrinsics_dict: Dict[SpotCamIds, Any],
        outputs_dict: Dict[SpotCamIds, Dict[str, Any]],
    ):
        """
        Initialize the april tag detectors for all requested cameras

        Args:
            outputs_list (List[Dict], optional): List of dictionary of outputs from each camera's april tag detectors. Defaults to [].

        Updates:
            - self.april_tag_detectors_list: List of AprilTagDetectorWrapper object, for each camera

        Returns:
            outputs_list (list[dict]): List of dictionary of outputs from the april tag detector, each dict contains the following keys:
                - "tag_image_list" - List of np.ndarrays of images with detections
                - "tag_image_metadata_list" - List of image metadata
                - "tag_base_T_marker_list" - List of Sophus SE3 transforms from base frame to marker
                                             where base is "device" frame for aria
                                             and "body" frame for spot
        """
        # Sanity checks
        assert len(cam_intrinsics_dict.keys()) == len(
            outputs_dict.keys()
        ), "Length of cam_intrinsics_dict and outputs_dict should be same"
        assert len(cam_intrinsics_dict.keys()) == len(
            self.april_tag_detector_dict.keys()
        ), "Length of cam_intrinsics_dict and april_tag_detector_dict should be same"

        # Initialize the april tag detectors for all requested cameras
        for cam_id in cam_intrinsics_dict:
            focal_lengths = (
                cam_intrinsics_dict[cam_id].focal_length.x,
                cam_intrinsics_dict[cam_id].focal_length.y,
            )  # type: Tuple[float, float]

            principal_point = (
                cam_intrinsics_dict[cam_id].principal_point.x,
                cam_intrinsics_dict[cam_id].principal_point.y,
            )  # type: Tuple[float, float]

            outputs_dict[cam_id].update(
                self.april_tag_detector_dict[cam_id]._init_april_tag_detector(
                    focal_lengths=focal_lengths,
                    principal_point=principal_point,
                    verbose=False,
                )
            )
        return outputs_dict

    def get_avg_spotWorld_T_marker(
        self,
        cam_id: SpotCamIds = SpotCamIds.HAND_COLOR,
        use_vision_as_spotWorld: bool = True,
        filter_dist: float = FILTER_DIST,
        fixed_data_length: int = 10,
    ):
        """
        Returns the average transformation of spot world frame to marker frame

        We get a camera_T_marker for each frame in which marker is detected.
        Depending on the frame rate of image capture, multiple frames may have captured the marker.
        Averaging all transforms would be best way to compensate for any noise that may exist in any frame's detections
        camera_T_marker is used to compute spot_T_marker[i] and thus spotWorld_T_marker[i].
        Then we average all spotWorld_T-marker to find average marker pose wrt spotWorld.
        """
        spot_world_frame = rf.SPOT_WORLD_VISION
        if not use_vision_as_spotWorld:
            spot_world_frame = rf.SPOT_WORLD_ODOM

        if cam_id not in self.april_tag_detector_dict.keys():
            raise ValueError(
                f"SpotQRDetector was not initialized for camera id - {cam_id}"
            )

        cv2.namedWindow(f"image for {cam_id}", cv2.WINDOW_AUTOSIZE)

        # Maintain a list of all poses where qr code is detected (w.r.t spotWorld)
        marker_positions_list = (
            []
        )  # type: List[np.ndarray] # List of  position as np.ndarray (x, y, z)
        marker_quaternion_list = (
            []
        )  # type: List[np.ndarray] # List of quaternions as np.ndarray (x, y, z, w)
        avg_spotWorld_T_marker = None  # type: Optional[sp.SE3]

        while len(marker_positions_list) < fixed_data_length:
            print(f"Iterating - {len(marker_positions_list)}")
            # Set camera_T_marker as None
            camera_T_marker = None

            # Obtain rgb image from list of BD ImageResponse objects as per specified cam_id
            # More info on BD ImageResponse object can be found here -
            # https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference#bosdyn-api-ImageResponse
            img_response = self.spot.get_image_responses([cam_id])[0]
            img = image_response_to_cv2(img_response)

            # Detect Marker in the image
            (viz_img, camera_T_marker) = self.april_tag_detector_dict[
                cam_id
            ].process_frame(img_frame=img)

            if camera_T_marker is not None:
                # Spot_T_marker computation
                frame_tree_snapshot = img_response.shot.transforms_snapshot
                spot_T_camera = self.spot.get_sophus_SE3_spot_a_T_b(
                    frame_tree_snapshot, rf.SPOT_BODY, SpotCamIdToFrameNameMap[cam_id]
                )
                spot_T_marker = spot_T_camera * camera_T_marker
                spotWorld_T_spot = self.spot.get_sophus_SE3_spot_a_T_b(
                    frame_tree_snapshot, spot_world_frame, rf.SPOT_BODY
                )

                # Frame: a = spotWorld
                # Frame: intermediate = spot (spot base frame)
                # Frame: b = marker / qr code
                (
                    marker_positions_list,
                    marker_quaternion_list,
                    avg_spotWorld_T_marker,
                ) = get_running_avg_a_T_b(
                    current_avg_a_T_b=avg_spotWorld_T_marker,
                    a_T_b_position_list=marker_positions_list,
                    a_T_b_quaternion_list=marker_quaternion_list,
                    a_T_intermediate=spotWorld_T_spot,
                    intermediate_T_b=spot_T_marker,
                    filter_dist=filter_dist,
                    fixed_data_length=fixed_data_length,
                )

                (viz_img, self.outputs_dict[cam_id],) = self.april_tag_detector_dict[
                    cam_id
                ].get_outputs(
                    img_frame=viz_img,
                    outputs=self.outputs_dict[cam_id],
                    base_T_marker=spot_T_marker,
                    timestamp=None,
                    img_metadata=None,
                )

            cv2.imshow(f"Spot QR Detector - {cam_id} Cam", viz_img)
            cv2.waitKey(1)

        return avg_spotWorld_T_marker

    def get_camera1_T_camera2(self, cam_id_1: SpotCamIds, cam_id_2: SpotCamIds):
        """
        Get the instantaneous transform of camera_2 in the frame of camera_1

        Args:
            cam_id_1 (SpotCamIds): Camera ID of camera_1
            cam_id_2 (SpotCamIds): Camera ID of camera_2

        Returns:
            camera_1_T_camera_2 (sp.SE3): The transform from camera1 to camera2 as a Sophus SE3 object
        """

        cv2.namedWindow("Spot QR Detector - Camera1", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Spot QR Detector - Camera2", cv2.WINDOW_NORMAL)
        # Detect Marker in both images
        camera1_T_marker = None
        camera2_T_marker = None
        while camera1_T_marker is None and camera2_T_marker is None:
            # Get the images from both cameras
            [img_response1, image_response2] = self.spot.get_image_responses(
                [cam_id_1, cam_id_2]
            )

            # Extract images from the image responses
            img1 = image_response_to_cv2(img_response1)
            img2 = image_response_to_cv2(image_response2)

            (viz_img1, camera1_T_marker,) = self.april_tag_detector_dict[
                cam_id_1
            ].process_frame(img_frame=img1)
            cv2.imshow("Spot QR Detector - Camera1", viz_img1)

            if camera1_T_marker is None:
                print("No marker detected in Camera_1. Trying again...")

            (viz_img2, camera2_T_marker,) = self.april_tag_detector_dict[
                cam_id_2
            ].process_frame(img_frame=img2)
            cv2.imshow("Spot QR Detector - Camera2", viz_img2)

            cv2.waitKey(1)
            if camera2_T_marker is None:
                print("No marker detected in Camera_2. Trying again...")

            # If no marker is detected in either camera, reset the transforms
            if camera1_T_marker is None or camera2_T_marker is None:
                camera1_T_marker = None
                camera2_T_marker = None

        # Get the transforms from the image responses
        camera1_T_camera2 = camera1_T_marker * camera2_T_marker.inverse()

        print("Compute camera1_T_camera2")
        print(
            "Dist between camera1 and camera2: ",
            np.linalg.norm(camera1_T_camera2.translation()),
        )
        return camera1_T_camera2


if __name__ == "__main__":
    spot = Spot("QRDetector")
    cam_ids = [SpotCamIds.FRONTLEFT_FISHEYE, SpotCamIds.HAND_COLOR]
    sqrd = SpotQRDetector(spot, cam_ids=cam_ids)
    avg_vision_T_marker = sqrd.get_avg_spotWorld_T_marker(cam_id=SpotCamIds.HAND_COLOR)
    cam1_T_cam2 = sqrd.get_camera1_T_camera2(
        SpotCamIds.FRONTLEFT_FISHEYE, SpotCamIds.HAND_COLOR
    )
