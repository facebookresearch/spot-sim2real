import os
from typing import Any, Dict, Tuple

import numpy as np
import sophus as sp
from aria_data_utils.detector_wrappers.generic_detector_interface import GenericDetector
from aria_data_utils.image_utils import decorate_img_with_text
from aria_data_utils.perception.april_tag_pose_estimator import AprilTagPoseEstimator

DOCK_ID = int(os.environ.get("SPOT_DOCK_ID", 520))


class AprilTagDetectorWrapper(GenericDetector):
    def __init__(self):
        super().__init__()

    def _init_april_tag_detector(
        self, focal_lengths: Any, principal_point: Any
    ) -> Dict[str, Any]:
        """
        Initialize April tag detector object for pose estimation of QR code

        Can only detect dock code
        """
        assert self.is_enabled is True

        # focal_lengths = self._dst_calib_params.get_focal_lengths()  # type:ignore
        # principal_point = self._dst_calib_params.get_principal_point()  # type:ignore
        calib_dict = {
            "fx": focal_lengths[0].item(),
            "fy": focal_lengths[1].item(),
            "ppx": principal_point[0].item(),
            "ppy": principal_point[1].item(),
            "coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        }

        self._qr_pose_estimator = AprilTagPoseEstimator(camera_intrinsics=calib_dict)
        self._qr_pose_estimator.register_marker_ids([DOCK_ID])  # type:ignore
        outputs: Dict[str, Any] = {}
        outputs["tag_device_T_marker_list"] = []
        outputs["tag_image_list"] = []
        outputs["tag_image_metadata_list"] = []
        return outputs

    def process_frame(
        self, img_frame: np.ndarray, verbose: bool = True
    ) -> Tuple[np.ndarray, sp.SE3]:
        assert self.is_enabled is True

        return self._qr_pose_estimator.detect_markers_and_estimate_pose(  # type: ignore
            image=img_frame, should_render=verbose, magnum=False
        )

    def get_outputs(
        self,
        img_frame: np.ndarray,
        outputs: Dict,
        device_T_marker,
        img_metadata: Any,
    ):
        # Decorate image with text for visualization
        img = decorate_img_with_text(
            img=img_frame,
            frame_name="device",
            position=device_T_marker.translation(),
        )
        print(f"Time stamp with Detections- {img_metadata.capture_timestamp_ns}")

        # Append data to lists for return
        outputs["tag_image_list"].append(img)
        outputs["tag_image_metadata_list"].append(img_metadata)
        outputs["tag_device_T_marker_list"].append(device_T_marker)

        return img_frame, outputs
