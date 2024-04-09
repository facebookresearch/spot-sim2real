import cv2
import numpy as np
import rospy
import sophuspy as sp
from perception_and_utils.perception.detector_wrappers.april_tag_detector import (
    AprilTagDetectorWrapper,
)
from spot_wrapper.spot import Spot, SpotCamIds, image_response_to_cv2

intel_img_src = [SpotCamIds.INTEL_REALSENSE_COLOR]  # type: ignore
gripper_img_src = [SpotCamIds.HAND_COLOR]


def get_intel_image(spot: Spot):
    return spot.get_image_responses(intel_img_src, quality=100, await_the_resp=False)  # type: ignore


def get_gripper_image(spot: Spot):
    return spot.get_image_responses(gripper_img_src, quality=100, await_the_resp=False)  # type: ignore


if __name__ == "__main__":
    spot: Spot = Spot("Calibration")

    intel_response = get_intel_image(spot)
    intel_response = intel_response.result()[0]
    intel_intrinsics = intel_response.source.pinhole.intrinsics
    intel_image: np.ndarray = image_response_to_cv2(intel_response)

    gripper_response = get_gripper_image(spot)
    gripper_response = gripper_response.result()[0]
    gripper_intrinsics = gripper_response.source.pinhole.intrinsics
    gripper_image: np.ndarray = image_response_to_cv2(gripper_response)

    aprilposeestimator_intel: AprilTagDetectorWrapper = AprilTagDetectorWrapper()
    aprilposeestimator_intel._init_april_tag_detector(
        focal_lengths=[
            intel_intrinsics.focal_length.x,
            intel_intrinsics.focal_length.y,
        ],
        principal_point=[
            intel_intrinsics.principal_point.x,
            intel_intrinsics.principal_point.y,
        ],
        verbose=False,
    )

    aprilposeestimator_gripper: AprilTagDetectorWrapper = AprilTagDetectorWrapper()
    aprilposeestimator_gripper._init_april_tag_detector(
        focal_lengths=[
            gripper_intrinsics.focal_length.x,
            gripper_intrinsics.focal_length.y,
        ],
        principal_point=[
            gripper_intrinsics.principal_point.x,
            gripper_intrinsics.principal_point.y,
        ],
        verbose=False,
    )
    prev_diff = np.zeros((4, 4), dtype=np.float32)
    while True:
        intel_image = get_intel_image(spot)
        gripper_image = get_gripper_image(spot)
        intel_image = image_response_to_cv2(intel_image.result()[0])
        gripper_image = image_response_to_cv2(gripper_image.result()[0])
        intel_image, intel_T_marker = aprilposeestimator_intel.process_frame(
            intel_image
        )
        gripper_image, gripper_T_marker = aprilposeestimator_gripper.process_frame(
            gripper_image
        )
        cv2.imshow("QR detection", np.hstack((intel_image, gripper_image)))
        marker_T_intel = intel_T_marker.inverse()
        gripper_T_intel = (gripper_T_marker * marker_T_intel).matrix()
        err = gripper_T_intel - prev_diff
        print(f"error {err} \n")
        prev_diff = gripper_T_intel
        save_tf = rospy.get_param("is_save", 0) == 1
        if save_tf and not np.any(err):
            np.save("gripper_T_intel.npy", gripper_T_intel)
            break
        cv2.waitKey(1)
