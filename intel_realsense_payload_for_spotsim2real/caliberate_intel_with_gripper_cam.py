import cv2
import numpy as np
import rospy

try:
    import sophuspy as sp
except Exception as e:
    print(f"Cannot import sophuspy due to {e}. Import sophus instead")
    import sophus as sp
from perception_and_utils.perception.detector_wrappers.april_tag_detector import (
    AprilTagDetectorWrapper,
)
from spot_wrapper.spot import Spot, SpotCamIds, image_response_to_cv2

intel_img_src = [SpotCamIds.INTEL_REALSENSE_COLOR]  # type: ignore
gripper_img_src = [SpotCamIds.HAND_COLOR]


MIN_LIN_DIST_THRESHOLD_BETWEEN_INTEL_AND_GRIPPER = (
    0.115  # 13cm is approx physical dist between gripper & intel (13 - 2.5 = 11.5)
)
MAX_LIN_DIST_THRESHOLD_BETWEEN_INTEL_AND_GRIPPER = (
    0.155  # 13cm is approx physical dist between gripper & intel (13 + 2.5 = 15.5)
)


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

    outputs_intel = {}
    aprilposeestimator_intel: AprilTagDetectorWrapper = AprilTagDetectorWrapper()
    outputs_intel = aprilposeestimator_intel._init_april_tag_detector(
        focal_lengths=[
            intel_intrinsics.focal_length.x,
            intel_intrinsics.focal_length.y,
        ],
        principal_point=[
            intel_intrinsics.principal_point.x,
            intel_intrinsics.principal_point.y,
        ],
        verbose=True,
    )

    outputs_gripper = {}
    aprilposeestimator_gripper: AprilTagDetectorWrapper = AprilTagDetectorWrapper()
    outputs_gripper = aprilposeestimator_gripper._init_april_tag_detector(
        focal_lengths=[
            gripper_intrinsics.focal_length.x,
            gripper_intrinsics.focal_length.y,
        ],
        principal_point=[
            gripper_intrinsics.principal_point.x,
            gripper_intrinsics.principal_point.y,
        ],
        verbose=True,
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
        if intel_T_marker is not None:
            intel_image, outputs = aprilposeestimator_intel.get_outputs(
                img_frame=intel_image,
                outputs=outputs_intel,
                base_T_marker=intel_T_marker,
                timestamp=0,
                img_metadata=None,
            )

        gripper_image, gripper_T_marker = aprilposeestimator_gripper.process_frame(
            gripper_image
        )
        if gripper_T_marker is not None:
            gripper_image, outputs = aprilposeestimator_gripper.get_outputs(
                img_frame=gripper_image,
                outputs=outputs_gripper,
                base_T_marker=gripper_T_marker,
                timestamp=0,
                img_metadata=None,
            )

        if intel_T_marker is not None and gripper_T_marker is not None:

            # Make sure euclidean distance between gripper & intelRS is 13cm +/- 2.5 cm . (13cm is the approx physical distance)
            marker_T_intel = intel_T_marker.inverse()
            gripper_T_intel = gripper_T_marker * marker_T_intel
            if (
                np.linalg.norm(gripper_T_intel.matrix()[:3, 3])
                < MAX_LIN_DIST_THRESHOLD_BETWEEN_INTEL_AND_GRIPPER
            ) and (
                np.linalg.norm(gripper_T_intel.matrix()[:3, 3])
                > MIN_LIN_DIST_THRESHOLD_BETWEEN_INTEL_AND_GRIPPER
            ):
                print(
                    f"^^^^^^^^^^^^^^VALID DISTANCE^^^^^^^^^^^^^^^^^  - {np.linalg.norm(gripper_T_intel.matrix()[:3, 3])}"
                )

                # Wait for consistent reading between 2 frames
                err = gripper_T_intel.matrix() - prev_diff
                print(f"error {err} \n")
                prev_diff = gripper_T_intel.matrix()
                save_tf = rospy.get_param("is_save", 0) == 1
                if save_tf and not np.any(err):
                    print("Will Save")
                    np.save("gripper_T_intel.npy", gripper_T_intel.matrix())
                    break
            else:
                print(
                    f"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  INVALID DISTANCE  - {np.linalg.norm(gripper_T_intel.matrix()[:3, 3])}"
                )
        else:
            print("Intel : ", intel_T_marker is None)
            print("Gripper : ", gripper_T_marker is None)

        cv2.imshow("QR detection", np.hstack((intel_image, gripper_image)))
        cv2.waitKey(1)
