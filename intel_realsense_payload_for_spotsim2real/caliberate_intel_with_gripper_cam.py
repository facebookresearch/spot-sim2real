import cv2
import numpy as np
import rospy
from perception_and_utils.utils.image_utils import decorate_img_with_text_for_qr

try:
    import sophuspy as sp
except Exception as e:
    print(f"Cannot import sophuspy due to {e}. Import sophus instead")
    import sophus as sp
# from perception_and_utils.perception.detector_wrappers.april_tag_detector import (
#     AprilTagDetectorWrapper,
# )
from april_tag_pose_detector import AprilTagDetectorWrapper
from spot_wrapper.spot import Spot, SpotCamIds, image_response_to_cv2

intel_img_src = [SpotCamIds.INTEL_REALSENSE_COLOR, SpotCamIds.INTEL_REALSENSE_DEPTH]  # type: ignore
gripper_img_src = [SpotCamIds.HAND_COLOR, SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME]


def get_intel_image(spot: Spot):
    return spot.get_image_responses(intel_img_src, quality=100, await_the_resp=True)  # type: ignore


def get_gripper_image(spot: Spot):
    return spot.get_image_responses(gripper_img_src, quality=100, await_the_resp=True)  # type: ignore


if __name__ == "__main__":
    spot: Spot = Spot("Calibration")

    intel_response = get_intel_image(spot)[0]
    intel_intrinsics = intel_response.source.pinhole.intrinsics

    gripper_response = get_gripper_image(spot)[0]
    gripper_intrinsics = gripper_response.source.pinhole.intrinsics

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
        intel_images = get_intel_image(spot)
        gripper_images = get_gripper_image(spot)
        intel_images = [
            image_response_to_cv2(intel_image) for intel_image in intel_images
        ]
        gripper_images = [
            image_response_to_cv2(gripper_image) for gripper_image in gripper_images
        ]

        intel_image, intel_T_marker = aprilposeestimator_intel.process_frame(
            *intel_images, allow_depth_correction=True
        )

        gripper_image, gripper_T_marker = aprilposeestimator_gripper.process_frame(
            *gripper_images, allow_depth_correction=True
        )
        (
            gripper_image_without_depth,
            gripper_T_marker_without_depth,
        ) = aprilposeestimator_gripper.process_frame(
            *gripper_images, allow_depth_correction=False
        )
        (
            intel_image_without_depth,
            intel_T_marker_without_depth,
        ) = aprilposeestimator_intel.process_frame(
            *intel_images, allow_depth_correction=False
        )

        if intel_T_marker is None or gripper_T_marker is None:
            continue
        gripper_image = decorate_img_with_text_for_qr(
            img=gripper_image,
            frame_name_str="gripper",
            qr_position=gripper_T_marker.translation(),
        )
        gripper_image_without_depth = decorate_img_with_text_for_qr(
            img=gripper_image_without_depth,
            frame_name_str="gripperwithoutdepth",
            qr_position=gripper_T_marker_without_depth.translation(),
        )
        intel_image = decorate_img_with_text_for_qr(
            img=intel_image,
            frame_name_str="intel",
            qr_position=intel_T_marker.translation(),
        )
        intel_image_without_depth = decorate_img_with_text_for_qr(
            img=intel_image_without_depth,
            frame_name_str="intelwithoutdepth",
            qr_position=intel_T_marker_without_depth.translation(),
        )
        cv2.imshow(
            "QR detection",
            np.hstack(
                (
                    intel_image_without_depth,
                    intel_image,
                    gripper_image_without_depth,
                    gripper_image,
                )
            ),
        )

        marker_T_intel = sp.invert_poses(intel_T_marker.matrix3x4().ravel()).reshape(
            (3, 4)
        )
        marker_T_intel = sp.SE3(marker_T_intel[:3, :3], marker_T_intel[:3, 3])
        gripper_T_marker *= marker_T_intel
        gripper_T_intel = gripper_T_marker.matrix()
        # breakpoint()
        err = gripper_T_intel - prev_diff
        print(f"error {err} \n")
        prev_diff = gripper_T_intel
        save_tf = rospy.get_param("is_save", 0) == 1
        if save_tf and not np.any(err[:3, :3]):
            np.save("gripper_T_intel.npy", gripper_T_intel)
            break
        cv2.waitKey(1)
