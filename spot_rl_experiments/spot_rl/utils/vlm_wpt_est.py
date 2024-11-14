import copy
import os.path as osp
import time

import cv2
import numpy as np
from spot_rl.utils.gripper_t_intel_path import GRIPPER_T_INTEL_PATH
from spot_rl.utils.pixel_to_3d_conversion_utils import get_3d_point
from spot_rl.utils.search_table_location import get_arguments, project_3d_to_pixel_uv
from spot_rl.utils.molmo_pointer import molmo_predict_waypoint
from spot_rl.utils.robopoint_pointer import robopoint_predict_waypoint


def get_robot_data(spot, GAZE_ARM_JOINT_ANGLES):
    assert osp.exists(GRIPPER_T_INTEL_PATH), f"{GRIPPER_T_INTEL_PATH} not found"
    gripper_T_intel = np.load(GRIPPER_T_INTEL_PATH)
    spot.close_gripper()
    gaze_arm_angles = copy.deepcopy(GAZE_ARM_JOINT_ANGLES)
    spot.set_arm_joint_positions(np.deg2rad(gaze_arm_angles), 1)

    # Wait for a bit to stabilize the gripper
    time.sleep(1.5)

    (
        img,
        depth_raw,
        camera_intrinsics_intel,
        camera_intrinsics_gripper,
        body_T_hand,
        gripper_T_intel,
    ) = get_arguments(spot, gripper_T_intel)

    return (
        img,
        depth_raw,
        camera_intrinsics_intel,
        camera_intrinsics_gripper,
        body_T_hand,
        gripper_T_intel,
    )


def convert_to_pixel_coordinates(img, wpt):
    # Get image dimensions
    height, width = img.shape[:2]

    # Convert normalized coordinates to pixel coordinates
    x_pixel = int(wpt[0] * (width - 1))
    y_pixel = int(wpt[1] * (height - 1))
    return x_pixel, y_pixel


def draw_wpt(image, wpt, color=(0, 255, 0), radius=5, thickness=2):
    # Make a copy of the image to avoid modifying the original
    img_draw = image.copy()
    x_pixel, y_pixel = convert_to_pixel_coordinates(img_draw, wpt)
    # Draw a circle at the point
    cv2.circle(img_draw, (x_pixel, y_pixel), radius, color, thickness)
    cv2.imwrite(f"table_detection_vlm_{time.time()*1000}.png", img_draw)


def visualize_reprojected_pt(
    selected_point, img_cv2, camera_intrinsics_intel, visualize=False
):
    selected_xy = project_3d_to_pixel_uv(
        selected_point.reshape(1, 3), camera_intrinsics_intel
    )[0]
    reproj_img = img_cv2.copy()
    reproj_img = cv2.circle(
        img_cv2, (int(selected_xy[0]), int(selected_xy[1])), 2, (0, 0, 255)
    )
    if visualize:
        cv2.imwrite(
            f"table_detection_vlm_after_v2_{time.time() * 1000}.png", reproj_img
        )
    return


def get_3d_point_in_body(
    img_cv2,
    depth_raw,
    avg_wpt,
    body_T_hand,
    gripper_T_intel,
    camera_intrinsics_intel,
    height_adjustment_offset=0.5,
    visualize=False,
):
    h, w = img_cv2.shape[:2]
    z = depth_raw[int(avg_wpt[1] * h), int(avg_wpt[0] * w)] / 1000
    x_pixel, y_pixel = convert_to_pixel_coordinates(img_cv2, avg_wpt)
    selected_point = get_3d_point(camera_intrinsics_intel, [x_pixel, y_pixel], z)
    selected_point_in_gripper = np.array(gripper_T_intel * selected_point)
    visualize_reprojected_pt(
        selected_point, img_cv2, camera_intrinsics_intel, visualize
    )
    point_in_body = body_T_hand * selected_point_in_gripper
    placexyz = np.array(point_in_body)
    # Static Offset adjustment
    placexyz[2] += height_adjustment_offset
    return placexyz


def vlm_predict_3d_waypoint(
    spot,
    place_config,
    GAZE_ARM_JOINT_ANGLES,
    tokenizer,
    model,
    image_processor,
    height_adjustment_offset,
    visualize=True,
):
    (
        img_cv2,
        depth_raw,
        camera_intrinsics_intel,
        camera_intrinsics_gripper,
        body_T_hand,
        gripper_T_intel,
    ) = get_robot_data(spot, GAZE_ARM_JOINT_ANGLES)
    if visualize:
        cv2.imwrite(f"table_detection_vlm_before_{time.time() * 1000}.png", img_cv2)
    start_time = time.time()
    vlm_model_path = place_config.waypoint_estimation_model
    if "robopoint" in vlm_model_path.lower():
        avg_wpt = robopoint_predict_waypoint(img_cv2, tokenizer, model, image_processor)
    elif "molmo" in vlm_model_path.lower():
        avg_wpt = molmo_predict_waypoint(img_cv2, model, image_processor)
    print("inference time: ", time.time() - start_time)
    placexyz = get_3d_point_in_body(
        img_cv2,
        depth_raw,
        avg_wpt,
        body_T_hand,
        gripper_T_intel,
        height_adjustment_offset,
    )

    return placexyz
