# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import time
from copy import deepcopy
from glob import glob

import cv2
import magnum
import magnum as mn
import numpy as np
import rospy
from scipy import stats as st

# from spot_rl.envs.skill_manager import SpotSkillManager
from spot_rl.models import OwlVit
from spot_rl.models.yolov8predictor import YOLOV8Predictor
from spot_rl.utils.mask_rcnn_utils import get_deblurgan_model
from spot_rl.utils.pixel_to_3d_conversion_utils import *
from spot_rl.utils.utils import construct_config
from spot_wrapper.spot import Spot, image_response_to_cv2, scale_depth_img


def get_z_offset_by_corner_detection(
    rgb_depth_mixed_image, unscaled_dep_img, ball_detection, z
):
    """
    Apply Harris Corner Detection on rgb X depth image, then filter corners such that we select one ahead of our bounding box
    """
    # center_x = (ball_detection[0] + ball_detection[2])/2
    # center_y = (ball_detection[1] + ball_detection[3])/2
    # bottom_center_x, bottom_center_y = center_x, ball_detection[3]
    gray = cv2.cvtColor(rgb_depth_mixed_image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)
    # Threshold for an optimal value, it may vary depending on the image.
    corners_yxs = np.argwhere(dst > 0.01 * dst.max()).reshape(-1, 2)

    zs = unscaled_dep_img[corners_yxs[:, 0], corners_yxs[:, 1]] * 0.001
    non_zero_indices = np.argwhere(zs > 0.0).flatten()
    corners_yxs = corners_yxs[non_zero_indices]
    zs = zs[non_zero_indices]
    if len(corners_yxs) > 0:
        # point_3ds = get_3d_points(imgs[1].source.pinhole.intrinsics, corners_yxs, zs)
        indices_closer = np.argwhere(zs < z).flatten()
        zs = zs[indices_closer]
        corners_yxs = corners_yxs[indices_closer]
        if len(corners_yxs) > 0:
            # y_limit = get_3d_point(intrinsics, (bottom_center_x, bottom_center_y), z)[1]
            # point_3ds = get_3d_points(intrinsics, corners_yxs, zs)

            indices_below = np.argwhere(
                corners_yxs[:, 0] > ball_detection[-1]
            ).flatten()
            zs = zs[indices_below]
            corners_yxs = corners_yxs[indices_below]
            # point_3ds = point_3ds[indices_below]
            if len(corners_yxs) > 0:
                # project bottom 2D points straight
                indices_in_x_limits = np.argwhere(
                    np.logical_and(
                        (ball_detection[0] <= corners_yxs[:, 1]),
                        (corners_yxs[:, 1] <= ball_detection[2]),
                    )
                ).flatten()
                if len(indices_in_x_limits) > 0:
                    zs = zs[indices_in_x_limits]
                    corners_yxs = corners_yxs[indices_in_x_limits]
                    # point_3ds = point_3ds[indices_in_x_limits]
                    mu, std = st.norm.fit(zs)
                    combined_score = 1 * np.absolute(
                        zs - mu
                    )  # + 2*distance_from_object_to_corners

                    # print("Gaussian Depth Distribution for the corners ", mu, std)
                    min_arg = np.argmin(combined_score)
                    best_corner_yx = corners_yxs[min_arg]
                    best_offseted_z = zs[min_arg]
                    return True, best_offseted_z, best_corner_yx, corners_yxs
                else:
                    return False, "couldnot find best corner within x limit", None, None
            else:
                return False, "couldnot find best corner below y limit", None, None
        else:
            return False, "couldnot find best corner closer than z", None, None
    else:
        return False, "couldnot find best non zero depth corner", None, None


def convert_point_from_local_to_global_nav_target(
    point_in_local_3d: np.ndarray,
    direction_local: np.ndarray,
    spot: Spot,
    vision_T_hand: mn.Matrix4,
    body_T_hand: mn.Matrix4,
):

    # point_in_body:mn.Vector3 = body_T_hand.transform_point(mn.Vector3(*point_in_local_3d))

    point_in_global_3d: mn.Vector3 = vision_T_hand.transform_point(
        mn.Vector3(*point_in_local_3d)
    )
    point_in_global_3d = np.array(
        [point_in_global_3d.x, point_in_global_3d.y, point_in_global_3d.z]
    )
    # print(f"Point in global 3d, {point_in_global_3d}")
    theta = np.arctan(point_in_global_3d[1] / point_in_global_3d[0])
    # print(f"theta before transform {np.degrees(theta)}")
    global_x, global_y, transformed_theta = spot.xy_yaw_global_to_home(
        point_in_global_3d[0], point_in_global_3d[1], theta
    )
    # Priyam's logic
    curr_x, curr_y, curr_yaw = spot.get_xy_yaw()
    delta_vec = np.array([global_x, global_y]) - np.array([curr_x, curr_y])
    # norm_delta_vec = np.linalg.norm(delta_vec)
    theta = np.arctan2(delta_vec[1], delta_vec[0])
    print(
        f"transformed theta {np.degrees(transformed_theta)}, new theta {np.degrees(theta)}"
    )
    return (global_x, global_y), theta


def pull_back_point_along_theta_by_offset(
    x: float, y: float, theta: float, offset: float = 0.5
):
    """
    Pulls back the x,y along theta direction for static offset in meters
    """
    x, y = x - offset * np.cos(theta), y - offset * np.sin(theta)
    return (x, y)


def push_forward_point_along_theta_by_offset(
    x: float, y: float, theta: float, offset: float = 0.5
):
    x, y = x + offset * np.cos(theta), y + offset * np.sin(theta)
    return (x, y)


def get_arguments_for_image_search(spot: Spot):
    """
    ImageSearch.search takes a lot of argumnents, this method prepares those
    """
    imgs = spot.get_hand_image()
    rgb_img = image_response_to_cv2(imgs[0])
    unscaled_dep_img = image_response_to_cv2(imgs[1])
    dep_img = scale_depth_img(
        unscaled_dep_img, max_depth=unscaled_dep_img.max() * 0.001, as_img=False
    )
    intrinsics = imgs[0].source.pinhole.intrinsics
    vision_T_hand: mn.Matrix4 = spot.get_magnum_Matrix4_spot_a_T_b(
        "vision", "hand_color_image_sensor", imgs[0].shot.transforms_snapshot
    )
    body_T_hand: mn.Matrix4 = spot.get_magnum_Matrix4_spot_a_T_b(
        "body", "hand_color_image_sensor", imgs[0].shot.transforms_snapshot
    )
    return (
        rgb_img,
        unscaled_dep_img,
        dep_img,
        vision_T_hand,
        body_T_hand,
        intrinsics,
        spot,
    )


class ImageSearch:
    """
    Object Detection Wrapper + offset detection
    Detects object in image using given image detector
    Estimates 3D position of the 2D bounding box in local view
    Estimates nearest corner point using harris corner detection & some geometric filtering (like points below the bounding box)
    Adjust the 3D position found in step 2
    Convert 3D position from hand's local view to x,y, theta global nav target
    """

    def __init__(
        self, corner_static_offset: float = 0.5, use_yolov8=True, visualize=True
    ):
        config = construct_config()
        self.image_scale = float(config.IMAGE_SCALE)
        self.grayscale = config.GRAYSCALE_MASK_RCNN
        self.visualize = visualize
        self.corner_static_offset = corner_static_offset
        self.owlvit = OwlVit([["ball"]], 0.05, False) if not use_yolov8 else None

        # if self.owlvit:
        self.deblur_gan = get_deblurgan_model(config)
        self.yolov8predictor: YOLOV8Predictor = (
            YOLOV8Predictor(
                "/home/tusharsangam/Desktop/spot-sim2real/spot_rl_experiments/weights/torchscript/yolov8x.torchscript"
            )
            if use_yolov8
            else None
        )
        self.normal_object_to_coco_class_id = {"ball": 32.0}

    def preprocess_image(self, img, image_scale):
        if image_scale != 1.0:
            img = cv2.resize(
                img,
                (0, 0),
                fx=self.image_scale,
                fy=self.image_scale,
                interpolation=cv2.INTER_AREA,
            )

        if self.deblur_gan is not None:
            img = self.deblur_gan(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return img

    def object_detection(self, rgb_img, object_target):
        if self.yolov8predictor:
            detections, _ = self.yolov8predictor(
                self.preprocess_image(rgb_img, 1.0), False
            )
            assert (
                object_target in self.normal_object_to_coco_class_id
            ), f"{object_target} not mapped in self.normal_to_coco_class_id for yolov8 detector"
            coco_class_id = self.normal_object_to_coco_class_id[object_target]
            print(detections)
            detections = np.array(
                [d for d in detections if d[-1] == coco_class_id]
            )  # check if detected object is sports ball
        elif self.owlvit:
            self.owlvit.update_label([[object_target]])

            detections, _ = self.owlvit.run_inference_and_return_img(
                self.preprocess_image(rgb_img, self.image_scale), False
            )

            if detections is not None and detections != []:
                detections = np.array(
                    [
                        [*map(float, d[2]), float(d[1]), 0.0]
                        for d in detections
                        if d[0] == object_target
                    ]
                )

        n = len(detections)
        if n > 0:
            max_conf_arg = np.argmax(detections.reshape(n, -1)[:, 4])
            # print(max_conf_arg, detections, detections.shape)
            return True, self.scale_detections_from_owlvit(
                *detections[max_conf_arg, :5].tolist()
            )

        return False, [None, None, None, None, None]

    def scale_detections_from_owlvit(self, x1, y1, x2, y2, conf):
        if self.owlvit:
            # x1, y1, x2, y2, conf = float(x1), float(y1), float(x2), float(y2), float(conf)
            x1, y1, x2, y2 = (
                x1 / self.image_scale,
                y1 / self.image_scale,
                x2 / self.image_scale,
                y2 / self.image_scale,
            )

        return x1, y1, x2, y2, conf

    def search(
        self,
        object_target: str,
        rgb_img: np.ndarray,
        unscaled_depth: np.ndarray,
        hand_depth_img: np.ndarray,
        vision_T_hand: mn.Matrix4,
        body_T_hand: mn.Matrix4,
        cam_intrinsics,
        spot: Spot,
    ):

        rgb_img_vis = rgb_img.copy() if self.visualize else None

        det_status, det = self.object_detection(rgb_img, object_target)
        if not det_status:
            return False, (None, None, None), rgb_img_vis
        x1, y1, x2, y2, conf = det
        x1_int, y1_int, x2_int, y2_int = int(x1), int(y1), int(x2), int(y2)

        (u, v), z = get_best_uvz_from_detection(unscaled_depth, [x1, y1, x2, y2])

        if self.visualize:
            rgb_img_vis = cv2.rectangle(
                rgb_img_vis,
                (x1_int, y1_int),
                (x2_int, y2_int),
                color=(0, 0, 255),
                thickness=2,
            )

        if z >= 0.5 and z <= 2.5:
            binary_depth_img = np.where(hand_depth_img > 0, 1, 0)
            binary_depth_img = np.uint8(binary_depth_img)
            mixed_image = rgb_img * binary_depth_img[:, :, None]
            point_in_local_3d = get_3d_point(cam_intrinsics, (u, v), z)
            (
                corner_det_status,
                best_z,
                best_corner_yx,
                other_best_yxs,
            ) = get_z_offset_by_corner_detection(
                mixed_image, unscaled_depth, [x1, y1, x2, y2], z
            )
            offset = point_in_local_3d[-1] - best_z if corner_det_status else 0
            point_in_local_3d[-1] -= offset + self.corner_static_offset
            (global_x, global_y), theta = convert_point_from_local_to_global_nav_target(
                point_in_local_3d, None, spot, vision_T_hand, body_T_hand
            )
            if self.visualize:
                if corner_det_status:
                    rgb_img_vis = cv2.circle(
                        rgb_img_vis,
                        (best_corner_yx[-1], best_corner_yx[0]),
                        7,
                        (0, 0, 255),
                        thickness=-1,
                    )
                    for yx in other_best_yxs:
                        rgb_img_vis = cv2.circle(
                            rgb_img_vis, (yx[-1], yx[0]), 1, (255, 0, 0)
                        )
                org = (0, y1_int - 50)
                text = "({:.2f}, {:.2f}, {:.2f}), O:{:.2f}, Z:{:.2f}".format(
                    global_x, global_y, np.degrees(theta), offset, z
                )
                rgb_img_vis = cv2.putText(
                    rgb_img_vis,
                    text=text,
                    org=org,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.9,
                    color=(0, 0, 255),
                    thickness=2,
                )
            return True, (global_x, global_y, theta), rgb_img_vis

        return False, (None, None, None), rgb_img_vis


def heurisitic_object_search_and_navigation(
    x: float,
    y: float,
    theta: float,
    object_target: str,
    image_search: ImageSearch = None,
    save_cone_search_images: bool = True,
    pull_back: bool = True,
    skillmanager=None,
    angle_start=-90,
    angle_end=110,
    angle_interval=20,
):
    """
    Args:
            x (float): x coordinate of the nav target (in meters) specified in the world frame
            y (float): y coordinate of the nav target (in meters) specified in the world frame
            theta (float): yaw for the nav target (in radians) specified in the world frame
            object_target: str object to search
            image_search : spot_rl.utils.heuristic_nav.ImageSearch, Optional, default=None, ImageSearch (object detector wrapper), if none creates a new one for you uses OwlVit
            save_cone_search_images: bool, optional, default= True, saves image with detections in each search cone
            pull_back : bool, optional, default=True, pulls back x,y along theta direction
            skill_manager: skill_manager object to perform low level skills
        Returns:
            bool: True if navigation was successful, False otherwise, if True you are good to fire .pick metho

    """
    if save_cone_search_images:
        previously_saved_images = glob("imagesearch*.png")
        for f in previously_saved_images:
            os.remove(f)

    if image_search is None:
        image_search = ImageSearch(
            corner_static_offset=0.5,
            use_yolov8=False,
            visualize=save_cone_search_images,
        )

    (x, y) = (
        pull_back_point_along_theta_by_offset(x, y, theta, 0.2) if pull_back else (x, y)
    )
    print(f"Nav targets adjusted on the theta direction ray {x, y, np.degrees(theta)}")

    skillmanager.nav(x, y, theta)
    skillmanager.nav_controller.nav_env.enable_nav_by_hand()

    spot: Spot = skillmanager.spot
    spot.open_gripper()
    gaze_arm_angles = deepcopy(skillmanager.pick_config.GAZE_ARM_JOINT_ANGLES)
    spot.set_arm_joint_positions(np.deg2rad(gaze_arm_angles), 1)
    time.sleep(1.5)
    found, (x, y, theta), visulize_img = image_search.search(
        object_target, *get_arguments_for_image_search(spot)
    )
    rate = angle_interval  # control time taken to rotate the arm, higher the rotation higher is the time
    if not found:
        # start semi circle search
        semicircle_range = np.arange(angle_start, angle_end, angle_interval)
        for i_a, angle in enumerate(semicircle_range):
            print(f"Searching in {angle} cone")
            angle_time = int(np.abs(gaze_arm_angles[0] - angle) / rate)
            gaze_arm_angles[0] = angle
            spot.set_arm_joint_positions(np.deg2rad(gaze_arm_angles), angle_time)
            time.sleep(1.5)
            (
                found,
                (x, y, theta),
                visulize_img,
            ) = image_search.search(  # type : ignore
                object_target, *get_arguments_for_image_search(spot)
            )
            if save_cone_search_images:
                cv2.imwrite(f"imagesearch_{angle}.png", visulize_img)
            if found:
                print(f"In Cone Search object found at {(x,y,theta)}")
                break

    else:
        if save_cone_search_images:
            cv2.imwrite("imagesearch_looking_forward.png", visulize_img)
    angle_time = int(
        np.abs(gaze_arm_angles[0] - skillmanager.pick_config.GAZE_ARM_JOINT_ANGLES[0])
        / rate
    )
    spot.set_arm_joint_positions(
        np.deg2rad(skillmanager.pick_config.GAZE_ARM_JOINT_ANGLES),
        angle_time,
    )
    if found:
        print(f"Nav goal after cone search {x, y, np.degrees(theta)}")
        # backup_steps = skillmanager.nav_controller.nav_env.max_episode_steps
        # skillmanager.nav_controller.nav_env.max_episode_steps = 50
        skillmanager.nav(x, y, theta)
        # skillmanager.nav_controller.nav_env.max_episode_steps = backup_steps
    skillmanager.nav_controller.nav_env.disable_nav_by_hand()
    return found


if __name__ == "__main__":
    print(
        "Please see the example in spot_rl_experiments/experiments/skill_test/test_spot_to_aria.py"
    )
