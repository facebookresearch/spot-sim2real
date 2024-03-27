from typing import List, Tuple

import cv2
import numpy as np
import rospy
from spot_rl.utils.heuristic_nav import (
    ImageSearch,
    get_3d_point,
    get_best_uvz_from_detection,
)
from spot_sim2real_integrationfn import Spot, image_response_to_cv2

image_searcher: ImageSearch = None


def plot_bbox_on_image(
    rgb_img_vis: np.ndarray, bbox_xyxy_conf, uv: Tuple[float, float]
) -> np.ndarray:
    bbox_xyxy_conf = map(int, bbox_xyxy_conf)
    x1_int, y1_int, x2_int, y2_int, _ = bbox_xyxy_conf
    rgb_img_vis = cv2.rectangle(
        rgb_img_vis,
        (x1_int, y1_int),
        (x2_int, y2_int),
        color=(0, 0, 255),
        thickness=2,
    )
    rgb_img_vis = cv2.circle(
        rgb_img_vis,
        (int(uv[0]), int(uv[1])),
        7,
        (0, 0, 255),
        thickness=-1,
    )

    return rgb_img_vis


def get_3d_point_given_image_resps(
    image_resps, object_target: str, depth_scale=None
) -> Tuple[np.ndarray, np.ndarray]:
    images = [image_response_to_cv2(image_resp) for image_resp in image_resps]
    image, depth = images
    print(
        f"Depth_scale from image response source_name : {image_resps[1].source.name}, depth_Scale: {image_resps[1].source.depth_scale}"
    )
    depth_scale = (
        depth_scale if depth_scale else 1.0 / image_resps[1].source.depth_scale
    )
    intrinsics = image_resps[0].source.pinhole.intrinsics
    det_status, bbox_xyxy_conf = image_searcher.object_detection(image, object_target)
    if det_status:
        (u, v), z = get_best_uvz_from_detection(
            depth, bbox_xyxy_conf[:-1], depth_scale=depth_scale
        )
        image = plot_bbox_on_image(image, bbox_xyxy_conf, (u, v))
        return get_3d_point(intrinsics, [u, v], z), image
    return np.array([0.0, 0.0, 0.0], dtype=np.float32), image


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "-ot",
        "--object-target",
        help="input:string -> target object to detect",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-s",
        "--src-feed",
        help="input:string -> Image Source Feed intel/gripper",
        required=True,
        type=str,
    )
    options = parser.parse_args()

    image_searcher = ImageSearch(use_yolov8=False)
    spot: Spot = Spot()
    rospy.set_param("is_gripper_blocked", 1 if "intel" in options.src_feed else 0)
    # get either intel or gripper images
    image_resps = spot.get_hand_image()
    point_in_local_3d, image = get_3d_point_given_image_resps(
        image_resps, options.object_target
    )
    cv2.imwrite(f"{options.src_feed}_Image_Detection.png", image)
    print(f"Point in local 3d for image {point_in_local_3d}")
