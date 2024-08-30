import gzip
import os
import os.path as osp
import pickle
from typing import List

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

#new changes: Select view poses based on distance
#query_class_names = ["furniture", "counter", "locker", "vanity", "wine glass"]  # keep all the class names as same as possible
PATH_TO_CACHE_FILE = "scene_map_cfslam_pruned.pkl.gz"
PATH_TO_RAW_DATA_PKL = "data.pkl"
VISUALIZE = False
VISUALIZATION_DIR = "image_vis"
ANCHOR_OBJECT_CENTER = np.array([8.2, 6.0, 0.1])


def resize_crop(masked_rgb_image, x1, y1, x2, y2):
    # Crop the image
    cropped_image = masked_rgb_image[y1:y2, x1:x2]

    # Get the height and width of the cropped image
    crop_height, crop_width = cropped_image.shape[:2]

    # Calculate the aspect ratio of the crop (target is 640x480, which is 4:3)
    target_aspect_ratio = 640 / 480
    current_aspect_ratio = crop_width / crop_height

    # Adjust the crop to maintain the 4:3 aspect ratio
    if current_aspect_ratio > target_aspect_ratio:
        # The crop is too wide, adjust the width
        new_width = int(crop_height * target_aspect_ratio)
        width_diff = crop_width - new_width
        x1_new = width_diff // 2
        cropped_image = cropped_image[:, x1_new : x1_new + new_width]
    else:
        # The crop is too tall, adjust the height
        new_height = int(crop_width / target_aspect_ratio)
        height_diff = crop_height - new_height
        y1_new = height_diff // 2
        cropped_image = cropped_image[y1_new : y1_new + new_height, :]

    # Resize the cropped image to 480x640
    resized_image = cv2.resize(cropped_image, (640, 480))
    return resized_image


def get_index_in_raw_data(rgb_path: str) -> int:
    return int(osp.basename(rgb_path).split(".")[0][5:])


def draw_bbox_with_confidence(image, bbox, confidence):
    """
    Draw a bounding box on the image and annotate it with confidence.

    Parameters:
    image (numpy.ndarray): Image array in HxWxC format (color image).
    bbox (tuple): Bounding box coordinates as (x1, y1, x2, y2).
    confidence (float): Confidence score to be displayed on the bounding box.

    Returns:
    numpy.ndarray: Image with the bounding box and confidence label drawn.
    """
    # Ensure the image is in color (HxWx3)
    if len(image.shape) == 2:  # If grayscale, convert to color
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Unpack bounding box coordinates
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Draw the bounding box
    color = (0, 255, 0)  # Green color for the box
    thickness = 2
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Prepare the label with confidence
    label = f"{confidence:.2f}"

    # Choose a font
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    # Calculate the size of the label
    (label_width, label_height), baseline = cv2.getTextSize(
        label, font, font_scale, font_thickness
    )

    # Position for the label (above the bounding box)
    label_x = x1
    label_y = y1 - 10 if y1 - 10 > 10 else y1 + 10

    # Draw the background rectangle for the label
    cv2.rectangle(
        image,
        (label_x, label_y - label_height - baseline),
        (label_x + label_width, label_y + baseline),
        color,
        -1,
    )

    # Draw the label text
    cv2.putText(
        image, label, (label_x, label_y), font, font_scale, (0, 0, 0), font_thickness
    )

    return image


if __name__ == "__main__":
    data_list = []
    previously_seen_data = {}
    raw_data = None

    with open(PATH_TO_RAW_DATA_PKL, "rb") as f:
        raw_data = pickle.load(f)

    with gzip.open(PATH_TO_CACHE_FILE, "rb") as f:
        cache_file = pickle.load(f)
        for i, object_item in enumerate(cache_file):
            class_names = object_item["class_name"]
            
            # intersection_with_search_query = list(
            #     set(query_class_names) & set(class_names)
            # )
            if True: #len(intersection_with_search_query) > 0:  # and len(set(class_names)) == 1:
                # breakpoint()
                for class_i, class_name in enumerate(class_names):
                    bbox_np = object_item["bbox_np"]
                    boxMin = np.array([bbox_np[:, 0].min(), bbox_np[:, 1].min(), bbox_np[:, -1].min()])
                    boxMax = np.array([bbox_np[:, 0].max(), bbox_np[:, 1].max(), bbox_np[:, -1].max()])
                    center = (boxMin + boxMax)/2.
                    dist_to_anchor_center = np.linalg.norm(center - ANCHOR_OBJECT_CENTER)   
                    
                    if dist_to_anchor_center < 0.1: 
                        print(set(class_names))
                        # if class_name in query_class_names
                        rgb_path = object_item["color_path"][class_i]
                        conf = object_item["conf"][class_i]
                        index_in_raw_data = get_index_in_raw_data(rgb_path)
                        add_data_flag = True
                        
                        if index_in_raw_data not in previously_seen_data:
                            previously_seen_data[index_in_raw_data] = {
                                "max_conf": conf,
                                "seen_at": len(data_list),
                            }
                        else:
                            prev_conf = previously_seen_data[index_in_raw_data][
                                "max_conf"
                            ]
                            if conf > prev_conf:
                                previously_seen_data[index_in_raw_data] = {
                                    "max_conf": conf,
                                    "seen_at": len(data_list),
                                }
                            else:
                                add_data_flag = False

                        # print(f"Raw path {osp.basename(rgb_path)}, index in raw {index_in_raw_data}")
                        if add_data_flag:
                            # base_T_camera_from_raw = raw_data[index_in_raw_data]["camera_data"][0]["base_T_camera"]

                            # vision_T_base = raw_data[index_in_raw_data]["vision_T_base"]

                            # vision_T_camera = R.from_matrix((vision_T_base@base_T_camera_from_raw)[:3, :3])
                            # yaw_gripper = vision_T_camera.as_euler("ZYX", True)[-1]
                            #print(type(raw_data[index_in_raw_data]["base_pose_xyt"]))
                            data = {
                                "conf": conf,
                                "bbox": object_item["xyxy"][class_i],
                                "pixel_area": object_item["pixel_area"][class_i],
                                "rgb_path": rgb_path,
                                "index_in_raw_data": index_in_raw_data,
                                "robot_xy_yaw": raw_data[index_in_raw_data][
                                    "base_pose_xyt"
                                ],
                            }
                            # print(f'Yaw Gripper {yaw_gripper}, Yaw BAse {np.rad2deg(raw_data[index_in_raw_data]["base_pose_xyt"][-1])}')
                            if VISUALIZE:
                                os.makedirs(VISUALIZATION_DIR, exist_ok=True)
                                raw_image_in_data = raw_data[index_in_raw_data][
                                    "camera_data"
                                ][0]["raw_image"]
                                image_vis = draw_bbox_with_confidence(
                                    raw_image_in_data, data["bbox"], data["conf"]
                                )
                                cv2.imwrite(
                                    osp.join(
                                        VISUALIZATION_DIR,
                                        f"comparison_{len(data_list)}.png",
                                    ),
                                    image_vis,
                                )
                            data_list.append(data)

    if len(data_list) > 0:
        print(f"Found {len(data_list)} instances for given bbox centers")
        with open("robot_view_poses_for_bedroom_dresser.pkl", "wb") as file:
            pickle.dump(data_list, file)
    else:
        print("No such class was found")
