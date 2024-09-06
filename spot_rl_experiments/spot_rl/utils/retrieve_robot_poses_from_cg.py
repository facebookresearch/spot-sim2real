import gzip
import os
import os.path as osp
import pickle
import shutil
from typing import List

import cv2
import numpy as np
import zmq
from scipy.spatial.transform import Rotation as R
from spot_rl.utils.path_planning import get_xyzxyz, plt

# new changes: Select view poses based on distance
# query_class_names = ["furniture", "counter", "locker", "vanity", "wine glass"]  # keep all the class names as same as possible
PATH_TO_CACHE_FILE = osp.join(
    osp.dirname(osp.abspath(__file__)), "scene_map_cfslam_pruned.pkl.gz"
)
PATH_TO_RAW_DATA_PKL = osp.join(osp.dirname(osp.abspath(__file__)), "data.pkl")
VISUALIZE = True
VISUALIZATION_DIR = "image_vis_for_mined_rgb_from_cg"
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


def get_max_diag(bbox_np):
    p0 = bbox_np[0]
    max_diag = -np.inf
    correct_index_pair = -1
    for i, bbox_coord in enumerate(bbox_np[1:]):
        diag = np.linalg.norm(bbox_coord - p0)
        if diag > max_diag:
            max_diag = diag
            correct_index_pair = i + 1
    return max_diag, correct_index_pair


def get_pixel_area(x1, y1, x2, y2):
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    return int(w * h)


socket = None


def connect_socket(port):
    global socket
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://localhost:{port}")
    print(f"Socket Connected at {port}")


def detect_with_yolo(rgb_image, object_classes, visualize=False, port=21001):
    global socket
    if socket is None:
        connect_socket(port)
    socket.send_pyobj((rgb_image, object_classes, visualize))
    bboxes, probs, visualization_img = socket.recv_pyobj()
    if visualize:
        # Show visualization img
        plt.imshow(visualization_img[:, :, ::-1])
        plt.title("Yolo world Detection")
        plt.show()
    if len(bboxes) > 0:
        argmax = int(np.argmax(probs))
        pixel_area = get_pixel_area(*bboxes[argmax])
        return bboxes[argmax], probs[argmax], pixel_area
    return None, None, None


def get_view_poses(
    anchor_object_center,
    anchor_object_extent,
    object_tags,
    query_class_names=[],
    visulize=False,
    visualize_dir=VISUALIZATION_DIR,
):
    data_list = []
    previously_seen_data = {}
    raw_data = None
    with open(PATH_TO_RAW_DATA_PKL, "rb") as f:
        raw_data = pickle.load(f)

    if visulize:
        try:
            shutil.rmtree(visualize_dir)
        except Exception:
            pass
        os.makedirs(visualize_dir, exist_ok=True)

    with gzip.open(PATH_TO_CACHE_FILE, "rb") as f:
        cache_file = pickle.load(f)
        anchorboxMin, anchorboxMax = get_xyzxyz(
            anchor_object_center, anchor_object_extent
        )
        dist_thresh = (
            2.0  # np.linalg.norm(anchorboxMax[:2] - anchorboxMin[:2]) / 2 + 0.5
        )
        print(f"Using Dist thresh {dist_thresh}")
        for i, object_item in enumerate(cache_file):
            class_names = object_item["class_name"]

            # intersection_with_search_query = list(
            #     set(query_class_names) & set(class_names)
            # )
            min_dist = np.inf
            # for class_i, class_name in enumerate(class_names):
            # breakpoint()
            bbox_np = object_item["bbox_np"]
            max_diag, correct_pair_index = get_max_diag(bbox_np)
            center = (bbox_np[0] + bbox_np[correct_pair_index]) / 2.0
            dist_to_anchor_center = np.linalg.norm(center - anchor_object_center)

            # bbox_center_from_caption_field = object_item["caption_dict"]["bbox_center"]
            min_dist = min(min_dist, dist_to_anchor_center)
            if (
                dist_to_anchor_center < dist_thresh
            ):  # and object_item["caption_dict"]["response"]["object_tag"] == object_tags[0]:
                print(f"Detected classes around given receptacle {set(class_names)}")
                # if class_name in query_class_names
                for class_i in range(len(object_item["color_path"])):
                    rgb_path = object_item["color_path"][class_i]
                    conf = object_item["conf"][class_i]
                    index_in_raw_data = get_index_in_raw_data(rgb_path)
                    add_data_flag = True
                    bbox = object_item["xyxy"][class_i]
                    pixel_area = object_item["pixel_area"][class_i]

                    # yoloworld detection
                    raw_image = raw_image_in_data = raw_data[index_in_raw_data][
                        "camera_data"
                    ][0]["raw_image"]
                    bbox, conf, pixel_area = detect_with_yolo(
                        raw_image, object_tags, visualize=False
                    )

                    if bbox is None:
                        continue

                    if index_in_raw_data not in previously_seen_data:
                        previously_seen_data[index_in_raw_data] = {
                            "max_conf": conf,
                            "seen_at": len(data_list),
                        }
                    else:
                        prev_conf = previously_seen_data[index_in_raw_data]["max_conf"]
                        if conf > prev_conf:
                            previously_seen_data[index_in_raw_data] = {
                                "max_conf": conf,
                                "seen_at": len(data_list),
                            }
                        else:
                            add_data_flag = False

                    # print(f"Raw path {osp.basename(rgb_path)}, index in raw {index_in_raw_data}")
                    if add_data_flag:
                        data = {
                            "conf": conf,
                            "bbox": bbox,
                            "pixel_area": pixel_area,
                            "rgb_path": rgb_path,
                            "index_in_raw_data": index_in_raw_data,
                            "robot_xy_yaw": raw_data[index_in_raw_data][
                                "base_pose_xyt"
                            ],
                        }
                        # print(f'Yaw Gripper {yaw_gripper}, Yaw BAse {np.rad2deg(raw_data[index_in_raw_data]["base_pose_xyt"][-1])}')
                        if visulize:
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
        print(f"Found {len(data_list)} instances for given bbox centers & object tag")
        # with open("robot_view_poses_for_bedroom_dresser.pkl", "wb") as file:
        #     pickle.dump(data_list, file)
    else:
        print(f"No such class was found, min distance to anchor object is {min_dist}")

    return data_list


if __name__ == "__main__":
    get_view_poses(ANCHOR_OBJECT_CENTER, [], [], [])
