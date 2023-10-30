import math
from typing import List, Tuple, Union

import cv2
import numpy as np


def rotate_pixel_coords(
    origin: Tuple[int, int],
    point: Tuple[int, int],
    angle: float,
) -> Tuple[int, int]:
    """
    Rotate a pixel-index counterclockwise by a given angle around a given origin.
    The angle should be given in radians.

    modified from answer here: https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return int(qx), int(qy)


def calculate_score(
    frame, object_id: int, box: list = None, pixel_thresh=5000
) -> Tuple[int, float]:
    """calculates the detection and recognition score for an object
    Returns two metrics:
    - Binary metric: detected object exists in the frame
    - IOU metric: (if object exists, box!=None) detected object overlaps with the ground truth object
    """
    object_pixels = np.sum(frame["segmentation"] == object_id)
    print(object_pixels)
    binary_score, iou_score = 0, -1.0
    if object_pixels > pixel_thresh:
        binary_score = 1
        if box is not None:
            # calculate iou
            gt_box = frame["2dbbox"][object_id].box_range
            # convert both boxes to same convention
            x1, x2, y1, y2 = gt_box
            angle = math.radians(90)
            img_center = (frame["rgb"].shape[1] // 2, frame["rgb"].shape[0] // 2)
            x1, y1 = rotate_pixel_coords(img_center, (x1, y1), angle)
            x2, y2 = rotate_pixel_coords(img_center, (x2, y2), angle)
            gt_box = [x1, y1, x2, y2]
            iou_score = calculate_iou(box, gt_box)
    return binary_score, iou_score


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """calculates the intersection over union score for two boxes"""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    x5, y5 = max(x1, x3), max(y1, y3)
    x6, y6 = min(x2, x4), min(y2, y4)
    intersection = max(0, x6 - x5) * max(0, y6 - y5)
    union = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - intersection
    iou = intersection / union
    assert iou <= 1.0
    assert iou >= 0.0
    return iou.item()


def centered_heuristic(
    detections: list, pixel_wt: float = 0.65, image_size: Tuple[int, int] = (512, 512)
) -> List[float]:
    """Given one instance of object detection from OWL-ViT, calculate the graspability
    score.
    graspability_score = w1 * (%_of_object_pixels) + w2 * (dist_bbox_from_img_center)
    """
    center_wt = 1 - pixel_wt
    scores = []
    for det in detections:
        xmin, ymin, xmax, ymax = det[2][0], det[2][1], det[2][2], det[2][3]
        bbox_center = ((xmin + xmax) // 2, (ymin + ymax) // 2)
        img_center = (image_size[0] // 2, image_size[1] // 2)
        dist_from_center = np.linalg.norm(np.array(bbox_center) - np.array(img_center))
        dist_from_center = 1 - dist_from_center / np.linalg.norm(np.array(img_center))
        bbox_area = (xmax - xmin) * (ymax - ymin)
        img_area = image_size[0] * image_size[1]
        percent_object_pixels = bbox_area / img_area
        score = pixel_wt * percent_object_pixels + center_wt * dist_from_center
        scores.append(score.item())
    return scores


def check_bbox_intersection(bbox1: List[int], bbox2: List[int]) -> bool:
    """Given two bounding boxes described in pixel convention, returns True
    if they intersect, False if they do not"""
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2
    # Check for intersection
    if xmax1 < xmin2 or xmax2 < xmin1 or ymax1 < ymin2 or ymax2 < ymin1:
        # No intersection
        return False
    else:
        # Intersection
        return True


def decorate_img_with_text(img: np.ndarray, frame_name: str, position: np.ndarray):
    """
    Helper method to label image with text

    Args:
        img (np.ndarray): Image to be labeled
        frame (str): Frame of reference (for labeling)
        position (np.ndarray): Position of object in frame of reference
    """
    label_img(img, "Detected QR Marker", (50, 50), color=(0, 0, 255))
    label_img(img, f"Frame = {frame_name}", (50, 75), color=(0, 0, 255))
    label_img(img, f"X : {position[0]}", (50, 100), color=(0, 0, 255))
    label_img(img, f"Y : {position[1]}", (50, 125), color=(0, 250, 0))
    label_img(img, f"Z : {position[2]}", (50, 150), color=(250, 0, 0))

    return img


def label_img(
    img: np.ndarray,
    text: str,
    org: tuple,
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.8,
    color: tuple = (0, 0, 255),
    thickness: int = 2,
    line_type: int = cv2.LINE_AA,
):
    """
    Helper method to label image with text

    Args:
        img (np.ndarray): Image to be labeled
        text (str): Text to be labeled
        org (tuple): (x,y) position of text
        font_face (int, optional): Font face. Defaults to cv2.FONT_HERSHEY_SIMPLEX.
        font_scale (float, optional): Font scale. Defaults to 0.8.
        color (tuple, optional): Color of text. Defaults to (0, 0, 255).
        thickness (int, optional): Thickness of text. Defaults to 2.
        line_type (int, optional): Line type. Defaults to cv2.LINE_AA.
    """
    cv2.putText(
        img,
        text,
        org,
        font_face,
        font_scale,
        color,
        thickness,
        line_type,
    )
