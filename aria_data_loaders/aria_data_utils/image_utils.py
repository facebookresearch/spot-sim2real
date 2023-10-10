import math

import numpy as np


def rotate_point(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.

    modified from answer here: https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return int(qx), int(qy)


def calculate_score(frame, object_id, box=None):
    """calculates the detection and recognition score for an object
    Returns two metrics:
    - Binary metric: detected object exists in the frame
    - IOU metric: (if object exists, box!=None) detected object overlaps with the ground truth object
    """
    object_pixels = np.sum(frame["segmentation"] == object_id)
    print(object_pixels)
    binary_score, iou_score = 0, -1.0
    if object_pixels > 5000:
        binary_score = 1
        if box is not None:
            # calculate iou
            gt_box = frame["2dbbox"][object_id].box_range
            # convert both boxes to same convention
            x1, x2, y1, y2 = gt_box
            breakpoint()
            angle = math.radians(90)
            center = (frame["rgb"].shape[1] // 2, frame["rgb"].shape[0] // 2)
            x1, y1 = rotate_point(center, (x1, y1), angle)
            x2, y2 = rotate_point(center, (x2, y2), angle)
            gt_box = [x1, y1, x2, y2]
            iou_score = calculate_iou(box, gt_box)
    return binary_score, iou_score


def calculate_iou(box1, box2):
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
