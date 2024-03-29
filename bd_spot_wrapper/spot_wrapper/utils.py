# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import subprocess

import cv2
import numpy as np
import quaternion


def say(text):
    try:
        text = text.replace("_", " ")
        text = f'"{text}"'
        cmd = f"/usr/bin/festival -b '(voice_cmu_us_slt_arctic_hts)' '(SayText {text})'"
        subprocess.Popen(cmd, shell=True)
    except Exception:
        pass
    print(f'Saying: "{text}"')


def resize_to_tallest(imgs, hstack=False):
    tallest = max([i.shape[0] for i in imgs])
    for idx, i in enumerate(imgs):
        height, width = i.shape[:2]
        if height != tallest:
            new_width = int(width * (tallest / height))
            imgs[idx] = cv2.resize(i, (new_width, tallest))
    if hstack:
        return np.hstack(imgs)
    return imgs


def inflate_erode(mask, size=50):
    mask_copy = mask.copy()
    mask_copy = cv2.blur(mask_copy, (size, size))
    mask_copy[mask_copy > 0] = 255
    mask_copy = cv2.blur(mask_copy, (size, size))
    mask_copy[mask_copy < 255] = 0

    return mask_copy


def erode_inflate(mask, size=20):
    mask_copy = mask.copy()
    mask_copy = cv2.blur(mask_copy, (size, size))
    mask_copy[mask_copy < 255] = 0
    mask_copy = cv2.blur(mask_copy, (size, size))
    mask_copy[mask_copy > 0] = 255

    return mask_copy


def contour_mask(mask):
    cnt, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    new_mask = np.zeros(mask.shape, dtype=np.uint8)
    max_area = 0
    max_index = 0
    for idx, c in enumerate(cnt):
        area = cv2.contourArea(c)
        if area > max_area:
            max_area = area
            max_index = idx
    cv2.drawContours(new_mask, cnt, max_index, 255, cv2.FILLED)

    return new_mask


def color_bbox(img, just_get_bbox=False):
    """Makes a bbox around a white object"""
    # Filter out non-white
    sensitivity = 80
    upper_white = np.array([255, 255, 255])
    lower_white = upper_white - sensitivity
    color_mask = cv2.inRange(img, lower_white, upper_white)

    # Filter out little bits of white
    color_mask = inflate_erode(color_mask)
    color_mask = erode_inflate(color_mask)

    # Only use largest contour
    color_mask = contour_mask(color_mask)

    # Calculate bbox
    x, y, w, h = cv2.boundingRect(color_mask)

    if just_get_bbox:
        return x, y, w, h

    height, width = color_mask.shape
    cx = (x + w / 2.0) / width
    cy = (y + h / 2.0) / height

    # Create bbox mask
    bbox_mask = np.zeros([height, width, 1], dtype=np.float32)
    bbox_mask[y : y + h, x : x + w] = 1.0

    # Determine if bbox intersects with central crosshair
    crosshair_in_bbox = x < width // 2 < x + w and y < height // 2 < y + h

    return bbox_mask, cx, cy, crosshair_in_bbox


def angle_between_quat(q1, q2):
    """A function that returns the angle between two quaternions"""
    q1_inv = np.conjugate(q1)
    dp = quaternion.as_float_array(q1_inv * q2)
    return 2 * np.arctan2(np.linalg.norm(dp[1:]), np.abs(dp[0]))


def get_angle_between_two_vectors(x, y):
    """Get angle between two vectors"""
    if np.linalg.norm(x) != 0:
        x_norm = x / np.linalg.norm(x)
    else:
        x_norm = x

    if np.linalg.norm(y) != 0:
        y_norm = y / np.linalg.norm(y)
    else:
        y_norm = y

    return np.arccos(np.clip(np.dot(x_norm, y_norm), -1, 1))


def get_angle_between_forward_and_target(rel_pos):
    """Get the angle bewtween forward and target vectors"""
    forward = np.array([1.0, 0, 0])
    rel_pos = np.array(rel_pos)
    forward = forward[[0, 1]]
    rel_pos = rel_pos[[0, 1]]

    heading_angle = get_angle_between_two_vectors(forward, rel_pos)
    c = np.cross(forward, rel_pos) < 0
    if not c:
        heading_angle = -1.0 * heading_angle
    return heading_angle
