import argparse
import time
from collections import deque
import pickle

import cv2
import numpy as np
from numpy.linalg import inv
from spot_wrapper.spot import (
    Spot,
    SpotCamIds,
    draw_crosshair,
    image_response_to_cv2,
    scale_depth_img
)


import sys
sys.path.append("/Users/jimmytyyang/HomeRobot/home-robot/src")
import os
from home_robot.agent.perception.detection.lseg import load_lseg_for_inference
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
checkpoint_path = "/Users/jimmytyyang/LSeg/checkpoints/demo_e200.ckpt"
DEVICE = torch.device("cpu")
lseg_model = load_lseg_for_inference(checkpoint_path, DEVICE)


SAVE_IMG_DIR = "/Users/jimmytyyang/Documents/spot_sem_map_walk_demo_v4/" # The saving directory


def get_lseg_det_bbox(det):
    """This function is only used for the lseg detection method.
    It only selects the class in the middle of the image."""
    # The input det size is (H, W, 2)
    # Transform the one hot image into the black and white image
    det_array = det.numpy()
    # Convert one-hot encodings into integers
    # binary_mask only contains 0 for "other" and 1 for the target text
    # And its size is (H, W)
    binary_mask = np.argmax(det_array, axis=2)
    # Expand the size to be (H, W, 1)
    grey = np.expand_dims(binary_mask, axis=2) * 255.0
    cv2.imshow("lseg_grey", grey)
    cy = grey.shape[0] // 2
    cx = grey.shape[1] // 2

    # Get the bounding box
    contours = cv2.findContours(grey.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # Loop over the candiate, find the one that happends cover the center
    # of the image
    # https://stackoverflow.com/questions/63923800/drawing-bounding-rectangles-around-multiple-objects-in-binary-image-in-python
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        # Transform them into array index with row and col
        x1, y1, x2, y2 = x, y, x + w, y + h
        if x1 < cx < x2 and y1 < cy < y2:
            return x1, y1, x2, y2
    # Nothing in the center
    return None

def main(episode_i, img_i):
    with open(SAVE_IMG_DIR + "data_" + str(episode_i) + ".pkl", 'rb') as handle:
        img_rgbs, img_depths, delta_x_y_raws = pickle.load(handle)

    print("Loaded the image")
    # Get the first image as a testing example
    img = img_rgbs[img_i]

    cv2.imshow("raw_rgb", img)
    # batch as a leading dim
    img = torch.unsqueeze(torch.tensor(img), 0)

    # Encode the rgb features
    # Pixel_features: (batch_size, 512, H, W)
    pixel_features = lseg_model.encode(img)

    # Get the prediction and the visualizations
    # one_hot_predictions: (batch_size, H, W, len(labels))
    # visualizations:  (batch_size, H, W, 3)
    one_hot_predictions, visualizations = lseg_model.decode(pixel_features, ["other", "pick up"])
    cv2.imshow("lseg_rgb", visualizations[0])

    bbox_xy = get_lseg_det_bbox(one_hot_predictions[0])

    print("bbox_xy:", bbox_xy)
    cv2.waitKey(1)
    import pdb; pdb.set_trace()

    return

if __name__ == "__main__":
    exmaple_dict = {}
    exmaple_dict["cup"] = (70, 9)
    exmaple_dict["door"] = (20, 5)
    exmaple_dict["random"] = (120, 9)
    target_text = "random"
    episode_i = exmaple_dict[target_text][0]
    img_i = exmaple_dict[target_text][1]
    main(episode_i, img_i)
