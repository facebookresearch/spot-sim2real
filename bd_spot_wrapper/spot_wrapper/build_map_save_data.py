"""
This is the code for saving the trajectory of Spot's pose (x, y, yaw) and RGB and depth images during
exploration in the environment.

For every program mentioned here, first, you do "source ~/.bash_profile" and then you do "conda activate spot_env".
Once we finished this, then we can do the following steps to collect the trajectory and build the map asynchronously:

Step 1. Launch the e-step: "python -m spot_wrapper.estop"

Step 2. Keyboard control of Spot: "python -m spot_wrapper.keyboard_teleop"

Step 3-1. Collect the trajectory: "python -m spot_wrapper.build_map_save_data"

Step 3-2. Build the map: "python -m spot_wrapper.build_map_gen_map"

Note that Step 3-1 and Step 3-2 can do at the same time since build_map_gen_map.py will asynchronously read the folder of the saved trajectory.

Note that build_map.py is a version that combines build_map_save_data.py and build_map_gen_map.py but sequentially collects the trajectory and builds the map.
build_map.py's trajectory will skip the robot's certain pose at the particular timestamp sometimes due to the delay in building the map. We recommend that we should
use the asynchronous version.

The step before launching the mamba env:  /Users/jimmytyyang/mambaforge/bin/mamba init zsh
"""

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
    scale_depth_img,
    clip_depth_img
)
from spot_wrapper.utils.utils import color_bbox
from spot_wrapper.utils.utils import resize_to_tallest_old_version as resize_to_tallest

import sys
sys.path.append("/Users/jimmytyyang/HomeRobot/home-robot/src")
import os
from home_robot.agent.perception.detection.lseg import load_lseg_for_inference
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
checkpoint_path = "/Users/jimmytyyang/LSeg/checkpoints/demo_e200.ckpt"
DEVICE = torch.device("cpu")
model = load_lseg_for_inference(checkpoint_path, DEVICE)

MAX_HAND_DEPTH = 3.0 # in meter
MAX_HEAD_DEPTH = 10.0 # in meter
DETECT_LARGEST_WHITE_OBJECT = False
# This is useful when transforming the front left/right depth into rgb for PIXEL_FORMAT
PIXEL_FORMAT_RGB_U8 = "PIXEL_FORMAT_RGB_U8"

# Define the label
TEXT_LABELS = [
    "computer",
    "robot",
    "chair",
    "human",
    "table",
    "cabinet",
    "mug",
    "clothing",
    "bottle",
    "box",
    "door",
    "other",
]
# Define Metadata for plotting the text using cv2
# Font
font = cv2.FONT_HERSHEY_SIMPLEX
# FontScale
fontScale = 1
# Color in RGB
color = (255, 255, 255)
# Line thickness of 2 px
thickness = 2

# Add the semantic map.
from PIL import Image

UPDATE_SEM_MAP_EVERY_LEN = 10
SAVE_IMG_DIR = "/Users/jimmytyyang/Downloads/build_map/0223_run2_kitchen"

# Create the save folder
os.makedirs(SAVE_IMG_DIR, exist_ok=True)
SAVE_IMG_DIR += "/"
assert SAVE_IMG_DIR[-1] == "/"


class spot_pose_tracker():
    """Keep track of the robot pose between each update."""
    def __init__(self, spot):
        self.spot = spot
        self.local_T_global = None
        self.global_T_home = None
        self.robot_recenter_yaw = None

    def update_origin(self, x, y, yaw):
        """Update the transformation of the pose given current pose (i.e., x, y, yaw)."""
        self.local_T_global = self.spot._get_local_T_global(x, y, yaw)
        self.global_T_home = np.linalg.inv(self.local_T_global)
        self.robot_recenter_yaw = yaw

    def get_delta_pose(self, x, y, yaw):
        """Locate the delta local pose of Spot given the transformation."""
        return self.xy_yaw_global_to_home(x, y, yaw)

    def xy_yaw_global_to_home(self, x, y, yaw):
        """Do the transformation given the transformation matrix. The code is from spot.py."""
        x, y, w = self.global_T_home.dot(np.array([x, y, 1.0]))
        x, y = x / w, y / w

        return x, y, self.wrap_heading(yaw - self.robot_recenter_yaw)

    def wrap_heading(self, heading):
        """Ensure input heading is between -180 an 180; can be float or np.ndarray."""
        return (heading + np.pi) % (2 * np.pi) - np.pi


def main(spot: Spot):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--no-display", action="store_true")
    parser.add_argument("-q", "--quality", type=int)
    args = parser.parse_args()
    window_name = "Spot Camera Viewer"
    time_buffer = deque(maxlen=10)
    sources = [
        #SpotCamIds.FRONTRIGHT_DEPTH,
        #SpotCamIds.FRONTLEFT_DEPTH,
        #SpotCamIds.FRONTRIGHT_FISHEYE,
        #SpotCamIds.FRONTLEFT_FISHEYE,
        SpotCamIds.HAND_DEPTH,
        SpotCamIds.HAND_COLOR,
    ]
    PIXEL_FORMAT = None
    PIXEL_FORMAT_MAP = None #[None, PIXEL_FORMAT_RGB_U8]
    RGB_LSeg_LIST = [] #[SpotCamIds.HAND_COLOR]
    # Get the rgb and depth for building the map
    sources_for_map = [
        # Aligh the depth image in the rgb frame
        SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME,
        SpotCamIds.HAND_COLOR,
    ]

    try:
        # Set the episode counter
        episode_i = 0
        # Initilize the current position as the global origin position
        spot.home_robot()
        global_T_home = spot.global_T_home.copy()
        # Save the transformation.
        # Get the current pose of the robot
        last_x, last_y, last_yaw = spot.get_xy_yaw(use_boot_origin=True)
        # Initilize spot pose using spot_pose_tracker class
        spot_pose = spot_pose_tracker(spot)
        # Update the origin
        spot_pose.update_origin(last_x, last_y, last_yaw)

        while True:
            # Print the episode
            print("episode:", episode_i)

            # Get the time
            start_time = time.time()

            # Initilize the lists and update the semantic map in every UPDATE_SEM_MAP_EVERY_LEN
            if episode_i % UPDATE_SEM_MAP_EVERY_LEN == 0:
                if episode_i != 0:
                    # Save the data
                    with open(SAVE_IMG_DIR + "data_" + str(episode_i) + ".pkl", 'wb') as handle:
                        pickle.dump([img_rgb, img_depth, delta_x_y_raw], handle)
                else:
                    # Save the transformation
                    with open(SAVE_IMG_DIR + "global_T_home" + ".pkl", 'wb') as handle:
                        # Record the initial location of the robot
                        pickle.dump([global_T_home, last_x, last_y, last_yaw], handle)

                # Init or reset the lists
                img_rgb = []
                img_depth = []
                delta_x_y_raw = []

            # Get Spot camera image for cv2 plot to keep track of observations
            image_responses = spot.get_image_responses(sources, quality=args.quality)
            imgs = []
            rgb_lsegs = []
            for image_response, source in zip(image_responses, sources):
                img = image_response_to_cv2(image_response, reorient=True)
                # Get the information for building the cv2 viewer
                if "depth" in source:
                    max_depth = MAX_HAND_DEPTH if "hand" in source else MAX_HEAD_DEPTH
                    # Scale the depth image
                    img = scale_depth_img(img, max_depth=max_depth, as_img=True)
                elif source is SpotCamIds.HAND_COLOR:
                    # Draw the red circle in the middle
                    img = draw_crosshair(img)
                    if DETECT_LARGEST_WHITE_OBJECT:
                        x, y, w, h = color_bbox(img, just_get_bbox=True)
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Get the lseg image
                if source in RGB_LSeg_LIST:
                    rgb_lsegs.append(img.copy())
                imgs.append(img)

            # Get the LSeg features
            get_text_and_img_pos = []
            for rgb_img in rgb_lsegs:
                rgb = torch.unsqueeze(torch.tensor(rgb_img), 0)
                # Pixel_features: (batch_size, 512, H, W)
                pixel_features = model.encode(rgb)
                # Get the prediction
                one_hot_predictions, visualizations = model.decode(pixel_features, TEXT_LABELS)
                # Get the text
                get_text_and_img_pos.append(model.get_text_and_img_pos()[0])
                imgs.append(visualizations[0])

            # Make sure all imgs are same height
            display_img, width_list = resize_to_tallest(imgs, hstack=True)

            # Get the current Spot's position and rotation
            curr_x, curr_y, curr_yaw = spot.get_xy_yaw(use_boot_origin=True)

            # Get the Spot's local delta pose using custom class
            delta_x, delta_y, delta_yaw = spot_pose.get_delta_pose(curr_x, curr_y, curr_yaw)
            delta_x_y_raw.append(np.array([delta_x, delta_y, delta_yaw]).copy())

            # Get the RGB and depth information for building the map
            image_responses = spot.get_image_responses(sources_for_map, quality=args.quality)
            for image_response, source in zip(image_responses, sources_for_map):
                img = image_response_to_cv2(image_response, reorient=True)
                # Get the RGB and depth information for building the map
                # hand_color_image's size: (480, 640, 3)
                # hand_depth's size (224, 171)
                # Align rgb image in depth frame
                if source is SpotCamIds.HAND_COLOR_IN_HAND_DEPTH_FRAME:
                    img_rgb.append(img.copy())
                    cv2.imshow("rgb image in depth frame", img)
                elif source is SpotCamIds.HAND_DEPTH:
                    img = clip_depth_img(img, max_depth=MAX_HAND_DEPTH)
                    img_depth.append(np.expand_dims(img.copy(), -1))
                    cv2.imshow("depth image", img)
                # Align depth image in rgb frame
                elif source is SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME:
                    img = clip_depth_img(img, max_depth=MAX_HAND_DEPTH)
                    img_depth.append(np.expand_dims(img.copy(), -1))
                    cv2.imshow("depth image in rgb frame", img)
                elif source is SpotCamIds.HAND_COLOR:
                    img_rgb.append(img.copy())
                    cv2.imshow("rgb image", img)

            # Display image
            if not args.no_display:
                # Add the text into the semantic RGB
                for i in range(len(RGB_LSeg_LIST)):
                    # Get the offset for plotting text on the semantic RGB
                    offset = 0
                    for j in range(len(sources)+i):
                        offset += width_list[j]
                    for text_label in get_text_and_img_pos[i]:
                        x = get_text_and_img_pos[i][text_label][0] + int(offset)
                        y = get_text_and_img_pos[i][text_label][1]
                        display_img = cv2.putText(display_img, text_label, (x, y),\
                            font,\
                            fontScale,\
                            color,\
                            thickness,\
                            cv2.LINE_AA)
                cv2.imshow(window_name, display_img)
                cv2.waitKey(1)

            # Update the episode
            episode_i += 1

            # Print the FPS and the pose
            time_buffer.append(time.time() - start_time)
            print("Avg FPS:", 1 / np.mean(time_buffer))

            print("Current pose in glocal frame:", curr_x, curr_y, curr_yaw)
            print("Current pose in local frame:", delta_x, delta_y, delta_yaw)

            delta_xy_global = ((curr_x-last_x)**2+(curr_y-last_y)**2)**0.5
            print("Delta x, y in global frame:", delta_xy_global)
            delta_xy_local = (delta_x**2+delta_y**2)**0.5
            print("Delta x, y in local frame:", delta_xy_local)

            delta_yaw_global = spot_pose.wrap_heading(curr_yaw - last_yaw)
            print("Delta yaw in global frame:", delta_yaw_global)
            print("Delta yaw in local frame", delta_yaw)

            # Terminate if the delta is too huge
            if abs(delta_xy_global-delta_xy_local) >= 2.5 * 1e-2: # 2.5cm differecne passed
                print("Error, delta x, y between global and local frames is too large")
                import pdb; pdb.set_trace()
            if abs(delta_yaw_global-delta_yaw) >= 2.5 * 1e-2: # 0.025 rad differecne passed
                print("Error, delta yaw between global and local frames is too large")
                import pdb; pdb.set_trace()

            # Store the last pose for computing the delta of the global frame in the next round
            last_x, last_y, last_yaw = curr_x, curr_y, curr_yaw
            # Update the origin of local frame using the current position
            spot_pose.update_origin(curr_x, curr_y, curr_yaw)
            # Use the current position as the original position of the robot
            spot.home_robot()

    finally:
        import pdb; pdb.set_trace()
        if not args.no_display:
            cv2.destroyWindow(window_name)


if __name__ == "__main__":
    spot = Spot("ViewCamera")
    # We don't need a lease because we're passively observing images (no motor ctrl)
    main(spot)
