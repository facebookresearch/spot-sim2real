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
from spot_wrapper.utils.utils import color_bbox, resize_to_tallest

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
    "soda_can",
    "fridge",
    "chair",
    "human",
    "table",
    "cabinet",
    "mug",
    "snack",
    "bottle",
    "box",
    "sink",
    "other",
]
# Define Metadata for plotting the text using cv2 Font
font = cv2.FONT_HERSHEY_SIMPLEX
# FontScale
fontScale = 1
# Color in RGB
color = (255, 255, 255)
# Line thickness of 2 px
thickness = 2

# Add the semantic map.
from home_robot.agent.mapping.dense.semantic.vision_language_2d_semantic_map_state import VisionLanguage2DSemanticMapState
from home_robot.agent.mapping.dense.semantic.vision_language_2d_semantic_map_module import VisionLanguage2DSemanticMapModule
from PIL import Image
from home_robot.agent.perception.detection.coco_maskrcnn.coco_categories import (
    coco_categories, coco_categories_color_palette, text_label_color_palette
)
# Map parameters
MAP_SIZE_CM = 1000 # The glocal map height and width size in cm
UPDATE_SEM_MAP_EVERY_LEN = 10 # The update frequency.
CAMERA_HEIGHT = 0.925 # for arm camera in meter. standing height 0.61 + arm height 0.35 = 0.96. Iphone app ruler = 0.61 
GLOBAL_DOWNSCALING = 2 # The ratio between the size of global and local maps
VISION_RADIUS_CELL = 125 # The vision radius of Spot in cell unit (hand camera is 3m)
MAP_RESOLUTION = 2 # The resoluation of the map. For example, if MAP_SIZE_CM = 1000 and MAP_RESOLUTION = 2, then global map has the cell size of 1000/2=500.
DU_SCALE = 5 # The point cloud resolution
NUM_POINTS_OBSTACLE = 50 # The number of points to be considered it as obstacles
LSEG_FEATURES_DIM = 512 # The size of the lseg features. I tried 128, 256, and they failed. It seems that 512 is a fixed value
SAVE_IMG_DIR = "/Users/jimmytyyang/Downloads/build_map/0223_run2_kitchen" # The saving directory

RADIUS_EXPLORE = VISION_RADIUS_CELL

# Create the save folder
os.makedirs(SAVE_IMG_DIR, exist_ok=True)
SAVE_IMG_DIR += "/"
assert SAVE_IMG_DIR[-1] == "/"


class SEM_MAP():
    """This is a class that generates the semantic map given the observations"""
    def __init__(self, frame_height=480, frame_width=640, hfov=56.0):
        # State holds global and local map and sensor pose
        # See class definition for argument info
        self.semantic_map = VisionLanguage2DSemanticMapState(
            device=DEVICE,
            num_environments=1,
            lseg_features_dim=LSEG_FEATURES_DIM,
            map_resolution=MAP_RESOLUTION,
            map_size_cm=MAP_SIZE_CM,
            global_downscaling=GLOBAL_DOWNSCALING,
        )
        self.semantic_map.init_map_and_pose()
        # Module is responsible for updating the local and global maps and poses
        # See class definition for argument info
        self.semantic_map_module = VisionLanguage2DSemanticMapModule(
            lseg_checkpoint_path=checkpoint_path,
            lseg_features_dim=LSEG_FEATURES_DIM,
            frame_height=frame_height,
            frame_width=frame_width,
            camera_height=CAMERA_HEIGHT, # camera sensor height (in metres)
            hfov=hfov, # horizontal field of view (in degrees)
            map_size_cm=MAP_SIZE_CM, # global map size (in centimetres)
            map_resolution=MAP_RESOLUTION, #  size of map bins (in centimeters): 1 cell = map_resolution cm
            vision_range=VISION_RADIUS_CELL, # radius of the circular region of the local map
                            # that is visible by the agent located in its center (unit is
                            # the number of local map cells). This vision range also affects the
                            # global map. True vision radius = vision_range * map_resolution = 
                            # 63 * 2 = 126 cm
            global_downscaling=GLOBAL_DOWNSCALING, # ratio of global over local map
            du_scale=DU_SCALE, #  frame downscaling before projecting to point cloud
            exp_pred_threshold=1.0, # number of depth points to be in bin to consider it as explored
            map_pred_threshold=NUM_POINTS_OBSTACLE, # number of depth points to be in bin to consider it as obstacle
            # Global map size = MAP_SIZE_CM / map_resolution (unit: cells) = 1000cm / 2 = 500 cells
            # Local map size = MAP_SIZE_CM / global_downscaling / map_resolution (unit: cells) = 1000cm/4/2=125 cells
            # Spot vision radius = 300cm, which is 600cm in diameter. This is euqal to 600cm/2 map_resolution = 300 cells
            # The local map only has the size of 125 cells. So the vision range is 125cells / 2 = 63 cells 
            radius_explore=RADIUS_EXPLORE,
        ).to(DEVICE)
        # Get the update iteration
        self.update_i = 0

    def update_sem_map(self, img_rgb, img_depth, delta_x_y_raw):
        """Update the semantic map"""

        # Process the image data
        img_rgbs = np.stack(img_rgb, axis=0) # img_rgbs = (seq, 480, 640, 3)
        img_depths = np.stack(img_depth, axis=0) * 0.1 # img_depths = (seq, 480, 640, 3) from mm (BD default metric) to cm (semantic map metric)
        seq_obs = np.concatenate((img_rgbs, img_depths),axis=-1)
        # Reshape the image data
        seq_obs = np.transpose(seq_obs, (0, 3, 1, 2))
        
        # Process the pose data
        seq_pose_delta = np.stack(delta_x_y_raw, axis=0) # delta_x_y_raws =  (seq, 3)

        # Format the data
        seq_obs = torch.from_numpy(
            seq_obs[:, :4, :, :]
        ).unsqueeze(0).to(DEVICE)
        seq_pose_delta = torch.from_numpy(
            seq_pose_delta[:]
        ).unsqueeze(0).to(DEVICE)
        seq_dones = torch.tensor(
            [False] * seq_obs.shape[1]
        ).unsqueeze(0).to(DEVICE)
        seq_update_global = torch.tensor(
            [True] * seq_obs.shape[1]
        ).unsqueeze(0).to(DEVICE)

        # Compute the map
        (
            seq_map_features,
            self.semantic_map.local_map,
            self.semantic_map.global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        ) = self.semantic_map_module(
            seq_obs,
            seq_pose_delta,
            seq_dones,
            seq_update_global,
            self.semantic_map.local_map,
            self.semantic_map.global_map,
            self.semantic_map.local_pose,
            self.semantic_map.global_pose,
            self.semantic_map.lmb,
            self.semantic_map.origins,
        )

        # Update the map local and global poses and the origins
        # We use the last seq_local_pose and seq_global_pose as the intilial poses for the next round,
        # and same for origins
        self.semantic_map.local_pose = seq_local_pose[:, -1]
        self.semantic_map.global_pose = seq_global_pose[:, -1]
        self.semantic_map.lmb = seq_lmb[:, -1]
        self.semantic_map.origins = seq_origins[:, -1]

        # Update the counter
        self.update_i += 1

    def export_legend(self, legend, filename="legend.png", save_legend=False):
        """Save the legend"""
        fig = legend.figure
        fig.canvas.draw()
        bbox = legend.get_window_extent().transformed(
            fig.dpi_scale_trans.inverted()
        )
        if save_legend:
            fig.savefig(filename, dpi="figure", bbox_inches=bbox)

    def get_legend(self, text_label_color_palette):
        """Get the legend given color map of the text labels"""
        colors = []
        texts = []
        text_i = 0
        for cc in range(0, len(text_label_color_palette), 3):
            r = text_label_color_palette[cc]
            g = text_label_color_palette[cc + 1]
            b = text_label_color_palette[cc + 2]
            temp = (r, g, b)
            colors.append(temp)
            texts.append(TEXT_LABELS[text_i])
            text_i += 1
        f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
        handles = [f("s", colors[i]) for i in range(text_i)]
        labels = texts
        legend = plt.legend(
            handles, labels, loc=3, framealpha=1, frameon=False
        )
        self.export_legend(legend, SAVE_IMG_DIR + "Legend_" + str(self.update_i) + ".png")

    def plot_sem_map(self):
        """Plot the semantic map function"""

        map_color_palette = [
            1.0,
            1.0,
            1.0,  # empty space
            0.6,
            0.6,
            0.6,  # obstacles
            0.95,
            0.95,
            0.95,  # explored area
            0.96,
            0.36,
            0.26,  # visited area
            *text_label_color_palette,
        ]
        map_color_palette = [int(x * 255.0) for x in map_color_palette]
        num_sem_categories = len(TEXT_LABELS)

        semantic_categories_map = self.semantic_map.get_semantic_map(
            0,
            self.semantic_map_module.lseg,
            labels=TEXT_LABELS
        )

        # Locate the position of the text and class
        label_x = {}
        label_y = {}
        for i in range(semantic_categories_map.shape[0]):
            for j in range(semantic_categories_map.shape[1]):
                if semantic_categories_map[i][j] != len(TEXT_LABELS) - 1:
                    text = TEXT_LABELS[semantic_categories_map[i][j]]
                    if text not in label_x:
                        label_x[text] = []
                        label_y[text] = []
                    label_x[text].append(i)
                    label_y[text].append(j)

        # Plot the map
        plt.figure(figsize=(20,14))
        for text in label_x:
            plt.scatter(label_x[text], label_y[text], label=text)
        plt.legend()
        plt.savefig(SAVE_IMG_DIR + "Sem_Local_Map_Raw_" + str(self.update_i) + ".png")
        plt.close()

        obstacle_map = self.semantic_map.get_obstacle_map(0)
        explored_map = self.semantic_map.get_explored_map(0)
        visited_map = self.semantic_map.get_visited_map(0)

        # Process the semantic map
        semantic_categories_map += 4
        no_category_mask = semantic_categories_map == 4 + num_sem_categories - 1
        obstacle_mask = np.rint(obstacle_map) == 1
        explored_mask = np.rint(explored_map) == 1
        visited_mask = visited_map == 1
        semantic_categories_map[no_category_mask] = 0
        semantic_categories_map[np.logical_and(no_category_mask, explored_mask)] = 2
        semantic_categories_map[np.logical_and(no_category_mask, obstacle_mask)] = 1
        semantic_categories_map[visited_mask] = 3

        # Plot the vis
        semantic_map_vis = Image.new("P", semantic_categories_map.shape)
        semantic_map_vis.putpalette(map_color_palette)
        semantic_map_vis.putdata(semantic_categories_map.flatten().astype(np.uint8))
        semantic_map_vis = semantic_map_vis.convert("RGB")
        # Change it to array.
        semantic_map_vis_flip = np.flipud(semantic_map_vis)
        self.get_legend(text_label_color_palette)
        plt.imshow(semantic_map_vis_flip)
        plt.savefig(SAVE_IMG_DIR + "Sem_Local_Map_" + str(self.update_i) + ".png")
        plt.close()

        print("Finished local map...")

        # Get the same thing for the global map
        semantic_categories_global_map = self.semantic_map.get_semantic_global_map(
            0,
            self.semantic_map_module.lseg,
            labels=TEXT_LABELS,
        )

        obstacle_map = self.semantic_map.get_obstacle_global_map(0)
        explored_map = self.semantic_map.get_explored_global_map(0)
        visited_map = self.semantic_map.get_visited_global_map(0)

        semantic_categories_global_map += 4
        no_category_mask = (
            semantic_categories_global_map == 4 + num_sem_categories - 1
        )
        obstacle_mask = np.rint(obstacle_map) == 1
        explored_mask = np.rint(explored_map) == 1
        visited_mask = visited_map == 1
        semantic_categories_global_map[no_category_mask] = 0
        semantic_categories_global_map[
            np.logical_and(no_category_mask, explored_mask)
        ] = 2
        semantic_categories_global_map[
            np.logical_and(no_category_mask, obstacle_mask)
        ] = 1
        semantic_categories_global_map[visited_mask] = 3

        semantic_map_vis = Image.new("P", semantic_categories_global_map.shape)
        semantic_map_vis.putpalette(map_color_palette)
        semantic_map_vis.putdata(
            semantic_categories_global_map.flatten().astype(np.uint8)
        )
        semantic_map_vis = semantic_map_vis.convert("RGB")
        # Change it to array.
        semantic_map_vis_flip = np.flipud(semantic_map_vis)
        self.get_legend(text_label_color_palette)
        plt.imshow(semantic_map_vis_flip)
        plt.savefig(SAVE_IMG_DIR + "Sem_Global_Map_" + str(self.update_i) + ".png")
        plt.close()

        semantic_map_vis_flip = np.flipud(semantic_map_vis)
        semantic_map_vis_unflip = np.flipud(semantic_map_vis_flip)
        self.get_legend(text_label_color_palette)
        plt.imshow(semantic_map_vis_unflip)
        plt.savefig(
            SAVE_IMG_DIR + "Sem_Global_Unflip_Map_" + str(self.update_i) + ".png"
        )
        plt.close()

        print("Finished global map...")


def height_fy_to_hfov(source_rows_height, fy):
    """Use camera parameter to get height field of view (hFOV)"""
    # We compute this using 
    # https://stackoverflow.com/questions/39992968/how-to-calculate-field-of-view-of-the-camera-from-camera-intrinsic-matrix
    return np.degrees(2 * np.arctan(source_rows_height/(2.0 * fy)))


def main():
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

    # Save map meta parameters
    with open(SAVE_IMG_DIR + "final_global_map_meta_data.pkl", 'wb') as handle:
        pickle.dump([MAP_SIZE_CM, UPDATE_SEM_MAP_EVERY_LEN, \
            CAMERA_HEIGHT, GLOBAL_DOWNSCALING, \
            VISION_RADIUS_CELL, MAP_RESOLUTION, \
            DU_SCALE, NUM_POINTS_OBSTACLE, \
            LSEG_FEATURES_DIM], handle)

    episode_i = 0
    # Set the flag for number of retry
    num_reload = 0
    while True:
        # Print the episode
        print("episode:", episode_i)

        # Init the semantic map
        if episode_i == 0:
            # Compute the hfov using camera parameters
            hfov = 46.994922418496195
            source_rows_height = 480
            source_cols_width = 640
            # Set up the semantic map
            sem_map = SEM_MAP(frame_height=source_rows_height, frame_width=source_cols_width, hfov=hfov)
            # Update the episode
            episode_i += UPDATE_SEM_MAP_EVERY_LEN
        # update the map
        else:
            # Load the dataset and build the map
            try:
                with open(SAVE_IMG_DIR + "data_" + str(episode_i) + ".pkl", 'rb') as handle:
                    img_rgb, img_depth, delta_x_y_raw = pickle.load(handle)
                # Update the map
                sem_map.update_sem_map(img_rgb, img_depth, delta_x_y_raw)
                # Plot the global and local map
                if episode_i % 10 == 0:
                    sem_map.plot_sem_map()
                # Save the global map tensor for future use
                # torch.save(sem_map.semantic_map.global_map, SAVE_IMG_DIR+"final_global_map_"+ str(episode_i) + ".pt")
                # Increase the episode_i by UPDATE_SEM_MAP_EVERY_LEN
                episode_i += UPDATE_SEM_MAP_EVERY_LEN
                # Reset the reload counter
                num_reload = 0
            except:
                # Save the global map tensor for future use
                torch.save(sem_map.semantic_map.global_map, SAVE_IMG_DIR+"final_global_map_"+ str(episode_i) + ".pt")
                # Plot the final map
                sem_map.plot_sem_map()
                # We wait for the data to generate (2 seconds)
                time.sleep(2)
                num_reload += 1
                if num_reload >= 10:
                    print("No data anymore...")
                    break
            
    print("Done...")
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    # We don't need a lease because we're passively observing images (no motor ctrl)
    main()