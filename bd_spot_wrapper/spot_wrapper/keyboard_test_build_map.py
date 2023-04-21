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
    "computer",
    "plant",
    "chair",
    "couch",
    "table",
    "cabinet",
    "sink",
    "fridge",
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
from home_robot.agent.mapping.dense.semantic.vision_language_2d_semantic_map_state import VisionLanguage2DSemanticMapState
from home_robot.agent.mapping.dense.semantic.vision_language_2d_semantic_map_module import VisionLanguage2DSemanticMapModule
from PIL import Image
from home_robot.agent.perception.detection.coco_maskrcnn.coco_categories import (
    coco_categories, coco_categories_color_palette, text_label_color_palette
)
# Map parameters
MAP_SIZE_CM = 1000
UPDATE_SEM_MAP_EVERY_LEN = 10
CAMERA_HEIGHT = 0.925 # for arm camera in meter. standing height 0.61 + arm height 0.35 = 0.96. Iphone app ruler = 0.61 
GLOBAL_DOWNSCALING = 2
VISION_RADIUS_CELL = 125
MAP_RESOLUTION = 2
DU_SCALE = 4
NUM_POINTS_OBSTACLE = 200
LSEG_FEATURES_DIM = 512
SAVE_IMG_DIR = "/Users/jimmytyyang/Documents/spot_sem_map_walk_v4_0213"

RADIUS_EXPLORE = VISION_RADIUS_CELL


# Create the save folder
try:
    os.mkdir(SAVE_IMG_DIR)
except:
    print("Save Folder Created...")
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

        self.semantic_map.local_pose = seq_local_pose[:, -1]
        self.semantic_map.global_pose = seq_global_pose[:, -1]
        self.semantic_map.lmb = seq_lmb[:, -1]
        self.semantic_map.origins = seq_origins[:, -1]

        self.update_i += 1

    def export_legend(self, legend, filename="legend.png"):
        fig = legend.figure
        fig.canvas.draw()
        bbox = legend.get_window_extent().transformed(
            fig.dpi_scale_trans.inverted()
        )
        #fig.savefig(filename, dpi="figure", bbox_inches=bbox)

    def get_legend(self, text_label_color_palette):
        # Save the legend
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
        num_sem_categories = len(TEXT_LABELS) #len(coco_categories)

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
        plt.savefig(SAVE_IMG_DIR + "Sem_Local_Map_Package_" + str(self.update_i) + ".png")
        plt.close()

        print("Finished the local map...")

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
        plt.savefig(SAVE_IMG_DIR + "Sem_Global_Map_Package_" + str(self.update_i) + ".png")
        plt.close()

        semantic_map_vis_flip = np.flipud(semantic_map_vis)
        semantic_map_vis_unflip = np.flipud(semantic_map_vis_flip)
        self.get_legend(text_label_color_palette)
        plt.imshow(semantic_map_vis_unflip)
        plt.savefig(
            SAVE_IMG_DIR + "Sem_Global_Unflip_Map_Package_" + str(self.update_i) + ".png"
        )
        plt.close()

        print("Finished Global Map...")

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
    RGB_LSeg_LIST = [SpotCamIds.HAND_COLOR]
    # Get the rgb and depth for building the map
    sources_for_map = [
        # Aligh the depth image in the rgb frame
        SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME,
        SpotCamIds.HAND_COLOR,
    ] 

    #try:
    episode_i = 0
    # Initilize the current position as the global origin position
    spot.home_robot()
    # Get the current pose of the robot
    last_x, last_y, last_yaw = spot.get_xy_yaw(use_boot_origin=True)
    while True:
        # Get the time
        start_time = time.time()

        # Print the episode
        print("episode:", episode_i)

        # Initilize the lists and update the semantic map if possible
        if episode_i % UPDATE_SEM_MAP_EVERY_LEN == 0:
            # Init the semantic map
            if episode_i == 0:
                # Get the camera parameters.
                image_responses = spot.get_image_responses(sources_for_map, quality=args.quality)
                for image_response, source in zip(image_responses, sources_for_map):
                    if source is SpotCamIds.HAND_COLOR:
                        # Get the camera matrics
                        source_rows_height = image_response.source.rows
                        source_cols_width = image_response.source.cols
                        fx = image_response.source.pinhole.intrinsics.focal_length.x
                        fy = image_response.source.pinhole.intrinsics.focal_length.y
                        cx = image_response.source.pinhole.intrinsics.principal_point.x
                        cy = image_response.source.pinhole.intrinsics.principal_point.y
                        break
                # Compute the hfov using 
                # https://stackoverflow.com/questions/39992968/how-to-calculate-field-of-view-of-the-camera-from-camera-intrinsic-matrix
                hfov = np.degrees(2 * np.arctan(source_rows_height/(2.0 * fy)))
                sem_map = SEM_MAP(frame_height=source_rows_height, frame_width=source_cols_width, hfov=hfov)
            # update the map
            else:
                sem_map.update_sem_map(img_rgb, img_depth, delta_x_y_raw)
                sem_map.plot_sem_map()

            # Init or reset the lists
            img_rgb = []
            img_depth = []
            delta_x_y_raw = []

        # Get Spot camera image for cv2 plot
        image_responses = spot.get_image_responses(sources, quality=args.quality)
        imgs = []
        rgb_lsegs = []
        for image_response, source in zip(image_responses, sources):
            img = image_response_to_cv2(image_response, reorient=True)
            # Get the information for building the cv2 viewer
            if "depth" in source:
                max_depth = MAX_HAND_DEPTH if "hand" in source else MAX_HEAD_DEPTH
                img = scale_depth_img(img, max_depth=max_depth, as_img=True)
            elif source is SpotCamIds.HAND_COLOR:
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

        # Get Spot's position and rotation
        curr_x, curr_y, curr_yaw = spot.get_xy_yaw(use_boot_origin=True)
        
        # Get the Spot's local delta position
        delta_x, delta_y, delta_yaw = spot.get_xy_yaw(use_boot_origin=False)
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
        print("delta pose x, y, yaw:", delta_x, delta_y, delta_yaw)
        delta_global = ((curr_x-last_x)**2+(curr_y-last_y)**2)**0.5
        print("x, y delta in global frame:", delta_global)
        delta_local = (delta_x**2+delta_y**2)**0.5
        print("x, y delta in local frame:", delta_local)
        # Terminate if the delta is too huge
        if abs(delta_global-delta_local) >= 2.5 * 1e-2: # 2.5cm differecne passed
            print("Error, delta between global and local transformations is too large")
            import pdb; pdb.set_trace()

        # Store the last position for computing the delta in the next round
        last_x, last_y, last_yaw = spot.get_xy_yaw(use_boot_origin=True)

        # Use the current position as the original position of the robot
        spot.home_robot()

    # finally:
    #     import pdb; pdb.set_trace()
    #     if not args.no_display:
    #         cv2.destroyWindow(window_name)


if __name__ == "__main__":
    spot = Spot("ViewCamera")
    # We don't need a lease because we're passively observing images (no motor ctrl)
    main(spot)
