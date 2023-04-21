import argparse
import time
from collections import deque
import pickle

import cv2
import numpy as np
from spot_wrapper.spot import (
    Spot,
    SpotCamIds,
    draw_crosshair,
    image_response_to_cv2,
    scale_depth_img,
)
from spot_wrapper.utils.utils import color_bbox, resize_to_tallest_old_version

import sys
sys.path.append("/Users/jimmytyyang/HomeRobot/home-robot/src")
from home_robot.agent.perception.detection.lseg import load_lseg_for_inference
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
checkpoint_path = "/Users/jimmytyyang/LSeg/checkpoints/demo_e200.ckpt"
device = torch.device("cpu")
#device = torch.device("mps")
model = load_lseg_for_inference(checkpoint_path, device)
model = model.eval()
# freeze all model
for param in model.parameters():
    param.requires_grad = False


MAX_HAND_DEPTH = 3.0
MAX_HEAD_DEPTH = 10.0
DETECT_LARGEST_WHITE_OBJECT = False
# This is useful when transforming the front left/right depth into rgb for PIXEL_FORMAT
PIXEL_FORMAT_RGB_U8 = "PIXEL_FORMAT_RGB_U8"

# Define the label
TEXT_LABELS = ["other", \
    "spoon", \
    "cup", \
    "bowl", \
    "plate", \
    "fork", \
    "bottle", \
    "mug", \
    "snack", \
    "photo_chip",\
    "head_phone", \
    "box", \
    "keyboard", \
    "toy", \
    "cubic", \
    "mouse"]
# Define Metadata for plotting the text using cv2
# Font
font = cv2.FONT_HERSHEY_SIMPLEX
# FontScale
fontScale = 1
# Color in RGB
color = (255, 255, 255)
# Line thickness of 2 px
thickness = 2

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
    RGB_LSeg = [SpotCamIds.HAND_COLOR]
    try:
        image_i = 0
        while True:
            start_time = time.time()
            # Get Spot camera image
            image_responses = spot.get_image_responses(sources, quality=args.quality)
            imgs = []
            rgb_lsegs = []
            for image_response, source in zip(image_responses, sources):
                img = image_response_to_cv2(image_response, reorient=True)
                if "depth" in source:
                    max_depth = MAX_HAND_DEPTH if "hand" in source else MAX_HEAD_DEPTH
                    img = scale_depth_img(img, max_depth=max_depth, as_img=True)
                elif source is SpotCamIds.HAND_COLOR:
                    img = draw_crosshair(img)
                    if DETECT_LARGEST_WHITE_OBJECT:
                        x, y, w, h = color_bbox(img, just_get_bbox=True)
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if source in RGB_LSeg:
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
            img, width_list = resize_to_tallest_old_version(imgs, hstack=True)

            if not args.no_display:
                # Add the text into the semantic RGB
                for i in range(len(RGB_LSeg)):
                    # Get the offset for plotting text on the semantic RGB
                    offset = 0
                    for j in range(len(sources)+i):
                        offset += width_list[j]
                    for text_label in get_text_and_img_pos[i]:
                        x = get_text_and_img_pos[i][text_label][0] + int(offset)
                        y = get_text_and_img_pos[i][text_label][1]
                        img = cv2.putText(img, text_label, (x, y),\
                            font,\
                            fontScale,\
                            color,\
                            thickness,\
                            cv2.LINE_AA)
                cv2.imshow(window_name, img)
                cv2.waitKey(1)

            time_buffer.append(time.time() - start_time)
            print("Avg FPS:", 1 / np.mean(time_buffer))
    finally:
        # # Save the file
        # with open('spot_1215.pkl', 'wb') as handle:
        #     pickle.dump(img_list, handle)
        if not args.no_display:
            cv2.destroyWindow(window_name)


if __name__ == "__main__":
    spot = Spot("ViewCamera")
    # We don't need a lease because we're passively observing images (no motor ctrl)
    main(spot)
