import argparse
import time
from collections import deque
import pickle

import cv2
import numpy as np

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


MAX_HAND_DEPTH = 3.0
MAX_HEAD_DEPTH = 10.0
DETECT_LARGEST_WHITE_OBJECT = False
# This is useful when transforming the front left/right depth into rgb for PIXEL_FORMAT
PIXEL_FORMAT_RGB_U8 = "PIXEL_FORMAT_RGB_U8"

# Define the label
TEXT_LABELS = ["other", "handle"]
# Define Metadata for plotting the text using cv2
# Font
font = cv2.FONT_HERSHEY_SIMPLEX
# FontScale
fontScale = 1
# Color in RGB
color = (255, 255, 255)
# Line thickness of 2 px
thickness = 2

def main():
    rgb_img = cv2.imread("/Users/jimmytyyang/Desktop/Screenshot 2023-02-23 at 10.41.08 AM.png")
    rgb_img = rgb_img[:,500:1500,:]
    cv2.imshow("Raw", rgb_img)
    # rgb_img's shape of (1122, 1872, 3)        
    rgb = torch.unsqueeze(torch.tensor(rgb_img), 0)
    # Pixel_features: (batch_size, 512, H, W)
    pixel_features = model.encode(rgb)
    # Get the prediction
    one_hot_predictions, visualizations = model.decode(pixel_features, TEXT_LABELS)

    cv2.imshow("Lseg", visualizations[0])   
    cv2.waitKey(1)

    import pdb; pdb.set_trace()



if __name__ == "__main__":
    main()
