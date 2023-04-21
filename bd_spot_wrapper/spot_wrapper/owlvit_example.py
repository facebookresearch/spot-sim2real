import argparse
import time
from collections import deque
import pickle
import cv2
import numpy as np
import sys
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
processor = AutoProcessor.from_pretrained("google/owlvit-large-patch14")
model = AutoModelForZeroShotObjectDetection.from_pretrained("google/owlvit-large-patch14")

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
    path = r"/Users/jimmytyyang/Desktop/Screenshot 2023-02-23 at 10.41.08 AM.png"
    image = Image.open(path)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    texts = [["Door Handle"]]
    inputs = processor(text=texts, images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    # Print detected objects and rescaled box coordinates
    score_threshold = 0.04
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        if score >= score_threshold:
            print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()
