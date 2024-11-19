# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Simple image capture tutorial."""

import argparse, os
import sys
import time
import cv2
import numpy as np

import bosdyn.client
import bosdyn.client.util
from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient, build_image_request

SPOT_IP = os.environ["SPOT_IP"]
SPOT_ADMIN_PW = os.environ["SPOT_ADMIN_PW"]

class SpotCamIds:
    r"""Enumeration of types of cameras."""

    BACK_DEPTH = "back_depth"
    BACK_DEPTH_IN_VISUAL_FRAME = "back_depth_in_visual_frame"
    BACK_FISHEYE = "back_fisheye_image"
    FRONTLEFT_DEPTH = "frontleft_depth"
    FRONTLEFT_DEPTH_IN_VISUAL_FRAME = "frontleft_depth_in_visual_frame"
    FRONTLEFT_FISHEYE = "frontleft_fisheye_image"
    FRONTRIGHT_DEPTH = "frontright_depth"
    FRONTRIGHT_DEPTH_IN_VISUAL_FRAME = "frontright_depth_in_visual_frame"
    FRONTRIGHT_FISHEYE = "frontright_fisheye_image"
    HAND_COLOR = "hand_color_image"
    HAND_COLOR_IN_HAND_DEPTH_FRAME = "hand_color_in_hand_depth_frame"
    HAND_DEPTH = "hand_depth"
    HAND_DEPTH_IN_HAND_COLOR_FRAME = "hand_depth_in_hand_color_frame"
    HAND = "hand_image"
    LEFT_DEPTH = "left_depth"
    LEFT_DEPTH_IN_VISUAL_FRAME = "left_depth_in_visual_frame"
    LEFT_FISHEYE = "left_fisheye_image"
    RIGHT_DEPTH = "right_depth"
    RIGHT_DEPTH_IN_VISUAL_FRAME = "right_depth_in_visual_frame"
    RIGHT_FISHEYE = "right_fisheye_image"
    INTEL_REALSENSE_COLOR = "intelrealsensergb"  # In habitat-lab, the intelrealsense camera is called jaw camera
    INTEL_REALSENSE_DEPTH = "intelrealsensedepth"

ROTATION_ANGLE = {
    'back_fisheye_image': 0,
    'frontleft_fisheye_image': -78,
    'frontright_fisheye_image': -102,
    'left_fisheye_image': 0,
    'right_fisheye_image': 180
}

def image_response_to_cv2(image_response, reorient=True):
    if (
        image_response.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16
        and image_response.shot.image.format == image_pb2.Image.FORMAT_RAW
    ):
        dtype = np.uint16
    else:
        dtype = np.uint8
    # img = np.fromstring(image_response.shot.image.data, dtype=dtype)
    img = np.frombuffer(image_response.shot.image.data, dtype=dtype)
    if image_response.shot.image.format == image_pb2.Image.FORMAT_RAW:
        img = img.reshape(
            image_response.shot.image.rows, image_response.shot.image.cols
        )
    else:
        img = cv2.imdecode(img, -1)

    return img


def main():
    # Parse args
    sdk = bosdyn.client.create_standard_sdk('image_capture')
    robot = sdk.create_robot(SPOT_IP)
    robot.authenticate("admin", SPOT_ADMIN_PW)
    #bosdyn.client.util.authenticate(robot)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()

    image_client = robot.ensure_client(ImageClient.default_service_name)
    intel_realsense_client = robot.ensure_client("intel-realsense-image-service")
    #
    image_sources = [SpotCamIds.HAND_COLOR, SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME]
    intel_sources = [SpotCamIds.INTEL_REALSENSE_COLOR, SpotCamIds.INTEL_REALSENSE_DEPTH]
    # Raise exception if no actionable argument provided
    if not image_sources :
        raise Exception('Must provide actionable argument (list or image-sources).')
    # Optionally capture one or more images.
    n = 0
    start_time = time.time()
    prev_frame_time = 0
  
    # used to record the time at which we processed current frame 
    new_frame_time = 0
    while image_sources:
        # Capture and save images to disk
        #, 

        pixel_formats = [image_pb2.Image.PIXEL_FORMAT_RGB_U8, image_pb2.Image.PIXEL_FORMAT_DEPTH_U16] #pixel_format_string_to_enum(options.pixel_format)
        image_request = [
            build_image_request(source, pixel_format=pixel_formats[i])
            for i, source in enumerate(intel_sources)
        ]
        image_responses = intel_realsense_client.get_image(image_request)
        images = [image_response_to_cv2(img) for img in image_responses]
        end_time = time.time()
        n += 1
        new_frame_time = time.time() 
        fps = 1/(new_frame_time-prev_frame_time) 
        prev_frame_time = new_frame_time 
        #socket.send_pyobj("")
        print(f"FPS {fps}")

    return True


if __name__ == '__main__':
    if not main():
        sys.exit(1)