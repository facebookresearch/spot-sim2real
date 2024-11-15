# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Simple image capture tutorial."""

import os
import time

import bosdyn.client
import bosdyn.client.util
import cv2
import numpy as np
import zmq
from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient, build_image_request

SPOT_IP = os.environ["SPOT_IP"]
SPOT_ADMIN_PW = os.environ["SPOT_ADMIN_PW"]


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

    # Create robot object with an image client.
    sdk = bosdyn.client.create_standard_sdk("image_capture")
    robot = sdk.create_robot(SPOT_IP)
    robot.authenticate("admin", SPOT_ADMIN_PW)

    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()

    image_client = robot.ensure_client(ImageClient.default_service_name)

    image_sources = ["hand_color_image", "hand_depth_in_hand_color_frame"]

    # Raise exception if no actionable argument provided
    if not image_sources:
        raise Exception("Must provide actionable argument (list or image-sources).")

    _ = time.time()
    encode_type = ".png"
    quality = 5
    encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), int(quality)]
    port = os.environ.get("FAST_DEPTH_PORT", 21998)

    context = zmq.Context()

    socket = context.socket(zmq.PUB)

    # Set the socket to immediately close and release the port upon termination
    socket.setsockopt(zmq.LINGER, 0)

    # Bind the socket to all interfaces and specified port
    socket.bind(f"tcp://*:{port}")
    print(f"ZeroMQ server listening on port {port}")

    while True:
        pixel_formats = [
            image_pb2.Image.PIXEL_FORMAT_RGB_U8,
            image_pb2.Image.PIXEL_FORMAT_DEPTH_U16,
        ]
        image_request = [
            build_image_request(source, pixel_format=pixel_formats[i])
            for i, source in enumerate(image_sources)
        ]
        image_responses = image_client.get_image(image_request)
        images = [image_response_to_cv2(img) for img in image_responses]

        # end_time = time.time()
        # n += 1
        # #print(f"FPS {n/(end_time-start_time)}")

        compressed_depth = cv2.imencode(encode_type, images[-1], encode_param)[
            1
        ].tobytes()
        image_responses[-1].shot.image.data = compressed_depth
        image_responses[-1].shot.image.format = image_pb2.Image.FORMAT_JPEG
        image_responses = [image.SerializeToString() for image in image_responses]
        socket.send_pyobj(image_responses)
        # socket.recv_pyobj()


if __name__ == "__main__":
    main()
