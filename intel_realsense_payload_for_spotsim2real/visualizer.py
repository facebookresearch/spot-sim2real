# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Visualize RGB & Depth from the image source specified (default = spot gripper )"""

import os
import sys
import time

import bosdyn.client as bosdynclient
import bosdyn.client.util as util
import cv2
import numpy as np
from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient, build_image_request

env_err_msg = (
    "\n{var_name} not found as an environment variable!\n"
    "Please run:\n"
    "echo 'export {var_name}=<YOUR_{var_name}>' >> ~/.bashrc\nor for MacOS,\n"
    "echo 'export {var_name}=<YOUR_{var_name}>' >> ~/.bash_profile\n"
    "Then:\nsource ~/.bashrc\nor\nsource ~/.bash_profile"
)

try:
    SPOT_ADMIN_PW = os.environ["SPOT_ADMIN_PW"]
except KeyError:
    raise RuntimeError(env_err_msg.format(var_name="SPOT_ADMIN_PW"))
try:
    SPOT_IP = os.environ["SPOT_IP"]
except KeyError:
    raise RuntimeError(env_err_msg.format(var_name="SPOT_IP"))


def plot(images):
    if len(images) == 1:
        cv2.imshow("Client Image Stream", images[0])
        cv2.waitKey(1)
    else:
        color_image, depth_image = images
        height, width = color_image.shape[:2]
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )
        depth_image_bin = (
            np.where(depth_image > 0.0, 1, 0).reshape(height, width, 1).astype(np.uint8)
        )
        cv2.imshow(
            "Client Image Stream",
            np.hstack((color_image, depth_colormap, depth_image_bin * color_image)),
        )
        cv2.waitKey(1)


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

    if reorient:  # and image_response.source.name in SHOULD_ROTATE:
        img = np.rot90(img, k=3)

    return img


def get_image_responses(sources, quality=None, pixel_format=None, image_client=None):
    """Retrieve images from Spot's cameras

    :param sources: list containing camera uuids
    :param quality: either an int or a list specifying what quality each source
        should return its image with
    :param pixel_format: either an int or a list specifying what pixel format each source
        should return its image with
    :return: list containing bosdyn image response objects
    """
    if quality is not None:
        if isinstance(quality, int):
            quality = [quality] * len(sources)
        else:
            assert len(quality) == len(sources)
        img_requests = [build_image_request(src, q) for src, q in zip(sources, quality)]
        image_responses = image_client.get_image(img_requests)
    elif pixel_format is not None:
        if isinstance(pixel_format, int):
            pixel_format = [pixel_format] * len(sources)
        else:
            assert len(pixel_format) == len(sources)
        img_requests = [
            build_image_request(src, pixel_format=pf)
            for src, pf in zip(sources, pixel_format)
        ]
        image_responses = image_client.get_image(img_requests)
    else:
        image_responses = image_client.get_image_from_sources(sources)

    return image_responses


def get_realsense_images(
    is_rgb=True, image_client=None, img_src=["intelrealsensergb", "intelrealsensedepth"]
):
    """
    Gets hand raw rgb & depth, returns List[rgbimage, unscaleddepthimage] image object is BD source image object which has kinematic snapshot & camera intrinsics along with pixel data
    """

    pixel_format_rgb = (
        image_pb2.Image.PIXEL_FORMAT_RGB_U8
        if is_rgb
        else image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8
    )
    pixel_format_depth = image_pb2.Image.PIXEL_FORMAT_DEPTH_U16
    pixel_formats = []
    for img_src_ in img_src:
        pixel_formats += (
            [pixel_format_depth] if "depth" in img_src_ else [pixel_format_rgb]
        )
    img_resp = get_image_responses(
        img_src,
        pixel_format=pixel_formats,
        image_client=image_client,
    )
    return img_resp


def get_images_from_socket(socket, n: int = 2):
    socke_images = [0] * n
    for i in range(n):
        buf = socket.recv()
        if i == 0:
            socke_images[0] = np.frombuffer(buf, dtype=np.uint8).reshape(480, 640, 3)
        else:
            socke_images[1] = np.frombuffer(buf, dtype=np.uint16).reshape(480, 640)
    return socke_images


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--verify-with-socket",
        action="store_true",
        required=False,
        help="If passed, socket will be connected to payload service server & verifies images recieved through gRPC, please also enable this in payload service only for testing & debugging",
    )
    parser.add_argument(
        "--intel",
        action="store_true",
        required=False,
        help="If passed, Intel Realsense Src will be used",
    )
    options = parser.parse_args()

    util.setup_logging()
    sdk = bosdynclient.create_standard_sdk("IntelRealSenseClient")
    robot = sdk.create_robot(SPOT_IP)
    robot.authenticate("admin", SPOT_ADMIN_PW)
    util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    robot.sync_with_directory()
    image_client = robot.ensure_client("intel-realsense-image-service")
    default_image_client = robot.ensure_client(ImageClient.default_service_name)

    if options.verify_with_socket:
        import zmq

        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        port = "21001"
        print("Connecting to socket")
        socket.connect("tcp://localhost:%s" % port)

    times, n = 0.0, 0
    while True:
        start_time = time.time()
        imageresp_from_source = get_realsense_images(
            image_client=image_client if options.intel else default_image_client,
            img_src=["intelrealsensergb", "intelrealsensedepth"]
            if options.intel
            else ["hand_color_image", "hand_depth_in_hand_color_frame"],
        )
        intrinsics = imageresp_from_source[0].source.pinhole.intrinsics
        # breakpoint()
        images = [
            image_response_to_cv2(imgresp, False) for imgresp in imageresp_from_source
        ]
        if options.verify_with_socket:
            images_from_socket = get_images_from_socket(socket, len(images))
            # print(images[0].shape, images[0].dtype, images[0].min(), images[0].max())
            # print(images_from_socket[0].shape, images_from_socket[0].dtype, images_from_socket[0].min(), images_from_socket[0].max())
            # print(images[-1].shape, images[-1].dtype, images[-1].min(), images[-1].max())
            # print(images_from_socket[-1].shape, images_from_socket[-1].dtype, images_from_socket[-1].min(), images_from_socket[-1].max())
            # assert np.allclose(images[0], images_from_socket[0]), "RGB image doesn't match"
            assert np.allclose(
                images[-1], images_from_socket[-1]
            ), "Depth image doesn't match"

        plot(images)
        dtime = time.time() - start_time
        times += dtime
        print(f"Time taken to get 1 frame {dtime}, fps {n/times}")
        n += 1
        # time.sleep(0.3)
