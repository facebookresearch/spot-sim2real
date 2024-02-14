import os
from typing import List

import bosdyn
import cv2
import numpy as np
import rospy
from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient, build_image_request

SPOT_ADMIN_PW = os.environ["SPOT_ADMIN_PW"]
SPOT_IP = os.environ["SPOT_IP"]


class SpotCamIds:
    r"""Enumeration of types of cameras."""
    # previous cam ids as it is
    HAND_COLOR = "hand_color_image"
    INTEL_REALSENSE_COLOR = "intelrealsensergb"
    INTEL_REALSENSE_DEPTH = "intelrealsensedepth"
    HAND_DEPTH_IN_HAND_COLOR_FRAME = "hand_depth_in_hand_color_frame"


SHOULD_ROTATE: List[
    str
] = []  # this is not relevant for this code none of our Spotcamids are in this array


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

    if reorient and image_response.source.name in SHOULD_ROTATE:
        img = np.rot90(img, k=3)

    return img


class Spot:
    def __init__(self, client_name_prefix: str = "SpotImageSDK"):

        bosdyn.client.util.setup_logging()
        sdk = bosdyn.client.create_standard_sdk(client_name_prefix)
        robot = sdk.create_robot(SPOT_IP)
        robot.authenticate("admin", SPOT_ADMIN_PW)
        robot.time_sync.wait_for_sync()
        self.robot = robot
        self.image_client = robot.ensure_client(ImageClient.default_service_name)

        # Make our intel image client
        self.intelrealsense_image_client = robot.ensure_client(
            "intel-realsense-image-service"
        )

    @property
    def IS_GRIPPER_BLOCKED(self):
        return rospy.get_param("is_gripper_blocked", default=0) == 1

    def get_image_responses(
        self, sources, quality=100, pixel_format=None, await_the_resp=True
    ):
        """Retrieve images from Spot's cameras

        :param sources: list containing camera uuids
        :param quality: either an int or a list specifying what quality each source
            should return its image with
        :param pixel_format: either an int or a list specifying what pixel format each source
            should return its image with
        :return: list containing bosdyn image response objects
        """
        image_client = (
            self.image_client
            if "intel" not in sources[0]
            else self.intelrealsense_image_client
        )
        if quality is not None:
            if isinstance(quality, int):
                quality = [quality] * len(sources)
            else:
                assert len(quality) == len(sources)
            img_requests = [
                build_image_request(src, q) for src, q in zip(sources, quality)
            ]
            image_responses = image_client.get_image_async(img_requests)
        elif pixel_format is not None:
            if isinstance(pixel_format, int):
                pixel_format = [pixel_format] * len(sources)
            else:
                assert len(pixel_format) == len(sources)
            img_requests = [
                build_image_request(src, pixel_format=pf)
                for src, pf in zip(sources, pixel_format)
            ]
            image_responses = image_client.get_image_async(img_requests)
        else:
            image_responses = image_client.get_image_from_sources_async(sources)

        return image_responses.result() if await_the_resp else image_responses

    def get_hand_image_old(self, is_rgb=True, img_src: List[str] = []):
        """
        Gets hand raw rgb & depth, returns List[rgbimage, unscaleddepthimage] image object is BD source image object which has kinematic snapshot & camera intrinsics along with pixel data
        """
        img_src = (
            img_src
            if img_src
            else [SpotCamIds.HAND_COLOR, SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME]
        )  # default img_src to gripper

        pixel_format_rgb = (
            image_pb2.Image.PIXEL_FORMAT_RGB_U8
            if is_rgb
            else image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8
        )
        pixel_format_depth = image_pb2.Image.PIXEL_FORMAT_DEPTH_U16
        img_resp = self.get_image_responses(
            img_src, pixel_format=[pixel_format_rgb, pixel_format_depth]
        )
        return img_resp

    def get_hand_image(self, is_rgb=True):
        """
        Gets hand raw rgb & depth, returns List[rgbimage, unscaleddepthimage] image object is BD source image object which has kinematic snapshot & camera intrinsics along with pixel data
        If is_gripper_blocked is True then returns intel realsense images
        If hand_image_sources are passed then above condition is ignored & will send image & depth for each source
        Thus if you send hand_image_sources=["gripper", "intelrealsense"] then 4 image resps should be returned
        """
        realsense_img_srcs: List[str] = [
            SpotCamIds.INTEL_REALSENSE_COLOR,
            SpotCamIds.INTEL_REALSENSE_DEPTH,
        ]
        if self.IS_GRIPPER_BLOCKED:  # return intel realsense
            return self.get_hand_image_old(img_src=realsense_img_srcs)
        else:
            return self.get_hand_image_old(is_rgb=is_rgb)


if __name__ == "__main__":

    def plot(images):
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
        # cv2.waitKey(1)

    spot = Spot()

    # should display images from intel realsense
    rospy.set_param("is_gripper_blocked", 1)
    image_resps = spot.get_hand_image()
    images = [image_response_to_cv2(image_resp) for image_resp in image_resps]
    plot(images)  # shows single image

    # should display images from gripper cam
    rospy.set_param("is_gripper_blocked", 0)
    image_resps = spot.get_hand_image()
    images = [image_response_to_cv2(image_resp) for image_resp in image_resps]
    plot(images)  # shows single image

    # passing srcs
    # image_resps = spot.get_hand_image()
    # images = [
    #     image_response_to_cv2(image_resp) for image_resp in image_resps
    # ]  # should be 4 image resps
    # plot(images[:2])  # plot gripper images
    # plot(images[2:])  # plot intel realsense images

    # in spot publisher add
    # image_responses = self.spot.get_image_responses(self.sources[:2], quality=100, await_the_resp=False)
    # hand_image_responses = self.spot.get_hand_image()
    # image_responses = image_responses.result()
    # image_responses.extend(hand_image_responses)
