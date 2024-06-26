# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Reference code IntelRS : https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/align-depth2color.py
# Reference code Boston Dynamics Spot-sdk: https://github.com/boston-dynamics/spot-sdk/tree/master/python/examples/web_cam_image_service

import logging
import time
from typing import Any, List

import cv2
import numpy as np
import pyrealsense2 as rs
from bosdyn.api import image_pb2
from bosdyn.client.image_service_helpers import (
    CameraInterface,
    convert_RGB_to_grayscale,
)

_LOGGER = logging.getLogger(__name__)


def create_realsense_pipeline(
    res_width: int = 640, res_height: int = 480, filter_depth: bool = False
):
    # Create a pipeline
    pipeline = rs.pipeline()
    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    print(f"Device {str(device.get_info(rs.camera_info.product_line))}")
    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == "RGB Camera":
            found_rgb = True
            break
    if not found_rgb:
        print("The service requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, res_width, res_height, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, res_width, res_height, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)
    intrinsics = (
        profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    )
    align_to = rs.stream.color
    align = rs.align(align_to)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    filters: List[Any] = []
    # warmup
    warmup_steps: int = 10
    [pipeline.wait_for_frames() for _ in range(warmup_steps)]
    if filter_depth:
        spatial_filter = rs.spatial_filter()
        spatial_filter.set_option(rs.option.holes_fill, 2)
        filters.append(spatial_filter)
        frames = []
        for x in range(warmup_steps):
            frameset = align.process(pipeline.wait_for_frames())
            frames.append(frameset.get_depth_frame())
        temporal = rs.temporal_filter()
        for x in range(warmup_steps):
            _ = temporal.process(frames[x])
        filters.append(temporal)

    return pipeline, profile, align, intrinsics, depth_scale, filters


class IntelRealSenseCameraInterface(CameraInterface):
    """Provide access to the RGB/Depth frames using Intel Realsense Wrapper
    Args: source_name : realsensergb or realsensedepth
    fps = 30
    show_debug_information = True/False, shows OpenCV visualization window
    res_width = 640
    res_height = 480
    """

    DEPTH_FRAME = None
    COLOR_FRAME = None
    CAPTURE_TIME = None

    def __init__(
        self,
        source_name: str = "intelrealsensergb",
        pipeline: rs.pipeline = None,
        profile=None,
        align=None,
        jpeg_quality: int = 75,
        png_quality: int = 9,
        show_debug_information: bool = False,
        res_width: int = 640,
        res_height: int = 480,
        socket=None,
        filters=None,
    ):

        self.show_debug_images = show_debug_information

        # Create the image source name from the device name.
        self.image_source_name = source_name
        self.mode: int = 0 if "rgb" in self.image_source_name else 1

        assert (
            pipeline is not None and profile is not None and align is not None
        ), "pipeline or profile can't be none"
        self.pipeline = pipeline
        self.profile = profile
        self.align = align
        self.default_jpeg_quality = jpeg_quality
        self.default_png_quality = png_quality

        self.socket = socket
        self.depthfilters = (
            filters  # should be applied before calling .get_data() method
        )

        if self.mode:
            # Getting the depth sensor's depth scale (see rs-align example for explanation)
            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            print("Depth Scale is: ", self.depth_scale)
            # We will be removing the background of objects more than
            #  clipping_distance_in_meters meters away
            clipping_distance_in_meters = 1  # 1 meter
            self.clipping_distance = clipping_distance_in_meters / self.depth_scale

        # Attempt to determine the gain and exposure for the camera.
        self.camera_exposure, self.camera_gain = None, None

        # Determine the dimensions of the image.
        self.rows = int(res_height)
        self.cols = int(res_width)

        # Determine the pixel format.
        if self.mode:
            # Depth mode
            self.supported_pixel_formats = [image_pb2.Image.PIXEL_FORMAT_DEPTH_U16]
        else:
            self.supported_pixel_formats = [
                image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8,
                image_pb2.Image.PIXEL_FORMAT_RGB_U8,
            ]

    def blocking_capture(self):
        # Get the image from the video capture.
        try:
            if not self.mode:

                # Align the depth frame to color frame
                aligned_frames = self.align.process(self.pipeline.wait_for_frames())
                capture_time = time.time()  # frames.get_timestamp()

                # Get aligned frames
                aligned_depth_frame = (
                    aligned_frames.get_depth_frame()
                )  # aligned_depth_frame is a 640x480 depth image
                # allow depth filtering
                if self.depthfilters:
                    for filter in self.depthfilters:
                        aligned_depth_frame = filter.process(aligned_depth_frame)

                color_frame = aligned_frames.get_color_frame()

                # Validate that both frames are valid
                if not aligned_depth_frame or not color_frame:
                    raise Exception("Either RGB frame or depth frame is None")

                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                IntelRealSenseCameraInterface.COLOR_FRAME = color_image
                IntelRealSenseCameraInterface.DEPTH_FRAME = depth_image
                IntelRealSenseCameraInterface.CAPTURE_TIME = capture_time

                # print(capture_time, self.mode)
                if self.show_debug_images and not self.mode:
                    try:
                        depth_colormap = cv2.applyColorMap(
                            cv2.convertScaleAbs(depth_image, alpha=0.03),
                            cv2.COLORMAP_JET,
                        )
                        depth_image_bin = (
                            np.where(depth_image > 0.0, 1, 0)
                            .reshape(self.rows, self.cols, 1)
                            .astype(np.uint8)
                        )
                        cv2.imshow(
                            "Depth Image Capture",
                            np.hstack(
                                (
                                    color_image,
                                    depth_colormap,
                                    depth_image_bin * color_image,
                                )
                            ),
                        )
                        cv2.waitKey(1)
                    except Exception as e:
                        _LOGGER.warning(
                            f"Unable to display the IntelRealSense images captured.{str(e)}"
                        )
                        pass
            else:
                return (
                    IntelRealSenseCameraInterface.DEPTH_FRAME,
                    IntelRealSenseCameraInterface.CAPTURE_TIME,
                )

            return color_image, capture_time
        except Exception as e:
            print(
                f"Unsuccessful in getting frames from self.pipeline.wait_for_frames() due to {str(e)}"
            )
            exit(0)

    def image_decode(self, image_data, image_proto, image_req):
        pixel_format = image_req.pixel_format
        image_format = image_req.image_format
        print(
            f"In Decode Method, Stream Mode: {self.mode}, Pixel_format : {pixel_format}, Image_format: {image_format}"
        )
        converted_image_data = image_data

        # Determine the pixel format for the data.
        if converted_image_data.shape[-1] == 3:
            # RGB image.
            if pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
                converted_image_data = convert_RGB_to_grayscale(
                    cv2.cvtColor(converted_image_data, cv2.COLOR_BGR2RGB)
                )
                image_proto.pixel_format = pixel_format
            else:
                image_proto.pixel_format = image_pb2.Image.PIXEL_FORMAT_RGB_U8
        elif converted_image_data.shape[-1] == 1:
            # Greyscale image.
            image_proto.pixel_format = image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8
        elif converted_image_data.shape[-1] == 4:
            # RGBA image.
            if pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
                converted_image_data = convert_RGB_to_grayscale(
                    cv2.cvtColor(converted_image_data, cv2.COLOR_BGRA2RGB)
                )
                image_proto.pixel_format = pixel_format
            else:
                image_proto.pixel_format = image_pb2.Image.PIXEL_FORMAT_RGBA_U8
        else:
            # The number of pixel channels did not match any of the known formats.
            image_proto.pixel_format = image_pb2.Image.PIXEL_FORMAT_DEPTH_U16

        # Note, we are currently not setting any information for the transform snapshot or the frame
        # name for an image sensor since this information can't be determined with openCV.

        resize_ratio = image_req.resize_ratio
        quality_percent = image_req.quality_percent

        if resize_ratio < 0 or resize_ratio > 1:
            raise ValueError(f"Resize ratio {resize_ratio} is out of bounds.")

        if resize_ratio != 1.0 and resize_ratio != 0:
            image_proto.rows = int(image_proto.rows * resize_ratio)
            image_proto.cols = int(image_proto.cols * resize_ratio)
            converted_image_data = cv2.resize(
                converted_image_data,
                (image_proto.cols, image_proto.rows),
                interpolation=cv2.INTER_AREA,
            )

        # Set the image data.
        if image_format == image_pb2.Image.FORMAT_RAW:
            image_proto.data = np.ndarray.tobytes(converted_image_data)
            image_proto.format = image_pb2.Image.FORMAT_RAW
        elif (
            image_format == image_pb2.Image.FORMAT_JPEG
            or image_format == image_pb2.Image.FORMAT_UNKNOWN
            or image_format is None
        ):
            # If the image format is requested as JPEG or if no specific image format is requested, return
            # a JPEG. Since this service is for a webcam, we choose a sane default for the return if the
            # request format is unpopulated.
            quality = (
                self.default_png_quality if self.mode else self.default_jpeg_quality
            )
            # A valid image quality percentage was passed with the image request,
            # so use this value instead of the service's default.
            encode_type: str = ".jpg"
            if self.mode:
                quality = (
                    quality_percent
                    if quality_percent > 0 and quality_percent <= 9
                    else quality
                )
                encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), int(quality)]
                encode_type = ".png"
            else:
                quality = (
                    quality_percent
                    if quality_percent > 0 and quality_percent <= 100
                    else quality
                )
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
                # encode_type = '.png'

            image_proto.data = cv2.imencode(
                encode_type, converted_image_data, encode_param
            )[1].tobytes()
            image_proto.format = image_pb2.Image.FORMAT_JPEG
            if self.socket:
                try:
                    self.socket.send(converted_image_data, 1, copy=True, track=False)
                except Exception as e:
                    print(f"Error while sending data from socket {str(e)}")
        else:
            # Unsupported format.
            raise Exception(
                f"Image format {image_pb2.Image.Format.Name(image_format)} is unsupported."
            )


if __name__ == "__main__":
    intelrealsensecapture = IntelRealSenseCameraInterface(source_name="realsensedepth")
    rgb_or_dpeth, cap_time = intelrealsensecapture.blocking_capture()
    print(rgb_or_dpeth.shape)
