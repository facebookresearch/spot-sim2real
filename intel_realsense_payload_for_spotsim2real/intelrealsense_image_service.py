# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Reference code IntelRS : https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/align-depth2color.py
# Reference code Boston Dynamics Spot-sdk: https://github.com/boston-dynamics/spot-sdk/tree/master/python/examples/web_cam_image_service

"""Register and run the Intel Cam Service."""

import logging

import bosdyn.util
from bosdyn.api import image_service_pb2_grpc
from bosdyn.client.directory_registration import (
    DirectoryRegistrationClient,
    DirectoryRegistrationKeepAlive,
)
from bosdyn.client.image_service_helpers import (
    CameraBaseImageServicer,
    VisualImageSource,
)
from bosdyn.client.server_util import GrpcServiceRunner
from bosdyn.client.util import setup_logging
from IntelRealSenseCameraInterface import (
    IntelRealSenseCameraInterface,
    create_realsense_pipeline,
)

DIRECTORY_NAME = "intel-realsense-image-service"
AUTHORITY = "robot-intelrealsense-stream"
SERVICE_TYPE = "bosdyn.api.ImageService"

_LOGGER = logging.getLogger(__name__)


def make_realsense_image_service(
    bosdyn_sdk_robot,
    service_name,
    show_debug_information=False,
    logger=None,
    res_width=640,
    res_height=480,
    jpeg_quality=75,
    png_quality=9,
    socket=None,
    filter_depth: bool = False,
):
    image_sources = []
    source_names = ["intelrealsensergb", "intelrealsensedepth"]

    try:
        (
            pipeline,
            profile,
            align,
            intrinsics,
            depth_scale,
            filters,
        ) = create_realsense_pipeline(res_width, res_height, filter_depth=filter_depth)
    except Exception as e:
        print(f"Unable to get intelrealsense working exiting dur to {str(e)}")
        exit(0)

    for source_name in source_names:
        # This Interface Class has two main methods, capturing the image/depth from source & encoding (compression, format conversion) the data for GRPC response
        intel_cam = IntelRealSenseCameraInterface(
            source_name,
            pipeline,
            profile,
            align,
            jpeg_quality=jpeg_quality,
            png_quality=png_quality,
            show_debug_information=show_debug_information,
            res_width=res_width,
            res_height=res_height,
            socket=socket,
            filters=filters,
        )
        # VisualImageSource will take care of GRPC request response parsing
        img_src = VisualImageSource(
            intel_cam.image_source_name,
            intel_cam,
            rows=intel_cam.rows,
            cols=intel_cam.cols,
            gain=intel_cam.camera_gain,
            exposure=intel_cam.camera_exposure,
            pixel_formats=intel_cam.supported_pixel_formats,
        )
        # TODO: Add caliberation data (added) & transforms tree, Can also add depth filtering if needed
        # will save processing at client end
        img_src.image_source_proto.pinhole.intrinsics.focal_length.x = intrinsics.fx
        img_src.image_source_proto.pinhole.intrinsics.focal_length.y = intrinsics.fy
        img_src.image_source_proto.pinhole.intrinsics.principal_point.x = intrinsics.ppx
        img_src.image_source_proto.pinhole.intrinsics.principal_point.y = intrinsics.ppy
        img_src.image_source_proto.depth_scale = 1.0 / depth_scale
        image_sources.append(img_src)
    return CameraBaseImageServicer(
        bosdyn_sdk_robot, service_name, image_sources, logger
    )


def run_service(
    bosdyn_sdk_robot,
    port,
    service_name,
    show_debug_information=False,
    logger=None,
    res_width=640,
    res_height=480,
    jpeg_quality=75,
    png_quality=9,
    socket=None,
    filter_depth: bool = False,
):
    # Proto service specific function used to attach a servicer to a server.
    add_servicer_to_server_fn = (
        image_service_pb2_grpc.add_ImageServiceServicer_to_server
    )

    # Instance of the servicer to be run.
    service_servicer = make_realsense_image_service(
        bosdyn_sdk_robot,
        service_name,
        show_debug_information,
        logger=logger,
        res_width=res_width,
        res_height=res_height,
        jpeg_quality=jpeg_quality,
        png_quality=png_quality,
        socket=socket,
        filter_depth=filter_depth,
    )
    return GrpcServiceRunner(
        service_servicer, add_servicer_to_server_fn, port, logger=logger
    )


def add_stream_arguments(parser):

    parser.add_argument(
        "--show-debug-info",
        action="store_true",
        required=False,
        help="If passed, openCV will try to display the captured web cam images.",
    )

    parser.add_argument(
        "--res-width",
        required=False,
        type=int,
        default=640,
        help="Resolution width (pixels).",
    )
    parser.add_argument(
        "--res-height",
        required=False,
        type=int,
        default=480,
        help="Resolution height (pixels).",
    )
    parser.add_argument(
        "--jpeg-quality",
        required=False,
        type=int,
        default=75,
        help="Jpeg cv2 Image Quality for RGB between 0 to 100",
    )
    parser.add_argument(
        "--png-quality",
        required=False,
        type=int,
        default=0,
        help="Png cv2 Image Quality for Depth between 0 to 9",
    )

    parser.add_argument(
        "--socket-verification",
        action="store_true",
        required=False,
        help="If passed, images will be also sent over socket for further verification.",
    )

    parser.add_argument(
        "--filter-depth",
        action="store_true",
        required=False,
        help="If passed, depth images will be filtered, default is False",
    )


if __name__ == "__main__":
    # Define all arguments used by this service.
    import argparse

    parser = argparse.ArgumentParser(allow_abbrev=False)
    bosdyn.client.util.add_base_arguments(parser)
    bosdyn.client.util.add_payload_credentials_arguments(parser)
    bosdyn.client.util.add_service_endpoint_arguments(parser)
    add_stream_arguments(parser)
    options = parser.parse_args()

    # Setup logging to use either INFO level or DEBUG level.
    setup_logging(options.verbose, include_dedup_filter=True)

    # Create and authenticate a bosdyn robot object.
    sdk = bosdyn.client.create_standard_sdk("ImageServiceSDK")

    robot = sdk.create_robot(options.hostname)

    robot.authenticate_from_payload_credentials(
        *bosdyn.client.util.get_guid_and_secret(options)
    )

    socket = None
    if options.socket_verification:
        import zmq

        port = "21001"
        context = zmq.Context()
        socket = context.socket(zmq.PUSH)
        socket.bind("tcp://*:%s" % port)

    # Create a service runner to start and maintain the service on background thread.
    service_runner = run_service(
        robot,
        options.port,
        DIRECTORY_NAME,
        options.show_debug_info,
        logger=_LOGGER,
        res_width=options.res_width,
        res_height=options.res_height,
        jpeg_quality=options.jpeg_quality,
        png_quality=options.png_quality,
        socket=socket,
        filter_depth=options.filter_depth,
    )

    # Use a keep alive to register the service with the robot directory.
    dir_reg_client = robot.ensure_client(
        DirectoryRegistrationClient.default_service_name
    )
    keep_alive = DirectoryRegistrationKeepAlive(dir_reg_client, logger=_LOGGER)
    keep_alive.start(
        DIRECTORY_NAME, SERVICE_TYPE, AUTHORITY, options.host_ip, service_runner.port
    )

    # Attach the keep alive to the service runner and run until a SIGINT is received.
    with keep_alive:
        service_runner.run_until_interrupt()
