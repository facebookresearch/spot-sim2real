# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Reference code IntelRS : https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/align-depth2color.py
# Reference code Boston Dynamics Spot-sdk: https://github.com/boston-dynamics/spot-sdk/tree/master/python/examples/web_cam_image_service

"""
Example code for using the payload service API
"""
import argparse
import io
import logging
import os
import struct
import sys
import time

import bosdyn.api.payload_pb2 as payload_protos
import bosdyn.api.robot_id_pb2 as robot_id_protos
import bosdyn.client
import bosdyn.client.util
from bosdyn.client.payload import PayloadClient
from bosdyn.client.payload_registration import (
    PayloadRegistrationClient,
    PayloadRegistrationKeepAlive,
)

LOGGER = logging.getLogger()
SPOT_ADMIN_PW = os.environ["SPOT_ADMIN_PW"]
SPOT_IP = os.environ["SPOT_IP"]


def payload_spot(options):
    """A simple example of using the Boston Dynamics API to communicate payload configs with spot.

    First registers a payload then lists all payloads on robot, including newly registered payload.
    """

    sdk = bosdyn.client.create_standard_sdk("PayloadSpotClient")

    robot = sdk.create_robot(SPOT_IP)

    # Authenticate robot before being able to use it
    robot.authenticate("admin", SPOT_ADMIN_PW)
    bosdyn.client.util.authenticate(robot)

    # Create a payload registration client
    payload_registration_client = robot.ensure_client(
        PayloadRegistrationClient.default_service_name
    )

    # Create a payload
    payload = payload_protos.Payload()
    payload.GUID = options.guid_to_register  # "78b076a2-b4ba-491d-a099-738928c4410c"
    payload_secret = "spot-sim2realteamsiro"
    payload.name = "IntelRealSenseAuxillaryImageServicePayload"
    payload.description = "This payload was created to stream IntelRealSense feed when gripper camera is obstructed."
    payload.label_prefix.append("IntelRealSenseStreamer")
    payload.is_authorized = False
    payload.is_enabled = False
    payload.is_noncompute_payload = False
    payload.version.major_version = 1
    payload.version.minor_version = 1
    payload.version.patch_level = 1
    # note: this field is not required, but highly recommended
    payload.mount_frame_name = payload_protos.MOUNT_FRAME_BODY_PAYLOAD

    # Register the payload
    payload_registration_client.register_payload(payload, payload_secret)

    # Create a payload client
    payload_client = robot.ensure_client(PayloadClient.default_service_name)

    # Update the payload version
    version = robot_id_protos.SoftwareVersion()
    version.major_version = 2
    version.minor_version = 2
    version.patch_level = 2
    payload_registration_client.update_payload_version(
        payload.GUID, payload_secret, version
    )

    # List all payloads
    payloads = payload_client.list_payloads()
    print("\n\nPayload Listing\n" + "-" * 40)
    print(payloads)
    print(
        "Payload Registered please approve this payload from Payload Section in WebAdmin of SPOT https://SPOT_IP/"
    )


def main(options):
    """Command line interface."""
    payload_spot(options)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--guid-to-register",
        required=False,
        type=str,
        default="78b076a2-b4ba-491d-a099-738928c4410c",
        help="Spot recognizable GUID, preferrably generated from Spot webUI",
    )
    options = parser.parse_args()
    if not main(options):
        sys.exit(1)
