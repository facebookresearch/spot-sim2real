# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from spot_wrapper.spot import Spot
from spot_wrapper.spot_vr_control import Spot_VR_Controller


def main(spot: Spot):
    """Let Spot follow the VR joystick"""
    # Init spot vr controller
    spot_vr_controller = Spot_VR_Controller(spot)

    try:
        while True:
            spot_vr_controller.track_vr()
    finally:
        print("done")


if __name__ == "__main__":
    spot = Spot("ArmKeyboardTeleop")
    with spot.get_lease(hijack=True) as lease:
        main(spot)
