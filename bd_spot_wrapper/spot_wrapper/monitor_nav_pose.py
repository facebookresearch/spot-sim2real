# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import time

import numpy as np
from spot_wrapper.spot import Spot


def main(spot: Spot):
    while True:
        x, y, yaw = spot.get_xy_yaw()
        spot.loginfo(f"x: {x}, y: {y}, yaw: {np.rad2deg(yaw)}")
        time.sleep(1 / 30.0)


if __name__ == "__main__":
    spot = Spot("NavPoseMonitor")
    main(spot)
