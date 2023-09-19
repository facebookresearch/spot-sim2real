# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from spot_wrapper.spot import Spot

SQUARE_CENTER = np.array([0.5, 0.0, 0.6])
SQUARE_SIDE = 0.4
GRIPPER_WAYPOINTS = [
    SQUARE_CENTER,
    SQUARE_CENTER + np.array([0.0, 0.0, SQUARE_SIDE / 2]),
    SQUARE_CENTER + np.array([0.0, SQUARE_SIDE / 2, SQUARE_SIDE / 2]),
    SQUARE_CENTER + np.array([0.0, -SQUARE_SIDE / 2, SQUARE_SIDE / 2]),
    SQUARE_CENTER + np.array([0.0, -SQUARE_SIDE / 2, -SQUARE_SIDE / 2]),
    SQUARE_CENTER + np.array([0.0, SQUARE_SIDE / 2, -SQUARE_SIDE / 2]),
    SQUARE_CENTER,
]


def main(spot: Spot):
    spot.power_robot()

    # Open the gripper
    spot.open_gripper()

    # Move arm to initial configuration
    try:
        for point in GRIPPER_WAYPOINTS:
            spot.loginfo("TRAVELING TO WAYPOINT")
            success_status = spot.move_gripper_to_point(
                point, [0.0, 0.0, 0.0], timeout_sec=10
            )
            if success_status:
                spot.loginfo("REACHED WAYPOINT")
            else:
                spot.loginfo("FAILED TO REACH WAYPOINT")
    finally:
        spot.power_off()


if __name__ == "__main__":
    spot = Spot("DrawSquare")
    with spot.get_lease() as lease:
        main(spot)
