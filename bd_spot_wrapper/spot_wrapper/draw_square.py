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
    spot.power_on()
    spot.blocking_stand()

    # Open the gripper
    spot.open_gripper()

    # Move arm to initial configuration
    try:
        for point in GRIPPER_WAYPOINTS:
            spot.loginfo("TRAVELING TO WAYPOINT")
            cmd_id = spot.move_gripper_to_point(point, [0.0, 0.0, 0.0])
            spot.block_until_arm_arrives(cmd_id, timeout_sec=10)
            spot.loginfo("REACHED WAYPOINT")
    finally:
        spot.power_off()


if __name__ == "__main__":
    spot = Spot("DrawSquare")
    with spot.get_lease() as lease:
        main(spot)
