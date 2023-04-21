import time

import numpy as np
from spot_wrapper.spot import Spot


def main(spot: Spot):
    while True:
        arm_prop = spot.get_arm_proprioception()
        current_joint_positions = np.array(
            [v.position.value for v in arm_prop.values()]
        )
        spot.loginfo(", ".join([str(i) for i in np.rad2deg(current_joint_positions)]))
        spot.loginfo([v.name for v in arm_prop.values()])
        time.sleep(1 / 30)


if __name__ == "__main__":
    spot = Spot("ArmJointControl")
    main(spot)
