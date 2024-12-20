# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import time

from spot_wrapper.spot import Spot


def main(spot: Spot):
    """Make Spot stand"""
    spot.power_on()
    spot.roll_over()

    # Wait 5 seconds to before powering down...
    while True:
        pass
    time.sleep(5)
    spot.power_off()


if __name__ == "__main__":
    spot = Spot("RolloverClient")
    with spot.get_lease(hijack=True) as lease:
        main(spot)
