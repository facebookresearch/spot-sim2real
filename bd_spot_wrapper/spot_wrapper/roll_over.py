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
    with spot.get_lease() as lease:
        main(spot)
