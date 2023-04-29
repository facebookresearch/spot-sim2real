import time

from spot_wrapper.spot import Spot


def main(spot: Spot):
    """Make Spot stand"""
    spot.power_on()
    spot.blocking_stand()

    # Wait 3 seconds to before powering down...
    while True:
        pass
    time.sleep(3)
    spot.power_off()


if __name__ == "__main__":
    spot = Spot("BasicStandingClient")
    with spot.get_lease() as lease:
        main(spot)
