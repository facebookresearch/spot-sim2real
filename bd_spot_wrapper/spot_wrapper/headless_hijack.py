"""
Hijacking the robot using a presentation tool ("pt").
Useful for stopping the robot without it sitting down on to rough pavement.
"""
from spot_wrapper.spot import Spot
from spot_wrapper.utils.headless import KeyboardListener
from spot_wrapper.utils.utils import say

DEBUG = False


class KEY_ID:
    r"""Keyboard id codes."""
    ENTER = 28


class SpotHeadlessHijack(KeyboardListener):
    name = "HeadlessHijack"
    debug = DEBUG

    def __init__(self):
        if not self.debug:
            self.spot = Spot(self.name)
        super().__init__()

    def process_pressed_key(self, pressed_key):
        if pressed_key == KEY_ID.ENTER:
            say("Hijacking lease!.")
            if self.debug:
                return
            with self.spot.get_lease(hijack=True):
                self.spot.power_on()
                self.spot.blocking_stand()
                while True:
                    pass


if __name__ == "__main__":
    SpotHeadlessHijack()
