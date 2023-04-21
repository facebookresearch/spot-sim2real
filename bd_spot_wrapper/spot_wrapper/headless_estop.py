"""
Controlling the robot using a presentation tool ("pt").
Program starts idle. In idle mode,
- Up releases the estop
- Down activates the estop
- Enter activates teleop mode

In teleop mode,
- Up and down will start moving the robot forward or backwards.
- Tab and Enter will start turning the robot left or right (respectively).
- Pressing any of these keys when the robot is moving will halt it.
- Pressing up and down simultaneously will make the robot dock.
- Double-pressing up and down simultaneously will estop the robot and the program idles.
"""

from bosdyn.client.estop import EstopClient

from spot_wrapper.estop import EstopNoGui
from spot_wrapper.spot import Spot
from spot_wrapper.utils.headless import KeyboardListener
from spot_wrapper.utils.utils import say

DEBUG = False


class KEY_ID:
    r"""Keyboard id codes."""
    UP = 103
    DOWN = 108


class SpotHeadlessEstop(KeyboardListener):
    name = "HeadlessEstop"
    debug = DEBUG

    def __init__(self):
        if not self.debug:
            self.spot = Spot(self.name)
            estop_client = self.spot.robot.ensure_client(
                EstopClient.default_service_name
            )
            self.estop_nogui = EstopNoGui(estop_client, 5, "Estop NoGUI")
        super().__init__()

    def process_pressed_key(self, pressed_key):
        if pressed_key == KEY_ID.UP:
            say("Releasing E-Stop.")
            self.release_estop()
        elif pressed_key == KEY_ID.DOWN:
            say("Activating E-Stop!")
            self.estop()

    def estop(self):
        if self.debug:
            return
        self.estop_nogui.settle_then_cut()

    def release_estop(self):
        if self.debug:
            return
        self.estop_nogui.allow()


if __name__ == "__main__":
    SpotHeadlessEstop()
