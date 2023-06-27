import re
import signal
import struct
import threading
import time

from bosdyn.client.estop import EstopClient
from spot_wrapper.estop import EstopNoGui
from spot_wrapper.spot import Spot
from spot_wrapper.utils import say

"""
To run without sudo, you need to run this command:
sudo gpasswd -a $USER input

and then reboot.
"""


class MyKeyEventClass2(object):
    def __init__(self):
        self.done = False
        signal.signal(signal.SIGINT, self.cleanup)

        with open("/proc/bus/input/devices") as f:
            devices_file_contents = f.read()

        # Spot-related code
        spot = Spot("HeadlessEstop")
        estop_client = spot.robot.ensure_client(EstopClient.default_service_name)
        self.estop_nogui = EstopNoGui(estop_client, 5, "Estop NoGUI")
        say("Headless e-stopping program initialized")

        for handlers in re.findall(
            r"""H: Handlers=([^\n]+)""", devices_file_contents, re.DOTALL
        ):
            dev_event_file = "/dev/input/event" + re.search(
                r"event(\d+)", handlers
            ).group(1)
            if "kbd" in handlers:
                t = threading.Thread(
                    target=self.read_events, kwargs={"dev_event_file": dev_event_file}
                )
                t.daemon = True
                t.start()

        while not self.done:  # Wait for Ctrl+C
            time.sleep(0.5)

    def cleanup(self, signum, frame):
        self.done = True

    def read_events(self, dev_event_file):
        print("Listening for kbd events on dev_event_file=" + str(dev_event_file))
        try:
            of = open(dev_event_file, "rb")
        except IOError as e:
            if e.strerror == "Permission denied":
                print(
                    "You don't have read permission on ({}). Are you root?".format(
                        dev_event_file
                    )
                )
                return
        while True:
            event_bin_format = (
                "llHHI"  # See kernel documentation for 'struct input_event'
            )
            # For details, read section 5 of this document:
            # https://www.kernel.org/doc/Documentation/input/input.txt
            data = of.read(struct.calcsize(event_bin_format))
            seconds, microseconds, e_type, code, value = struct.unpack(
                event_bin_format, data
            )
            full_time = seconds + microseconds / 1000000
            if e_type == 0x1:  # 0x1 == EV_KEY means key press or release.
                d = (
                    "RELEASE" if value == 0 else "PRESS"
                )  # value == 0 release, value == 1 press
                print(
                    "Got key "
                    + d
                    + " from "
                    + str(dev_event_file)
                    + ": t="
                    + str(full_time)
                    + "us type="
                    + str(e_type)
                    + " code="
                    + str(code)
                )

                # Spot-related code
                if d == "PRESS":
                    if 0:  # str(code) == "108":  # down
                        self.estop_nogui.settle_then_cut()
                        say("Activating e-stop")
                    elif str(code) == "103":  # up
                        self.estop_nogui.allow()
                        say("Releasing e-stop")


if __name__ == "__main__":
    try:
        a = MyKeyEventClass2()
    finally:
        say("Headless e-stopping program terminating.")
