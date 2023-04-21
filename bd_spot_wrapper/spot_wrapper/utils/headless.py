"""
Controlling the robot using a presentation tool ("pt").
Up and down will control the linear velocity, or the angular velocity, depending
on the mode. Mode is controlled with Tab. Enter will hijack or return the robot's lease.
"""
import re
import signal
import struct
import threading
import time

"""
To run without sudo, you need to run this command:
sudo gpasswd -a $USER input

and then reboot the computer.
"""


class KeyboardListener(object):
    silent = False

    def __init__(self):
        self.done = False
        signal.signal(signal.SIGINT, self.cleanup)

        with open("/proc/bus/input/devices") as f:
            devices_file_contents = f.read()

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

        while not self.done:  #  Wait for Ctrl+C
            time.sleep(0.5)

    def cleanup(self, signum, frame):
        self.done = True

    def read_events(self, dev_event_file):
        print("Listening for kbd events on dev_event_file=" + str(dev_event_file))
        try:
            of = open(dev_event_file, "rb")
            while True:
                pressed_key = self.listen(of, dev_event_file)
                self.process_pressed_key(pressed_key)
        except IOError as e:
            if e.strerror == "Permission denied":
                print(
                    f"You don't have read permission on ({dev_event_file}). Are you "
                    f"root? Running the following should fix this issue:\n"
                    f"sudo gpasswd -a $USER input"
                )
                return

    def listen(self, of, dev_event_file):
        event_bin_format = "llHHI"  #  See kernel documentation for 'struct input_event'
        #  For details, read section 5 of this document:
        #  https://www.kernel.org/doc/Documentation/input/input.txt
        data = of.read(struct.calcsize(event_bin_format))
        seconds, microseconds, e_type, code, value = struct.unpack(
            event_bin_format, data
        )
        if e_type == 0x1:  #  0x1 == EV_KEY means key press or release.
            d = "RELEASE" if value == 0 else "PRESS"
            if not self.silent:
                print(
                    f"Got key {d} from {dev_event_file}: "
                    f"t={seconds + microseconds / 1000000}us type={e_type} code={code}"
                )
            if d == "PRESS":
                return code
        return None

    def process_pressed_key(self, pressed_key):
        pass


if __name__ == "__main__":
    a = KeyboardListener()
