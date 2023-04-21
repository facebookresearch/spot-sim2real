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
import os
import time

import numpy as np

from spot_wrapper.headless_estop import SpotHeadlessEstop

UPDATE_PERIOD = 0.2
LINEAR_VEL = 1.0
ANGULAR_VEL = np.deg2rad(50)
DOCK_ID = int(os.environ.get("SPOT_DOCK_ID", 520))
DEBUG = False


class FSM_ID:
    r"""Finite state machine IDs."""
    IDLE = 0  # robot is NOT in teleop, just listening for estop or teleop activation
    HALTED = 1  # robot in teleop, but not moving
    FORWARD = 2  # robot in teleop, moving forward
    BACK = 3  # robot in teleop, moving backwards
    LEFT = 4  # robot in teleop, turning left
    RIGHT = 5  # robot in teleop, turning right


class KEY_ID:
    r"""Keyboard id codes."""
    UP = 103
    DOWN = 108
    TAB = 15
    ENTER = 28
    UP_DOWN = 123123123  # virtual key representing both up and down pressed


class SpotHeadlessTeleop(SpotHeadlessEstop):
    silent = True
    name = "HeadlessTeleop"
    debug = DEBUG

    def __init__(self):
        self.fsm_state = FSM_ID.IDLE
        self.lease = None
        self.last_up = 0
        self.last_down = 0
        self.last_up_and_down = 0
        super().__init__()

    def process_pressed_key(self, pressed_key):
        if pressed_key not in [KEY_ID.UP, KEY_ID.DOWN, KEY_ID.ENTER, KEY_ID.TAB]:
            return

        # Handlers for when both up and down are pressed at the same time
        if pressed_key in [KEY_ID.UP, KEY_ID.DOWN]:
            if pressed_key == KEY_ID.DOWN:
                self.last_down = time.time()
            elif pressed_key == KEY_ID.UP:
                self.last_up = time.time()
            if abs(self.last_up - self.last_down) < 0.1:
                # Both keys have been pressed
                self.last_up, self.last_down = 0, 1
                pressed_key = KEY_ID.UP_DOWN
                # Double click of both keys detected
                double_click = time.time() - self.last_up_and_down < 0.4
                self.last_up_and_down = time.time()
                if double_click and self.fsm_state != FSM_ID.IDLE:
                    self.estop()
                    if not self.debug:
                        self.lease.return_lease()
                    self.fsm_state = FSM_ID.IDLE
                    print("Activating E-Stop! Entering IDLE mode.")
        if (
            self.last_up_and_down > 0
            and time.time() - self.last_up_and_down > 0.4
            and self.fsm_state != FSM_ID.IDLE
        ):
            self.last_up_and_down = 0
            print("Halting and docking robot...")
            self.halt_robot()
            try:
                if not self.debug:
                    self.spot.dock(DOCK_ID)
                    self.spot.home_robot()
                    self.lease.return_lease()
                    self.fsm_state = FSM_ID.IDLE
                    print("Docking successful! Entering IDLE mode.")
            except:
                print("Dock was not found!")
            return

        if self.fsm_state == FSM_ID.IDLE:
            if pressed_key == KEY_ID.UP:
                print("Releasing E-Stop.")
                self.release_estop()
            elif pressed_key == KEY_ID.DOWN:
                print("Activating E-Stop!")
                self.estop()
            elif pressed_key == KEY_ID.ENTER:
                print("Hijacking robot! Entering teleop mode.")
                self.fsm_state = FSM_ID.HALTED
                self.hijack_robot()
        else:
            if self.fsm_state != FSM_ID.HALTED:
                # Halt the robot if any key is pressed, and it's not idle
                print("Halting robot.")
                self.halt_robot()
                self.fsm_state = FSM_ID.HALTED
            elif pressed_key == KEY_ID.UP:
                print("Moving forwards.")
                self.move_forward()
                self.fsm_state = FSM_ID.FORWARD
            elif pressed_key == KEY_ID.DOWN:
                print("Moving backwards.")
                self.move_backwards()
                self.fsm_state = FSM_ID.BACK
            elif pressed_key == KEY_ID.TAB:
                print("Turning left.")
                self.turn_left()
                self.fsm_state = FSM_ID.LEFT
            elif pressed_key == KEY_ID.ENTER:
                print("Turning right.")
                self.turn_right()
                self.fsm_state = FSM_ID.RIGHT

    def hijack_robot(self):
        if self.debug:
            return
        self.lease = self.spot.get_lease(hijack=True)
        self.spot.power_on()
        self.spot.blocking_stand()

    def halt_robot(self, x_vel=0.0, ang_vel=0.0):
        if self.debug:
            return
        self.spot.set_base_velocity(x_vel, 0.0, ang_vel, vel_time=UPDATE_PERIOD * 2)

    def move_forward(self):
        self.halt_robot(x_vel=LINEAR_VEL)

    def move_backwards(self):
        self.halt_robot(x_vel=-LINEAR_VEL)

    def turn_left(self):
        self.halt_robot(ang_vel=ANGULAR_VEL)

    def turn_right(self):
        self.halt_robot(ang_vel=-ANGULAR_VEL)


if __name__ == "__main__":
    SpotHeadlessTeleop()
