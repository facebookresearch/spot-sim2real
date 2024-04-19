import time

import siro
from siro_robot_connection.base_robot_bridge import BaseRobotBridge


class BaseCommandTask(object):
    def __init__(self, spot: BaseRobotBridge, task: siro.CommandTaskData, use_policy):
        self.spot = spot
        self.task = task
        self.use_policy = use_policy
        self.current_time = 0
        self.task.set_state(siro.TaskState.NoState)
        self.start_time = 0
        self.timeout_start = 0
        self.is_disposed = False
        self.is_resetting = False

    def start(self):
        # Try to power on the robot
        if not self.spot.is_powered_on():
            print("Robot is NOT powered on, will power it on!")
            self.spot.power_on()

        print(f"Starting task {self.task.state.task_type} | {self.use_policy}")
        self.current_time = 0
        self.task.set_state(siro.TaskState.InProgress)
        self.timeout_start = time.time()
        self.start_time = time.time()

    def update(self):
        pass

    def dispose(self):
        print(
            f"Task {self.task.state.task_type} completed in {(time.time() - self.start_time)}"
        )
        self.is_disposed = True

    def process_reset(self):
        self.is_resetting = False
