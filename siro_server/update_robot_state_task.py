import threading
import time
from queue import Queue

from siro import TaskState
from siro_robot_connection import BaseRobotBridge
from siro_tasks import BaseCommandTask


class UpdateRobotStateTaskThread(threading.Thread):
    started_issuing_command = False
    current_task_creation = -1
    stop_thread = False

    def __init__(self, command_processor=None, name="update-robot-state-thread"):
        self.command_processor = command_processor
        super(UpdateRobotStateTaskThread, self).__init__(name=name)
        self.start()

    def run(self):
        while not self.stop_thread:
            self.command_processor()

    def shutdown(self):
        self.stop_thread = True


class UpdateRobotStateTask:
    def __init__(self, spot: BaseRobotBridge):
        self.position_update_rate = 1.0 / 30.0
        self.spot = spot
        self.kthread = UpdateRobotStateTaskThread(self.update)

    def dispose(self):
        if self.kthread is not None:
            self.kthread.shutdown()
            self.kthread = None

    def update(self):
        self.spot.get_latest_xy_yaw()
        time.sleep(self.position_update_rate)
