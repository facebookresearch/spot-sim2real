import threading
import time
from queue import Queue

from siro import TaskState
from siro_robot_connection import BaseRobotBridge
from siro_tasks import BaseCommandTask


class CommandTaskRunnerThread(threading.Thread):
    started_issuing_command = False
    current_task_creation = -1
    stop_thread = False

    def __init__(self, command_processor=None, name="command-runner-thread"):
        self.command_processor = command_processor
        super(CommandTaskRunnerThread, self).__init__(name=name)
        self.start()

    def run(self):
        while not self.stop_thread:
            self.command_processor()

    def shutdown(self):
        self.stop_thread = True


class CommandTaskRunner:
    def __init__(self, command_queue: Queue, spot: BaseRobotBridge):
        self.command_queue = command_queue
        self.spot = spot
        self.current_task: BaseCommandTask = None
        self.kthread = CommandTaskRunnerThread(self.process_commands)

    def dispose(self):
        if self.kthread is not None:
            self.kthread.shutdown()
            self.kthread = None
            self.current_task = None

    def process_commands(self):
        if self.current_task is not None:
            current_state = self.current_task.task.get_state()
            if current_state == TaskState.InProgress:
                self.current_task.update()
            elif current_state == TaskState.Success or current_state == TaskState.Fail:
                self.current_task.dispose()
                self.current_task = None
        elif self.command_queue.qsize() > 0:
            self.current_task = self.command_queue.get()
            print(f"Starting command task {self.current_task}")
            self.current_task.start()
        else:
            self.spot.get_latest_xy_yaw()
            time.sleep(0.1)
