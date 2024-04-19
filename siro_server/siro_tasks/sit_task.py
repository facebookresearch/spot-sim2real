from siro import SitData, TaskState
from siro_robot_connection.base_robot_bridge import BaseRobotBridge
from siro_tasks.base_command_task import BaseCommandTask


class SitTask(BaseCommandTask):
    def __init__(self, spot: BaseRobotBridge, task: SitData, use_policy):
        super().__init__(spot, task, use_policy)
        self.timeout = 15
        self.reset_cmd = 0

    def start(self):
        super().start()
        print("sit_task in progress")

    def update(self):
        self.spot.sit()
        self.task.set_state(TaskState.Success)

    def dispose(self):
        super().dispose()
        print("sit_task disposed")
