from siro import TaskState
from siro.data_types import StandData
from siro_robot_connection.base_robot_bridge import BaseRobotBridge
from siro_tasks.base_command_task import BaseCommandTask


class StandTask(BaseCommandTask):
    def __init__(self, spot: BaseRobotBridge, task: StandData, use_policy):
        super().__init__(spot, task, use_policy)
        self.timeout = 15
        self.reset_cmd = 0

    def start(self):
        super().start()
        print("stand_task in progress")

    def update(self):
        self.spot.stand_up()
        self.task.set_state(TaskState.Success)

    def dispose(self):
        super().dispose()
        print("stand_task disposed")
