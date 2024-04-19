import time

import numpy as np
from siro import NavigateToData, TaskState
from siro_robot_connection.base_robot_bridge import BaseRobotBridge
from siro_tasks.base_command_task import BaseCommandTask


class NavigateToTask(BaseCommandTask):
    def __init__(self, spot: BaseRobotBridge, task: NavigateToData, use_policy):
        super().__init__(spot, task, use_policy)
        self.navigate_to_data = task
        self.base_action = None
        self.env = None
        self.observations = None
        self.timeout = 300

    def start(self):
        super().start()
        print(
            f"starting navigation:: {self.navigate_to_data.position.x} | {self.navigate_to_data.position.y} | {self.navigate_to_data.yaw}"
        )
        if not self.use_policy:
            self.spot.set_base_position(
                self.navigate_to_data.position.x,
                self.navigate_to_data.position.y,
                self.navigate_to_data.yaw,
            )
        else:
            self.spot.get_nav_policy().reset()
            self.env = self.spot.get_navigation_env()
            angle = np.deg2rad(self.navigate_to_data.yaw)
            self.observations = self.env.reset(
                np.array(
                    [self.navigate_to_data.position.x, self.navigate_to_data.position.y]
                ),
                angle,
            )

    def update(self):
        super().update()
        current_task_time = time.time() - self.start_time
        if self.use_policy:
            action_dict = {}
            action_dict["base_action"] = self.spot.get_nav_policy().act(
                self.observations
            )
            self.observations, _, done, info = self.env.step(action_dict=action_dict)
            if self.env.get_success(self.observations):
                print(f"Task Success {self.task.state.task_type} | {self.use_policy}")
                self.task.set_state(TaskState.Success)
            elif done:
                print(
                    f"Received 'done' on task but with no success!? {self.task.state.task_type} | {self.use_policy}"
                )
                self.task.set_state(TaskState.Fail)
            x = self.env.x
            y = self.env.y
            yaw = self.env.yaw
            self.spot.robot_state.set_x_y_yaw(x, y, yaw)
        else:
            self.spot.get_trajectory_feedback(self.task)
            self.spot.get_latest_xy_yaw()
        if current_task_time >= self.timeout:
            print(f"Task Failure {self.task.state.task_type} | {self.use_policy}")
            self.task.set_state(TaskState.Fail)

    def dispose(self):
        super().dispose()
        self.observations = None
        self.base_action = None
        self.env = None
        print("NavigateToTask disposed")
