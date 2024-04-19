import time

import numpy as np
from siro import TaskState
from siro.data_types import PlaceObjectData
from siro_robot_connection.base_robot_bridge import BaseRobotBridge
from siro_tasks.base_command_task import BaseCommandTask


class PlaceObjectTask(BaseCommandTask):
    def __init__(self, spot: BaseRobotBridge, task: PlaceObjectData, use_policy):
        super().__init__(spot, task, use_policy)
        self.place_object_data = task
        self.env = None
        self.observations = None
        self.timeout = 30
        self.reset_cmd = 0

    def start(self):
        super().start()
        print("place_object_task in progress")
        arm_target = self.place_object_data.arm_placement_position
        place_target = [arm_target.x, arm_target.y, arm_target.z]
        if not self.use_policy:
            self.spot.place_object_at_point(place_target)
        else:
            self.spot.get_place_policy().reset()
            self.env = self.spot.get_place_env()
            self.observations = self.env.reset(place_target, False)

    def update(self):
        current_task_time = time.time() - self.start_time
        if not self.use_policy:
            self.spot.get_place_object_feedback(self.task)
        else:
            action_dict = {}
            action_dict["arm_action"] = self.spot.get_place_policy().act(
                self.observations
            )
            self.observations, _, done, info = self.env.step(action_dict=action_dict)
            if self.env.get_success(self.observations):
                print("place_object_task success")
                self.reset_cmd = self.spot.reset_arm(self.env.initial_arm_joint_angles)
                self.task.set_state(TaskState.Success)
            if current_task_time >= self.timeout:
                print("place_object_task fail due to timeout")
                self.task.set_state(TaskState.Fail)
                print("place_object_task::dispose:open gripper")
                self.spot.open_gripper()
                time.sleep(1)
                print("place_object_task::dispose:reset arm")
                self.reset_cmd = self.spot.reset_arm(self.env.initial_arm_joint_angles)
            x = self.env.x
            y = self.env.y
            yaw = self.env.yaw
            self.spot.robot_state.set_x_y_yaw(x, y, yaw)
        if current_task_time >= self.timeout:
            print(f"Task Failure {self.task.state.task_type} | {self.use_policy}")
            self.task.set_state(TaskState.Fail)

    def dispose(self):
        super().dispose()
        self.observations = None
        self.env = None
        print("place_object_task disposed")
