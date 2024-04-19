import time

from siro import FindObjectData, TaskState
from siro_robot_connection.base_robot_bridge import BaseRobotBridge
from siro_tasks.base_command_task import BaseCommandTask


class FindObjectTask(BaseCommandTask):
    def __init__(self, spot: BaseRobotBridge, task: FindObjectData, use_policy):
        super().__init__(spot, task, use_policy)
        self.pick_up_data = task
        self.env = None
        self.observations = None
        self.timeout = 30

    def start(self):
        super().start()
        if not self.use_policy:
            self.task.set_state(TaskState.InProgress)
            self.spot.look_for_objects(self.pick_up_data.objects_to_find)
        else:
            self.spot.get_gaze_policy().reset()
            self.env = self.spot.get_gaze_env()
            self.observations = self.env.reset(
                target_obj_name=self.pick_up_data.objects_to_find[0]
            )

    def update(self):
        super().update()
        current_task_time = time.time() - self.start_time
        if not self.use_policy:
            self.spot.get_find_object_feedback(self.task)
            self.spot.get_latest_xy_yaw()
        else:
            action_dict = {}
            action_dict["arm_action"] = self.spot.get_gaze_policy().act(
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
        if current_task_time >= self.timeout:
            print(
                f"Task Failure *timeout* {self.task.state.task_type} | {self.use_policy}"
            )
            self.task.set_state(TaskState.Fail)

    def dispose(self):
        super().dispose()
        self.observations = None
        self.env = None
        print("place_object_task disposed")
