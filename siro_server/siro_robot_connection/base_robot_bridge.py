import time

from siro import CommandTaskData, ConnectionStatus, RobotState, Vector2


class BaseRobotBridge:
    robot_state = RobotState(Vector2(0, 0), 0, ConnectionStatus.Connected)

    def startup(self):
        self.get_lease()
        self.power_on()

    def sit(self):
        pass

    def dock(self):
        pass

    def undock(self):
        pass

    def update(self):
        pass

    def reset_policies(self):
        pass

    def get_trajectory_feedback(self, task: CommandTaskData):
        pass

    def get_latest_xy_yaw(self):
        pass

    def get_pick_up_object_feedback(self, task: CommandTaskData):
        pass

    def get_find_object_feedback(self, task: CommandTaskData):
        pass

    def get_place_object_feedback(self, task: CommandTaskData):
        pass

    def set_base_position(self, x, y, yaw):
        pass

    def get_lease(self, hijack=True):
        pass

    def power_on(self):
        pass

    def power_off(self):
        pass

    def is_powered_on(self):
        pass

    def stand_up(self):
        pass

    def return_lease(self):
        pass

    def reset(self):
        pass

    def rotate_gripper_with_delta(self, wrist_yaw=0.0, wrist_roll=0.0):
        pass

    def open_gripper(self):
        pass

    def look_for_objects(self, objects_to_find):
        pass

    def get_gaze_policy(self):
        pass

    def get_place_policy(self):
        pass

    def get_nav_policy(self):
        pass

    def get_navigation_env(self):
        pass

    def get_place_env(self):
        pass

    def get_gaze_env(self):
        pass

    def move_gripper_to_point(
        self, point, rotation, seconds_to_goal=3.0, timeout_sec=10
    ):
        pass

    def get_fiducial_world_objects(self):
        pass

    def place_object_at_point(self, place_target):
        pass

    def shutdown(self):
        pass

    def reset_arm(self, angles):
        time.sleep(3)
