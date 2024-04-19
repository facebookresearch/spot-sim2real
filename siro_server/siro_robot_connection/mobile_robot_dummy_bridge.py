import time

import numpy as np
from siro import (
    CommandTaskData,
    ConnectionStatus,
    FindObjectData,
    MobileRobotDummyData,
    TaskState,
)
from siro.data_types import PlaceObjectData, Vector3
from siro_robot_connection import BaseRobotBridge


class MobileRobotDummyBridge(BaseRobotBridge):

    NAV_DISTANCE_SUCCESS: float = 0.3
    find_object_start_time: float = 0
    find_object_state: int = 0
    place_object_start_time: float = 0
    place_object_state: int = 0

    def __init__(self, fake_robot_data: MobileRobotDummyData):
        super().__init__()
        self.lease = None
        self.fake_robot = fake_robot_data

    def set_base_position(self, x, y, yaw):
        self.fake_robot.set_base_position(x, y, yaw)

    def get_trajectory_feedback(self, task: CommandTaskData):
        xy = np.array([self.fake_robot.position.x, self.fake_robot.position.y])
        target_xy = np.array(
            [self.fake_robot.targetPosition.x, self.fake_robot.targetPosition.y]
        )
        dif = target_xy - xy
        mag = np.linalg.norm(dif)
        if mag <= self.NAV_DISTANCE_SUCCESS:
            task.set_state(TaskState.Success)
        else:
            task.set_state(TaskState.InProgress)

    def get_latest_xy_yaw(self):
        self.robot_state.position.x = self.fake_robot.position.x
        self.robot_state.position.y = self.fake_robot.position.y
        self.robot_state.yaw = self.fake_robot.yaw

    def get_lease(self, hijack=False):
        self.robot_state.connection_status = ConnectionStatus.Connected
        return None

    def return_lease(self):
        self.robot_state.connection_status = ConnectionStatus.NotConnected

    def power_on(self):
        print("Spot>>>Powering On")
        time.sleep(0.2)
        print("Spot>>>Robot On")

    def power_off(self):
        print("Spot>>>Powering Off")
        time.sleep(0.2)
        print("Spot>>>Robot Off")

    def is_powered_on(self):
        raise NotImplementedError

    def sit(self):
        print("Spot>>>Sitting Down!")
        time.sleep(0.2)
        print("Spot>>>Robot Sitting")

    def stand_up(self):
        print("Spot>>>Stand Up!")
        time.sleep(0.2)
        print("Spot>>>Robot Standing Up")

    def dock(self):
        print("Spot>>>Docking Robot!")
        time.sleep(0.2)
        print("Spot>>>Robot Docked")

    def undock(self):
        print("Spot>>>undocking")
        time.sleep(0.2)
        print("Spot>>>Robot undocked")

    def look_for_objects(self, objects_to_find):
        print(f"Spot>>>begin finding object {objects_to_find}")
        self.find_object_start_time = time.time()
        self.find_object_state = 0

    def get_find_object_feedback(self, task: FindObjectData):
        task.set_state(TaskState.InProgress)
        time.sleep(1)
        print("Spot>>>Found object ")
        time.sleep(1)
        print("Spot>>>Grabbing... ")
        time.sleep(1)
        print("Spot>>>Grabbed... ")
        time.sleep(1)
        print("Spot>>>Grabbed... ")
        task.set_state(TaskState.Success)

    def place_object_at_point(self, arm_placement_target):
        print(f"Spot>>>begin placing object {arm_placement_target}")
        self.place_object_start_time = time.time()
        self.place_object_state = 0

    def get_place_object_feedback(self, task: PlaceObjectData):
        task.set_state(TaskState.InProgress)
        time.sleep(1)
        print(
            "Spot>>>Moving hand at position ",
            task.arm_placement_position.x,
            task.arm_placement_position.y,
            task.arm_placement_position.z,
        )
        time.sleep(1)
        print("Spot>>>Dropped object!")
        task.set_state(TaskState.Success)

    def shutdown(self, should_dock=False) -> None:
        try:
            if should_dock:
                print("Docking Spot")
                time.sleep(1)
                print("Spot Docked")
            else:
                print("Sitting Spot")
                time.sleep(1)
                print("Spot Sat")
        finally:
            print("Spot Power Off")
