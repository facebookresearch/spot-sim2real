import time

from siro import CommandTaskData, ConnectionStatus, FindObjectData, Sim, TaskState
from siro.data_types import PlaceObjectData
from siro_robot_connection.base_robot_bridge import BaseRobotBridge


class SimulationRobotBridge(BaseRobotBridge):

    find_object_start_time: float = 0
    find_object_state: int = 0
    place_object_start_time: float = 0
    place_object_state: int = 0

    def __init__(self):
        super().__init__()
        self.lease = None
        self.sim = Sim()
        self.last_time_recorded = 0

    def update(self):
        super().update()
        current_time = time.time()
        if self.last_time_recorded == 0:
            self.last_time_recorded = current_time
        time_delta2 = current_time - self.last_time_recorded
        self.last_time_recorded = current_time
        self.sim.internal_sim_tick(time_delta2)

    def get_trajectory_feedback(self, task: CommandTaskData):
        currently_moving, currently_rotating = self.sim.get_is_moving_is_rotating()
        self.update()
        if currently_moving or currently_rotating:
            task.set_state(TaskState.InProgress)
        else:
            task.set_state(TaskState.Success)

    def get_latest_xy_yaw(self):
        (
            self.robot_state.position.x,
            self.robot_state.position.y,
            self.robot_state.yaw,
        ) = self.sim.get_xy_yaw()

    def set_base_position(self, x, y, yaw):
        self.sim.set_base_position(x, y, yaw)
        self.sim.internal_sim_tick(0)

    def get_lease(self, hijack=False):
        self.robot_state.connection_status = ConnectionStatus.Connected
        return None

    def return_lease(self):
        self.robot_state.connection_status = ConnectionStatus.NotConnected

    def dock(self):
        print("Spot>>>Docking")
        time.sleep(0.2)
        print("Spot>>>Robot Docked")

    def undock(self):
        print("Spot>>>undocking")
        time.sleep(0.2)
        print("Spot>>>Robot undocked")

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

    def stand_up(self):
        print("Spot>>>Stand Up!")
        time.sleep(0.2)
        print("Spot>>>Robot Standing Up")

    def sit(self):
        print("Spot>>>Sitting Down!")
        time.sleep(0.2)
        print("Spot>>>Robot Sitting")

    def reset(self):
        super().reset()
        self.sim.reset()

    def get_find_object_feedback(self, task: FindObjectData):
        print("Spot>>>Finding object ")
        time.sleep(1)
        print("Spot>>>Found object ")
        print("Spot>>>Grabbing... ")
        time.sleep(2)
        print("Spot>>>Grabbed... ")
        task.set_state(TaskState.Success)

    def get_place_object_feedback(self, task: PlaceObjectData):
        print(
            "Spot>>>Moving hand at position ",
            task.arm_placement_position.x,
            task.arm_placement_position.y,
            task.arm_placement_position.z,
        )
        time.sleep(2)
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
