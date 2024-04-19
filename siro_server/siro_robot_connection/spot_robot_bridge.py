import numpy as np
from spot_rl.utils.geometry_utils import (
    generate_intermediate_point,
    get_RPY_from_vector,
)

print(">>>>>>>>>>>>>>>>>>>>>>>>>")
from siro import (
    CommandTaskData,
    ConnectionStatus,
    Fiducial,
    RobotTrajectoryStatus,
    TaskState,
    Vector3,
)
from siro_robot_connection.base_robot_bridge import BaseRobotBridge
from spot_rl.envs.gaze_env import SpotGazeEnv
from spot_rl.envs.nav_env import SpotNavEnv
from spot_rl.envs.place_env import SpotPlaceEnv
from spot_rl.utils.construct_configs import (
    construct_config_for_gaze,
    construct_config_for_nav,
    construct_config_for_place,
)

print("<<<<<<<<<<<<<<<<<<<<")
import os
import time

from bosdyn.api import world_object_pb2
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    VISION_FRAME_NAME,
    get_a_tform_b,
    get_vision_tform_body,
)
from bosdyn.client.world_object import WorldObjectClient
from spot_rl.real_policy import GazePolicy, NavPolicy, PlacePolicy
from spot_wrapper.spot import Spot

DOCK_ID = int(os.environ.get("SPOT_DOCK_ID", 520))
env_device = "cuda"


class SpotRobotBridge(BaseRobotBridge):
    spot = None

    def __init__(self):
        super().__init__()

        try:
            self.spot = Spot("SIRO")
            self.world_object_client = self.spot.robot.ensure_client(
                WorldObjectClient.default_service_name
            )
        except Exception as error:
            self.robot_state.connection_status = ConnectionStatus.Error
            print(error)
        self.nav_config = construct_config_for_nav()
        self.place_config = construct_config_for_place()
        self.gaze_config = construct_config_for_gaze(max_episode_steps=350)
        self.lease = None

        self.nav_policy = None
        # try:
        print("Init Nav Policy")
        self.nav_policy = NavPolicy(
            self.nav_config.WEIGHTS.NAV, env_device, config=self.nav_config
        )
        # except Exception as error:
        #     print(error)

        self.gaze_policy = None
        # try:
        print("Init Gaze Policy")
        self.gaze_policy = GazePolicy(
            self.gaze_config.WEIGHTS.GAZE, env_device, config=self.gaze_config
        )
        # except Exception as error:
        #     print(error)

        self.place_policy = None
        # try:
        print("Init Place Policy")
        self.place_policy = PlacePolicy(
            self.place_config.WEIGHTS.PLACE, env_device, config=self.place_config
        )
        # except Exception as error:
        #     print(error)

    # https://dev.bostondynamics.com/docs/concepts/geometry_and_frames#frames-in-the-spot-robot-world
    def get_fiducial_world_objects(self):
        request_fiducials = [world_object_pb2.WORLD_OBJECT_APRILTAG]
        fiducial_objects = self.world_object_client.list_world_objects(
            object_type=request_fiducials
        ).world_objects
        fiducials = []
        for bd_fiducial in fiducial_objects:
            if bd_fiducial.apriltag_properties.fiducial_pose_status == 1:
                fiducial_position = self.get_position_from_fiducial_data(bd_fiducial)
                if fiducial_position is None:
                    continue
                fiducial = Fiducial(bd_fiducial.tag_id, fiducial_position)
                fiducials.append(fiducial)
        return fiducials

    def get_position_from_fiducial_data(self, fiducial):
        vision_tform_fiducial = get_a_tform_b(
            fiducial.transforms_snapshot,
            VISION_FRAME_NAME,
            fiducial.apriltag_properties.frame_name_fiducial,
        ).to_proto()
        if vision_tform_fiducial is not None:
            return vision_tform_fiducial.position
        return None

    def get_trajectory_feedback(self, task: CommandTaskData):
        if self.spot is not None:
            feedback_resp = self.spot.get_cmd_feedback(task.robot_command_id)
            cmd_status = (
                feedback_resp.feedback.synchronized_feedback.mobility_command_feedback
            ).se2_trajectory_feedback.status

            if cmd_status == RobotTrajectoryStatus.Unknown:
                task.set_state(TaskState.Error)
            elif cmd_status == RobotTrajectoryStatus.AtGoal:
                task.set_state(TaskState.Success)
            else:
                task.set_state(TaskState.InProgress)

    def get_pick_up_object_feedback(self, task: CommandTaskData):
        pass

    def get_find_object_feedback(self, task: CommandTaskData):
        pass

    def place_object_at_point(self, arm_placement_target):

        place_env = self.get_place_env()
        place_env.reset(arm_placement_target, False)

    def get_place_object_feedback(self, task: CommandTaskData):

        place_env = self.get_place_env()
        # End effector positions in base frame (as needed by the API)
        curr_ee_pos = place_env.get_gripper_position_in_base_frame_spot()
        goal_ee_pos = place_env.get_base_frame_place_target_spot()
        intr_ee_pos = generate_intermediate_point(curr_ee_pos, goal_ee_pos)

        # Get direction vector from current ee position to goal ee position for EE orientation
        dir_rpy_to_intr = get_RPY_from_vector(goal_ee_pos - curr_ee_pos)

        # Go to intermediate point
        self.spot.move_gripper_to_point(
            intr_ee_pos,
            dir_rpy_to_intr,
            self.place_config.ARM_TRAJECTORY_TIME_IN_SECONDS,
            timeout_sec=10,
        )

        # Direct the gripper to face downwards
        dir_rpy_to_goal = [0.0, np.pi / 2, 0.0]

        # Go to goal point
        self.spot.move_gripper_to_point(
            goal_ee_pos,
            dir_rpy_to_goal,
            self.place_config.ARM_TRAJECTORY_TIME_IN_SECONDS,
            timeout_sec=10,
        )
        self.spot.open_gripper()
        # Add sleep as open_gripper() is a non-blocking call
        time.sleep(1)
        task.set_state(TaskState.Success)

    def rotate_gripper_with_delta(self, wrist_yaw=0.0, wrist_roll=0.0):
        if self.spot is not None:
            self.spot.rotate_gripper_with_delta(wrist_yaw, wrist_roll)

    def move_gripper_to_point(
        self, point, rotation, seconds_to_goal=3.0, timeout_sec=10
    ):
        if self.spot is not None:
            self.spot.move_gripper_to_point(
                point, rotation, seconds_to_goal, timeout_sec
            )

    def open_gripper(self):
        if self.spot is not None:
            self.spot.open_gripper()

    def dock(self, home_robot=False):
        if self.spot is not None:
            self.spot.dock(DOCK_ID, home_robot)

    def undock(self):
        if self.spot is not None:
            print(
                "Please make sure robot is on the dock! This will first reset the home"
            )
            print(">>>>>>>>>>>>>>>>> Resetting Home for Spot (writing)")
            self.spot.home_robot()

            print(">>>>>>>>>>>>>>>>> Updating the home for spot (reading)")
            self.spot.recenter_origin_of_global_frame()

            self.spot.undock()

    def sit(self):
        if self.spot is not None:
            self.spot.sit()
            time.sleep(0.35)

    def get_latest_xy_yaw(self):
        if self.spot is not None and self.robot_state is not None:
            x, y, yaw = self.spot.get_xy_yaw()
            self.robot_state.set_x_y_yaw(x, y, yaw)

    def set_base_position(self, x, y, yaw):
        if self.spot is not None:
            self.spot.set_base_position(x, y, yaw, 100)

    def return_lease(self):

        if self.lease is not None:
            self.lease.__exit__()
        self.robot_state.connection_status = ConnectionStatus.NotConnected

    def get_lease(self, hijack=True):
        if self.robot_state.connection_status is not ConnectionStatus.Error:
            self.lease = self.spot.get_lease(hijack)
            print("Hijacking the lease")
            self.robot_state.connection_status = ConnectionStatus.Connected
        return self.lease

    def power_on(self):
        if self.spot is not None:
            self.spot.power_on()

    def power_off(self):
        if self.spot is not None:
            self.spot.power_off()

    def is_powered_on(self):
        return self.spot.is_powered_on()

    def stand_up(self):
        if self.spot is not None:
            self.spot.blocking_stand()
            time.sleep(0.35)

    def reset_policies(self):
        self.nav_policy.reset()
        self.gaze_policy.reset()
        self.place_policy.reset()

    def reset_arm(self, angles):
        if self.spot is not None:
            self.spot.close_gripper()
            return self.spot.set_arm_joint_positions(positions=angles, travel_time=0.75)
        else:
            return -1

    def reset(self):
        self.reset_policies()
        self.shutdown()

    def get_gaze_policy(self):
        return self.gaze_policy

    def get_place_policy(self):
        return self.place_policy

    def get_nav_policy(self):
        return self.nav_policy

    def get_navigation_env(self):
        return SpotNavEnv(config=self.nav_config, spot=self.spot)

    def get_place_env(self):
        return SpotPlaceEnv(self.place_config, self.spot)

    def get_gaze_env(self):
        return SpotGazeEnv(config=self.gaze_config, spot=self.spot)

    def shutdown(self, should_dock=False) -> None:
        if self.spot is None:
            return
        if self.lease is None:
            return
        try:
            if should_dock:
                print("Docking Spot")
                dock_start_time = time.time()
                while time.time() - dock_start_time < 2:
                    try:
                        self.dock()
                    except Exception:
                        print("Dock not found... trying again")
                        time.sleep(0.1)
            else:
                self.sit()
        finally:
            self.power_off()
