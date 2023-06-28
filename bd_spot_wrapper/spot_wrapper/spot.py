# mypy: ignore-errors
# Copyright (c) 2021 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

""" Easy-to-use wrapper for properly controlling Spot """
import os
import os.path as osp
import time
from collections import OrderedDict

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import cv2
import numpy as np
from bosdyn import geometry
from bosdyn.api import (
    arm_command_pb2,
    basic_command_pb2,
    geometry_pb2,
    image_pb2,
    manipulation_api_pb2,
    robot_command_pb2,
    synchronized_command_pb2,
    trajectory_pb2,
)
from bosdyn.api.geometry_pb2 import SE2Velocity, SE2VelocityLimit, Vec2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import math_helpers
from bosdyn.client.docking import blocking_dock_robot, blocking_undock
from bosdyn.client.frame_helpers import (
    GRAV_ALIGNED_BODY_FRAME_NAME,
    HAND_FRAME_NAME,
    VISION_FRAME_NAME,
    get_vision_tform_body,
)
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import (
    RobotCommandBuilder,
    RobotCommandClient,
    block_until_arm_arrives,
    blocking_selfright,
    blocking_stand,
)
from bosdyn.client.robot_state import RobotStateClient
from google.protobuf import wrappers_pb2

# Get Spot password and IP address
env_err_msg = (
    "\n{var_name} not found as an environment variable!\n"
    "Please run:\n"
    "echo 'export {var_name}=<YOUR_{var_name}>' >> ~/.bashrc\nor for MacOS,\n"
    "echo 'export {var_name}=<YOUR_{var_name}>' >> ~/.bash_profile\n"
    "Then:\nsource ~/.bashrc\nor\nsource ~/.bash_profile"
)
try:
    SPOT_ADMIN_PW = os.environ["SPOT_ADMIN_PW"]
except KeyError:
    raise RuntimeError(env_err_msg.format(var_name="SPOT_ADMIN_PW"))
try:
    SPOT_IP = os.environ["SPOT_IP"]
except KeyError:
    raise RuntimeError(env_err_msg.format(var_name="SPOT_IP"))

ARM_6DOF_NAMES = [
    "arm0.sh0",
    "arm0.sh1",
    "arm0.el0",
    "arm0.el1",
    "arm0.wr0",
    "arm0.wr1",
]

HOME_TXT = osp.join(osp.dirname(osp.abspath(__file__)), "home.txt")


class SpotCamIds:
    r"""Enumeration of types of cameras."""

    BACK_DEPTH = "back_depth"
    BACK_DEPTH_IN_VISUAL_FRAME = "back_depth_in_visual_frame"
    BACK_FISHEYE = "back_fisheye_image"
    FRONTLEFT_DEPTH = "frontleft_depth"
    FRONTLEFT_DEPTH_IN_VISUAL_FRAME = "frontleft_depth_in_visual_frame"
    FRONTLEFT_FISHEYE = "frontleft_fisheye_image"
    FRONTRIGHT_DEPTH = "frontright_depth"
    FRONTRIGHT_DEPTH_IN_VISUAL_FRAME = "frontright_depth_in_visual_frame"
    FRONTRIGHT_FISHEYE = "frontright_fisheye_image"
    HAND_COLOR = "hand_color_image"
    HAND_COLOR_IN_HAND_DEPTH_FRAME = "hand_color_in_hand_depth_frame"
    HAND_DEPTH = "hand_depth"
    HAND_DEPTH_IN_HAND_COLOR_FRAME = "hand_depth_in_hand_color_frame"
    HAND = "hand_image"
    LEFT_DEPTH = "left_depth"
    LEFT_DEPTH_IN_VISUAL_FRAME = "left_depth_in_visual_frame"
    LEFT_FISHEYE = "left_fisheye_image"
    RIGHT_DEPTH = "right_depth"
    RIGHT_DEPTH_IN_VISUAL_FRAME = "right_depth_in_visual_frame"
    RIGHT_FISHEYE = "right_fisheye_image"


# CamIds that need to be rotated by 270 degrees in order to appear upright
SHOULD_ROTATE = [
    SpotCamIds.FRONTLEFT_DEPTH,
    SpotCamIds.FRONTRIGHT_DEPTH,
    SpotCamIds.HAND_DEPTH,
    SpotCamIds.HAND,
]


class Spot:
    def __init__(self, client_name_prefix):
        bosdyn.client.util.setup_logging()
        sdk = bosdyn.client.create_standard_sdk(client_name_prefix)
        robot = sdk.create_robot(SPOT_IP)
        robot.authenticate("admin", SPOT_ADMIN_PW)
        robot.time_sync.wait_for_sync()
        self.robot = robot
        self.command_client = None
        self.spot_lease = None

        # Get clients
        self.command_client = robot.ensure_client(
            RobotCommandClient.default_service_name
        )
        self.image_client = robot.ensure_client(ImageClient.default_service_name)
        self.manipulation_api_client = robot.ensure_client(
            ManipulationApiClient.default_service_name
        )
        self.robot_state_client = robot.ensure_client(
            RobotStateClient.default_service_name
        )

        # Used to re-center origin of global frame
        if osp.isfile(HOME_TXT):
            with open(HOME_TXT) as f:
                data = f.read()
            self.global_T_home = np.array([float(d) for d in data.split(", ")[:9]])
            self.global_T_home = self.global_T_home.reshape([3, 3])
            self.robot_recenter_yaw = float(data.split(", ")[-1])
        else:
            self.global_T_home = None
            self.robot_recenter_yaw = None

        # Print the battery charge level of the robot
        self.loginfo(f"Current battery charge: {self.get_battery_charge()}%")

    def get_lease(self, hijack=False):
        # Make sure a lease for this client isn't already active
        assert self.spot_lease is None
        self.spot_lease = SpotLease(self, hijack=hijack)
        return self.spot_lease

    def get_cmd_feedback(self, cmd_id):
        return self.command_client.robot_command_feedback(cmd_id)

    def is_estopped(self):
        return self.robot.is_estopped()

    def power_on(self, timeout_sec=20):
        self.robot.power_on(timeout_sec=timeout_sec)
        assert self.robot.is_powered_on(), "Robot power on failed."
        self.loginfo("Robot powered on.")

    def power_off(self, cut_immediately=False, timeout_sec=20):
        self.loginfo("Powering robot off...")
        self.robot.power_off(cut_immediately=cut_immediately, timeout_sec=timeout_sec)
        assert not self.robot.is_powered_on(), "Robot power off failed."
        self.loginfo("Robot safely powered off.")

    def blocking_stand(self, timeout_sec=10):
        self.loginfo("Commanding robot to stand (blocking)...")
        blocking_stand(self.command_client, timeout_sec=timeout_sec)
        self.loginfo("Robot standing.")

    def stand(self, timeout_sec=10):
        stand_command = RobotCommandBuilder.synchro_stand_command()
        cmd_id = self.command_client.robot_command(stand_command, timeout=timeout_sec)
        return cmd_id

    def blocking_selfright(self, timeout_sec=20):
        self.loginfo("Commanding robot to self-right (blocking)...")
        blocking_selfright(self.command_client, timeout_sec=timeout_sec)
        self.loginfo("Robot has self-righted.")

    def loginfo(self, *args, **kwargs):
        self.robot.logger.info(*args, **kwargs)

    def open_gripper(self):
        """Does not block, be careful!"""
        gripper_command = RobotCommandBuilder.claw_gripper_open_command()
        self.command_client.robot_command(gripper_command)

    def close_gripper(self):
        """Does not block, be careful!"""
        gripper_command = RobotCommandBuilder.claw_gripper_close_command()
        self.command_client.robot_command(gripper_command)

    def rotate_gripper_with_delta(self, wrist_yaw=0.0, wrist_roll=0.0):
        """
        Takes in relative wrist rotations targets and moves each wrist joint to the corresponding target.
        Waits for 0.5 sec after issuing motion command
        :param wrist_yaw: relative yaw for wrist in radians
        :param wrist_roll: relative roll for wrist in radians
        """
        print(
            f"Rotating the wrist with the following relative rotations: yaw={wrist_yaw}, roll={wrist_roll}"
        )

        arm_joint_positions = self.get_arm_joint_positions(as_array=True)
        # Maybe also wrap angles?
        # Ordering: sh0, sh1, el0, el1, wr0, wr1
        joint_rotation_delta = np.array([0.0, 0.0, 0.0, 0.0, wrist_yaw, wrist_roll])
        new_arm_joint_states = np.add(arm_joint_positions, joint_rotation_delta)
        self.set_arm_joint_positions(new_arm_joint_states)
        time.sleep(0.5)

    def move_gripper_to_point(self, point, rotation):
        """
        Moves EE to a point relative to body frame
        :param point: XYZ location
        :param rotation: Euler roll-pitch-yaw or WXYZ quaternion
        :return: cmd_id
        """
        if len(rotation) == 3:  # roll pitch yaw Euler angles
            roll, pitch, yaw = rotation
            quat = geometry.EulerZXY(yaw=yaw, roll=roll, pitch=pitch).to_quaternion()
        elif len(rotation) == 4:  # w, x, y, z quaternion
            w, x, y, z = rotation
            quat = math_helpers.Quat(w=w, x=x, y=y, z=z)
        else:
            raise RuntimeError(
                "rotation needs to have length 3 (euler) or 4 (quaternion),"
                f"got {len(rotation)}"
            )

        hand_pose = math_helpers.SE3Pose(*point, quat)
        hand_trajectory = trajectory_pb2.SE3Trajectory(
            points=[trajectory_pb2.SE3TrajectoryPoint(pose=hand_pose.to_proto())]
        )
        arm_cartesian_command = arm_command_pb2.ArmCartesianCommand.Request(
            pose_trajectory_in_task=hand_trajectory,
            root_frame_name=GRAV_ALIGNED_BODY_FRAME_NAME,
        )

        # Pack everything up in protos.
        arm_command = arm_command_pb2.ArmCommand.Request(
            arm_cartesian_command=arm_cartesian_command
        )
        synchronized_command = synchronized_command_pb2.SynchronizedCommand.Request(
            arm_command=arm_command
        )
        command = robot_command_pb2.RobotCommand(
            synchronized_command=synchronized_command
        )
        cmd_id = self.command_client.robot_command(command)

        return cmd_id

    def block_until_arm_arrives(self, cmd_id, timeout_sec=5):
        block_until_arm_arrives(self.command_client, cmd_id, timeout_sec=timeout_sec)

    def get_image_responses(self, sources, quality=None):
        """Retrieve images from Spot's cameras

        :param sources: list containing camera uuids
        :param quality: either an int or a list specifying what quality each source
            should return its image with
        :return: list containing bosdyn image response objects
        """
        if quality is not None:
            if isinstance(quality, int):
                quality = [quality] * len(sources)
            else:
                assert len(quality) == len(sources)
            sources = [build_image_request(src, q) for src, q in zip(sources, quality)]
            image_responses = self.image_client.get_image(sources)
        else:
            image_responses = self.image_client.get_image_from_sources(sources)

        return image_responses

    def grasp_point_in_image(
        self,
        image_response,
        pixel_xy=None,
        timeout=10,
        data_edge_timeout=2,
        top_down_grasp=False,
        horizontal_grasp=False,
    ):
        # If pixel location not provided, select the center pixel
        if pixel_xy is None:
            height = image_response.shot.image.rows
            width = image_response.shot.image.cols
            pixel_xy = [width // 2, height // 2]

        pick_vec = geometry_pb2.Vec2(x=pixel_xy[0], y=pixel_xy[1])
        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec,
            transforms_snapshot_for_camera=image_response.shot.transforms_snapshot,
            frame_name_image_sensor=image_response.shot.frame_name_image_sensor,
            camera_model=image_response.source.pinhole,
            walk_gaze_mode=3,
        )
        if top_down_grasp or horizontal_grasp:
            if top_down_grasp:
                # Add a constraint that requests that the x-axis of the gripper is
                # pointing in the negative-z direction in the vision frame.

                # The axis on the gripper is the x-axis.
                axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)

                # The axis in the vision frame is the negative z-axis
                axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=-1)

            else:
                # Add a constraint that requests that the y-axis of the gripper is
                # pointing in the positive-z direction in the vision frame. That means
                # that the gripper is constrained to be rolled 90 degrees and pointed
                # at the horizon.

                # The axis on the gripper is the y-axis.
                axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=0, y=1, z=0)

                # The axis in the vision frame is the positive z-axis
                axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=1)

            grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME
            # Add the vector constraint to our proto.
            constraint = grasp.grasp_params.allowable_orientation.add()
            constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(
                axis_on_gripper_ewrt_gripper
            )
            constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(
                axis_to_align_with_ewrt_vo
            )

            # Take anything within about 10 degrees for top-down or horizontal grasps.
            constraint.vector_alignment_with_tolerance.threshold_radians = 1.0 * 2

        # Ask the robot to pick up the object
        grasp_request = manipulation_api_pb2.ManipulationApiRequest(
            pick_object_in_image=grasp
        )
        # Send the request
        cmd_response = self.manipulation_api_client.manipulation_api_command(
            manipulation_api_request=grasp_request
        )

        # Get feedback from the robot (WILL BLOCK TILL COMPLETION)
        start_time = time.time()
        success = False
        while time.time() < start_time + timeout:
            feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=cmd_response.manipulation_cmd_id
            )

            # Send the request
            response = self.manipulation_api_client.manipulation_api_feedback_command(
                manipulation_api_feedback_request=feedback_request
            )

            print(
                "Current grasp_point_in_image state: ",
                manipulation_api_pb2.ManipulationFeedbackState.Name(
                    response.current_state
                ),
            )

            if (
                response.current_state
                == manipulation_api_pb2.MANIP_STATE_GRASP_PLANNING_WAITING_DATA_AT_EDGE
            ) and time.time() > start_time + data_edge_timeout:
                break
            elif (
                response.current_state
                == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED
            ):
                success = True
                break
            elif response.current_state in [
                manipulation_api_pb2.MANIP_STATE_GRASP_FAILED,
                manipulation_api_pb2.MANIP_STATE_GRASP_PLANNING_NO_SOLUTION,
            ]:
                break

            time.sleep(0.25)
        return success

    def grasp_hand_depth(self, *args, **kwargs):
        image_responses = self.get_image_responses(
            # [SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME]
            [SpotCamIds.HAND_COLOR]
        )
        hand_image_response = image_responses[0]  # only expecting one image
        return self.grasp_point_in_image(hand_image_response, *args, **kwargs)

    def set_base_velocity(
        self,
        x_vel,
        y_vel,
        ang_vel,
        vel_time,
        disable_obstacle_avoidance=False,
        return_cmd=False,
    ):
        body_tform_goal = math_helpers.SE2Velocity(x=x_vel, y=y_vel, angular=ang_vel)
        params = spot_command_pb2.MobilityParams(
            obstacle_params=spot_command_pb2.ObstacleParams(
                disable_vision_body_obstacle_avoidance=disable_obstacle_avoidance,
                disable_vision_foot_obstacle_avoidance=False,
                disable_vision_foot_constraint_avoidance=False,
                obstacle_avoidance_padding=0.05,  # in meters
            )
        )
        command = RobotCommandBuilder.synchro_velocity_command(
            v_x=body_tform_goal.linear_velocity_x,
            v_y=body_tform_goal.linear_velocity_y,
            v_rot=body_tform_goal.angular_velocity,
            params=params,
        )

        if return_cmd:
            return command

        cmd_id = self.command_client.robot_command(
            command, end_time_secs=time.time() + vel_time
        )

        return cmd_id

    def set_base_position(
        self,
        x_pos,
        y_pos,
        yaw,
        end_time,
        relative=False,
        max_fwd_vel=2,
        max_hor_vel=2,
        max_ang_vel=np.pi / 2,
        disable_obstacle_avoidance=False,
        blocking=False,
    ):
        vel_limit = SE2VelocityLimit(
            max_vel=SE2Velocity(
                linear=Vec2(x=max_fwd_vel, y=max_hor_vel), angular=max_ang_vel
            ),
            min_vel=SE2Velocity(
                linear=Vec2(x=-max_fwd_vel, y=-max_hor_vel), angular=-max_ang_vel
            ),
        )
        params = spot_command_pb2.MobilityParams(
            vel_limit=vel_limit,
            obstacle_params=spot_command_pb2.ObstacleParams(
                disable_vision_body_obstacle_avoidance=disable_obstacle_avoidance,
                disable_vision_foot_obstacle_avoidance=False,
                disable_vision_foot_constraint_avoidance=False,
                obstacle_avoidance_padding=0.05,  # in meters
            ),
        )
        curr_x, curr_y, curr_yaw = self.get_xy_yaw(use_boot_origin=True)
        coors = np.array([x_pos, y_pos, 1.0])
        if relative:
            local_T_global = self._get_local_T_global(curr_x, curr_y, curr_yaw)
            x, y, w = local_T_global.dot(coors)
            global_x_pos, global_y_pos = x / w, y / w
            global_yaw = wrap_heading(curr_yaw + yaw)
        else:
            global_x_pos, global_y_pos, global_yaw = self.xy_yaw_home_to_global(
                x_pos, y_pos, yaw
            )
        robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=global_x_pos,
            goal_y=global_y_pos,
            goal_heading=global_yaw,
            frame_name=VISION_FRAME_NAME,
            params=params,
        )
        cmd_id = self.command_client.robot_command(
            robot_cmd, end_time_secs=time.time() + end_time
        )

        if blocking:
            cmd_status = None
            while cmd_status != 1:
                time.sleep(0.1)
                feedback_resp = self.get_cmd_feedback(cmd_id)
                cmd_status = (
                    feedback_resp.feedback.synchronized_feedback
                ).mobility_command_feedback.se2_trajectory_feedback.status
            return None

        return cmd_id

    def get_robot_state(self):
        return self.robot_state_client.get_robot_state()

    def get_battery_charge(self):
        state = self.get_robot_state()
        return state.power_state.locomotion_charge_percentage.value

    def roll_over(self, roll_over_left=True):
        if roll_over_left:
            dir_hint = basic_command_pb2.BatteryChangePoseCommand.Request.HINT_LEFT
        else:
            dir_hint = basic_command_pb2.BatteryChangePoseCommand.Request.HINT_RIGHT
        cmd = RobotCommandBuilder.battery_change_pose_command(dir_hint=dir_hint)
        self.command_client.robot_command(cmd)

    def sit(self):
        cmd = RobotCommandBuilder.synchro_sit_command()
        self.command_client.robot_command(cmd)

    def get_arm_proprioception(self, robot_state=None):
        """Return state of each of the 6 joints of the arm"""
        if robot_state is None:
            robot_state = self.robot_state_client.get_robot_state()
        arm_joint_states = OrderedDict(
            {
                i.name[len("arm0.") :]: i
                for i in robot_state.kinematic_state.joint_states
                if i.name in ARM_6DOF_NAMES
            }
        )

        return arm_joint_states

    def get_proprioception(self, robot_state=None):
        """Return state of each of the 6 joints of the arm"""
        if robot_state is None:
            robot_state = self.robot_state_client.get_robot_state()
        joint_states = OrderedDict(
            {i.name: i for i in robot_state.kinematic_state.joint_states}
        )

        return joint_states

    def get_arm_joint_positions(self, as_array=True):
        """
        Gives in joint positions of the arm in radians in the following order
        Ordering: sh0, sh1, el0, el1, wr0, wr1
        :param as_array: bool, True for output as an np.array, False for list
        :return: 6 element data structure (np.array or list) of joint positions as radians
        """
        arm_joint_states = self.get_arm_proprioception()
        arm_joint_positions = np.fromiter(
            (arm_joint_states[joint].position.value for joint in arm_joint_states),
            float,
        )

        if as_array:
            return arm_joint_positions
        return arm_joint_positions.tolist()

    def set_arm_joint_positions(
        self, positions, travel_time=1.0, max_vel=2.5, max_acc=15, return_cmd=False
    ):
        """
        Takes in 6 joint targets and moves each arm joint to the corresponding target.
        Ordering: sh0, sh1, el0, el1, wr0, wr1
        :param positions: np.array or list of radians
        :param travel_time: how long execution should take
        :param max_vel: max allowable velocity
        :param max_acc: max allowable acceleration
        :return: cmd_id
        """
        sh0, sh1, el0, el1, wr0, wr1 = positions
        traj_point = RobotCommandBuilder.create_arm_joint_trajectory_point(
            sh0, sh1, el0, el1, wr0, wr1, travel_time
        )
        arm_joint_traj = arm_command_pb2.ArmJointTrajectory(
            points=[traj_point],
            maximum_velocity=wrappers_pb2.DoubleValue(value=max_vel),
            maximum_acceleration=wrappers_pb2.DoubleValue(value=max_acc),
        )
        command = make_robot_command(arm_joint_traj)

        if return_cmd:
            return command

        cmd_id = self.command_client.robot_command(command)

        return cmd_id

    def set_base_vel_and_arm_pos(
        self,
        x_vel,
        y_vel,
        ang_vel,
        arm_positions,
        travel_time,
        disable_obstacle_avoidance=False,
    ):
        base_cmd = self.set_base_velocity(
            x_vel,
            y_vel,
            ang_vel,
            vel_time=travel_time,
            disable_obstacle_avoidance=disable_obstacle_avoidance,
            return_cmd=True,
        )
        arm_cmd = self.set_arm_joint_positions(
            arm_positions, travel_time=travel_time, return_cmd=True
        )
        synchro_command = RobotCommandBuilder.build_synchro_command(base_cmd, arm_cmd)
        cmd_id = self.command_client.robot_command(
            synchro_command, end_time_secs=time.time() + travel_time
        )
        return cmd_id

    def get_xy_yaw(self, use_boot_origin=False, robot_state=None):
        """
        Returns the relative x and y distance from start, as well as relative heading
        """
        if robot_state is None:
            robot_state = self.robot_state_client.get_robot_state()
        robot_state_kin = robot_state.kinematic_state
        self.body = get_vision_tform_body(robot_state_kin.transforms_snapshot)
        robot_tform = self.body
        yaw = math_helpers.quat_to_eulerZYX(robot_tform.rotation)[0]
        if self.global_T_home is None or use_boot_origin:
            return robot_tform.x, robot_tform.y, yaw
        return self.xy_yaw_global_to_home(robot_tform.x, robot_tform.y, yaw)

    def xy_yaw_global_to_home(self, x, y, yaw):
        x, y, w = self.global_T_home.dot(np.array([x, y, 1.0]))
        x, y = x / w, y / w

        return x, y, wrap_heading(yaw - self.robot_recenter_yaw)

    def xy_yaw_home_to_global(self, x, y, yaw):
        local_T_global = np.linalg.inv(self.global_T_home)
        x, y, w = local_T_global.dot(np.array([x, y, 1.0]))
        x, y = x / w, y / w

        return x, y, wrap_heading(self.robot_recenter_yaw - yaw)

    def _get_local_T_global(self, x=None, y=None, yaw=None):
        if x is None:
            x, y, yaw = self.get_xy_yaw(use_boot_origin=True)
        # Create offset transformation matrix
        local_T_global = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), x],
                [np.sin(yaw), np.cos(yaw), y],
                [0.0, 0.0, 1.0],
            ]
        )
        return local_T_global

    def home_robot(self):
        x, y, yaw = self.get_xy_yaw(use_boot_origin=True)
        local_T_global = self._get_local_T_global()
        self.global_T_home = np.linalg.inv(local_T_global)
        self.robot_recenter_yaw = yaw

        as_string = list(self.global_T_home.flatten()) + [yaw]
        as_string = f"{as_string}"[1:-1]  # [1:-1] removes brackets
        with open(HOME_TXT, "w") as f:
            f.write(as_string)
        self.loginfo(f"Wrote:\n{as_string}\nto: {HOME_TXT}")

    def get_base_transform_to(self, child_frame):
        kin_state = self.robot_state_client.get_robot_state().kinematic_state
        kin_state = kin_state.transforms_snapshot.child_to_parent_edge_map.get(
            child_frame
        ).parent_tform_child
        return kin_state.position, kin_state.rotation

    def dock(self, dock_id, home_robot=False):
        blocking_dock_robot(self.robot, dock_id)
        if home_robot:
            self.home_robot()

    def undock(self):
        blocking_undock(self.robot)


class SpotLease:
    """
    A class that supports execution with Python's "with" statement for safe return of
    the lease and settle-then-estop upon exit. Grants control of the Spot's motors.
    """

    def __init__(self, spot, hijack=False):
        self.lease_client = spot.robot.ensure_client(
            bosdyn.client.lease.LeaseClient.default_service_name
        )
        if hijack:
            self.lease = self.lease_client.take()
        else:
            self.lease = self.lease_client.acquire()
        self.lease_keep_alive = bosdyn.client.lease.LeaseKeepAlive(self.lease_client)
        self.spot = spot

    def __enter__(self):
        return self.lease

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Exit the LeaseKeepAlive object
        self.lease_keep_alive.__exit__(exc_type, exc_val, exc_tb)
        # Return the lease
        self.lease_client.return_lease(self.lease)
        self.spot.loginfo("Returned the lease.")
        # Clear lease from Spot object
        self.spot.spot_lease = None

    def create_sublease(self):
        return self.lease.create_sublease()


def make_robot_command(arm_joint_traj):
    """Helper function to create a RobotCommand from an ArmJointTrajectory.
    The returned command will be a SynchronizedCommand with an ArmJointMoveCommand
    filled out to follow the passed in trajectory."""

    joint_move_command = arm_command_pb2.ArmJointMoveCommand.Request(
        trajectory=arm_joint_traj
    )
    arm_command = arm_command_pb2.ArmCommand.Request(
        arm_joint_move_command=joint_move_command
    )
    sync_arm = synchronized_command_pb2.SynchronizedCommand.Request(
        arm_command=arm_command
    )
    arm_sync_robot_cmd = robot_command_pb2.RobotCommand(synchronized_command=sync_arm)
    return RobotCommandBuilder.build_synchro_command(arm_sync_robot_cmd)


def image_response_to_cv2(image_response, reorient=True):
    if image_response.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
        dtype = np.uint16
    else:
        dtype = np.uint8
    # img = np.fromstring(image_response.shot.image.data, dtype=dtype)
    img = np.frombuffer(image_response.shot.image.data, dtype=dtype)
    if image_response.shot.image.format == image_pb2.Image.FORMAT_RAW:
        img = img.reshape(
            image_response.shot.image.rows, image_response.shot.image.cols
        )
    else:
        img = cv2.imdecode(img, -1)

    if reorient and image_response.source.name in SHOULD_ROTATE:
        img = np.rot90(img, k=3)

    return img


def scale_depth_img(img, min_depth=0.0, max_depth=10.0, as_img=False):
    min_depth, max_depth = min_depth * 1000, max_depth * 1000
    img_copy = np.clip(img.astype(np.float32), a_min=min_depth, a_max=max_depth)
    img_copy = (img_copy - min_depth) / (max_depth - min_depth)
    if as_img:
        img_copy = cv2.cvtColor((255.0 * img_copy).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    return img_copy


def draw_crosshair(img):
    height, width = img.shape[:2]
    cx, cy = width // 2, height // 2
    img = cv2.circle(
        img,
        center=(cx, cy),
        radius=5,
        color=(0, 0, 255),
        thickness=1,
    )

    return img


def wrap_heading(heading):
    """Ensures input heading is between -180 an 180; can be float or np.ndarray"""
    return (heading + np.pi) % (2 * np.pi) - np.pi
