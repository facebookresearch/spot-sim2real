# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


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
from typing import Any, Dict, List, Tuple

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import cv2
import magnum as mn
import numpy as np
import quaternion
import rospy

try:
    import sophuspy as sp
except Exception as e:
    print(f"Cannot import sophuspy due to {e}. Import sophus instead")
    import sophus as sp

from bosdyn import geometry
from bosdyn.api import (
    arm_command_pb2,
    basic_command_pb2,
    geometry_pb2,
    image_pb2,
    manipulation_api_pb2,
    mobility_command_pb2,
    robot_command_pb2,
    synchronized_command_pb2,
    trajectory_pb2,
)
from bosdyn.api.geometry_pb2 import SE2Velocity, SE2VelocityLimit, Vec2, Vec3
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.api.spot.inverse_kinematics_pb2 import (
    InverseKinematicsRequest,
    InverseKinematicsResponse,
)
from bosdyn.client import math_helpers
from bosdyn.client.docking import blocking_dock_robot, blocking_undock
from bosdyn.client.frame_helpers import (
    GRAV_ALIGNED_BODY_FRAME_NAME,
    GROUND_PLANE_FRAME_NAME,
    ODOM_FRAME_NAME,
    VISION_FRAME_NAME,
    get_a_tform_b,
    get_vision_tform_body,
)
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.inverse_kinematics import InverseKinematicsClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.math_helpers import Quat, SE3Pose
from bosdyn.client.robot_command import (
    RobotCommandBuilder,
    RobotCommandClient,
    block_for_trajectory_cmd,
    block_until_arm_arrives,
    blocking_selfright,
    blocking_stand,
)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.util import seconds_to_duration
from geometry_msgs.msg import Pose, TransformStamped
from google.protobuf import wrappers_pb2  # type: ignore
from perception_and_utils.utils.conversions import (
    bd_SE3Pose_to_ros_Pose,
    bd_SE3Pose_to_ros_TransformStamped,
    bd_SE3Pose_to_sophus_SE3,
)
from spot_rl.utils.gripper_t_intel_path import GRIPPER_T_INTEL_PATH
from spot_rl.utils.pixel_to_3d_conversion_utils import project_3d_to_pixel_uv
from spot_rl.utils.utils import ros_frames as rf
from spot_wrapper.utils import (
    get_angle_between_forward_and_target,
    get_position_and_vel_values,
)

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
    rf.SPOT_ARM_SHOULDER_0,
    rf.SPOT_ARM_SHOULDER_1,
    rf.SPOT_ARM_ELBOW_0,
    rf.SPOT_ARM_ELBOW_1,
    rf.SPOT_ARM_WRIST_0,
    rf.SPOT_ARM_WRIST_1,
]

HOME_TXT = osp.join(osp.dirname(osp.abspath(__file__)), "home.txt")

# Get Spot DOCK ID
DOCK_ID = int(os.environ.get("SPOT_DOCK_ID", 520))

# For constructing constrained manipulation task
POSITION_MODE = (
    basic_command_pb2.ConstrainedManipulationCommand.Request.CONTROL_MODE_POSITION
)
VELOCITY_MODE = (
    basic_command_pb2.ConstrainedManipulationCommand.Request.CONTROL_MODE_VELOCITY
)


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
    INTEL_REALSENSE_COLOR = "intelrealsensergb"  # In habitat-lab, the intelrealsense camera is called jaw camera
    INTEL_REALSENSE_DEPTH = "intelrealsensedepth"


# Maps SpotCamId (name of camera in spot) to
# frame of respective camera as defined in spot
SpotCamIdToFrameNameMap = {
    SpotCamIds.BACK_DEPTH: "back",
    SpotCamIds.BACK_DEPTH_IN_VISUAL_FRAME: "back_fisheye",
    SpotCamIds.BACK_FISHEYE: "back_fisheye",
    SpotCamIds.FRONTLEFT_DEPTH: "frontleft",
    SpotCamIds.FRONTLEFT_DEPTH_IN_VISUAL_FRAME: "frontleft_fisheye",
    SpotCamIds.FRONTLEFT_FISHEYE: "frontleft_fisheye",
    SpotCamIds.FRONTRIGHT_DEPTH: "frontright",
    SpotCamIds.FRONTRIGHT_DEPTH_IN_VISUAL_FRAME: "frontright_fisheye",
    SpotCamIds.FRONTRIGHT_FISHEYE: "frontright_fisheye",
    SpotCamIds.HAND_COLOR: "hand_color_image_sensor",
    SpotCamIds.HAND_COLOR_IN_HAND_DEPTH_FRAME: "hand_depth_sensor",
    SpotCamIds.HAND_DEPTH: "hand_depth_sensor",
    SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME: "hand_color_image_sensor",
    SpotCamIds.HAND: "hand_depth_sensor",
    SpotCamIds.LEFT_DEPTH: "left",
    SpotCamIds.LEFT_DEPTH_IN_VISUAL_FRAME: "left_fisheye",
    SpotCamIds.LEFT_FISHEYE: "left_fisheye",
    SpotCamIds.RIGHT_DEPTH: "right",
    SpotCamIds.RIGHT_DEPTH_IN_VISUAL_FRAME: "right_fisheye",
    SpotCamIds.RIGHT_FISHEYE: "right_fisheye",
    SpotCamIds.INTEL_REALSENSE_COLOR: "hand_color_image_sensor",
    SpotCamIds.INTEL_REALSENSE_DEPTH: "hand_color_image_sensor",
}  # type: Dict[SpotCamIds, str]


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

        # Make our intel image client
        try:
            self.intelrealsense_image_client = robot.ensure_client(
                "intel-realsense-image-service"
            )
            self.gripper_T_intel: sp.SE3 = sp.SE3(np.load(GRIPPER_T_INTEL_PATH))
            print(f"Loaded gripper_T_intel (sp.SE3) as {self.gripper_T_intel.matrix()}")

        except Exception:
            print("There is no intel-realsense-image_service. Using gripper cameras")
            self.intelrealsense_image_client = None
            self.gripper_T_intel = None
            print(f"Loaded gripper_T_intel (sp.SE3) as {self.gripper_T_intel}")

        self.manipulation_api_client = robot.ensure_client(
            ManipulationApiClient.default_service_name
        )
        self.robot_state_client = robot.ensure_client(
            RobotStateClient.default_service_name
        )
        self.ik_client = robot.ensure_client(
            InverseKinematicsClient.default_service_name
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

    @property
    def is_gripper_blocked(self):
        """A function to set the ros parameter: is_gripper_blocked for choosing between
        Spot's gripper camera or intelrealsense camera (jaw camera). 0 for using Spot's gripper camera,
        and 1 for using intelrealsense camera (jaw camera).
        """
        return rospy.get_param("is_gripper_blocked", default=0) == 1

    def get_lease(self, hijack=False):
        # Make sure a lease for this client isn't already active
        assert self.spot_lease is None
        self.spot_lease = SpotLease(self, hijack=hijack)
        return self.spot_lease

    def get_cmd_feedback(self, cmd_id):
        return self.command_client.robot_command_feedback(cmd_id)

    def is_estopped(self):
        return self.robot.is_estopped()

    def is_powered_on(self):
        return self.robot.is_powered_on()

    def power_on(self, timeout_sec=20):
        if not self.is_powered_on():
            self.loginfo("Powering robot on...")
            self.robot.power_on(timeout_sec=timeout_sec)

        assert self.is_powered_on(), "Robot power on failed."
        self.loginfo("Robot powered on.")

    def power_off(self, cut_immediately=False, timeout_sec=20):
        if self.is_powered_on():
            self.loginfo("Powering robot off...")
            self.robot.power_off(
                cut_immediately=cut_immediately, timeout_sec=timeout_sec
            )

        assert not self.is_powered_on(), "Robot power off failed."
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

    def move_gripper_to_points(
        self,
        point_list: List[np.ndarray],
        rotations: List[Tuple],
        allow_body_follow: bool = False,
        seconds_to_goal: float = 10.0,
        timeout_sec: float = 20,
    ):
        """
        Moves EE to a point relative to body frame
        However it can accept list of rotations such that you can add interpolated rotations in between such that move_gripper_ never fails
        :param point: XYZ location
        :param rotations: list[Euler roll-pitch-yaw or WXYZ quaternion]
        :return: cmd_id
        """
        points = []

        assert len(point_list) == len(
            rotations
        ), f"len(point) {len(point_list)} not matching len(rotations) {len(rotations)}"

        for i, (rotation, point) in enumerate(zip(rotations, point_list)):
            if len(rotation) == 3:  # roll pitch yaw Euler angles
                roll, pitch, yaw = rotation  # xyz
                quat = geometry.EulerZXY(
                    yaw=yaw, roll=roll, pitch=pitch
                ).to_quaternion()
            elif len(rotation) == 4:  # w, x, y, z quaternion
                w, x, y, z = rotation
                quat = math_helpers.Quat(w=w, x=x, y=y, z=z)
            else:
                raise RuntimeError(
                    "rotation needs to have length 3 (euler) or 4 (quaternion),"
                    f"got {len(rotation)}"
                )
            point = point.tolist() if isinstance(point, type(np.array([]))) else point
            hand_pose = math_helpers.SE3Pose(*point, quat)
            points.append(
                trajectory_pb2.SE3TrajectoryPoint(
                    pose=hand_pose.to_proto(),
                    time_since_reference=seconds_to_duration(seconds_to_goal * i),
                )
            )

        hand_trajectory = trajectory_pb2.SE3Trajectory(points=points)
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

        if allow_body_follow and len(point_list) == 1:
            follow_arm_command = RobotCommandBuilder.follow_arm_command()
            # Combine the arm and mobility commands into one synchronized command.
            command = RobotCommandBuilder.build_synchro_command(
                follow_arm_command, command
            )
        cmd_id = self.command_client.robot_command(command)
        success_status = self.block_until_arm_arrives(cmd_id, timeout_sec=timeout_sec)

        # Set the robot base velocity to reset the base motion after calling body movement.
        # Without this, calling the move gripper function casues the base to move.
        self.set_base_velocity(x_vel=0, y_vel=0, ang_vel=0, vel_time=0.8)
        return success_status

    def move_gripper_to_point(
        self, point, rotation, seconds_to_goal=3.0, timeout_sec=10
    ):
        """
        Moves EE to a point relative to body frame
        :param point: XYZ location
        :param rotation: Euler roll-pitch-yaw or WXYZ quaternion
        :return: cmd_id
        """
        if len(rotation) == 3:  # roll pitch yaw Euler angles
            roll, pitch, yaw = rotation  # x, y, z
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
            points=[
                trajectory_pb2.SE3TrajectoryPoint(
                    pose=hand_pose.to_proto(),
                    time_since_reference=seconds_to_duration(seconds_to_goal),
                )
            ]
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

        success_status = self.block_until_arm_arrives(cmd_id, timeout_sec=timeout_sec)
        return success_status

    def block_until_arm_arrives(self, cmd_id, timeout_sec=5):
        return block_until_arm_arrives(
            self.command_client, cmd_id, timeout_sec=timeout_sec
        )

    def get_image_responses(
        self, sources, quality=100, pixel_format=None, await_the_resp=True
    ):
        """Retrieve images from Spot's cameras

        :param sources: list containing camera uuids
        :param quality: either an int or a list specifying what quality each source
            should return its image with
        :param pixel_format: either an int or a list specifying what pixel format each source
            should return its image with
        :param await_the_resp: either get image response result() or not
        :return: list containing bosdyn image response objects (google.protobuf.pyext._message.RepeatedCompositeContainer)
        """

        # Choose between intelrealsense camera or gripper camera
        image_client = (
            self.image_client
            if "intel" not in sources[0]
            else self.intelrealsense_image_client
        )
        if quality is not None:
            if isinstance(quality, int):
                quality = [quality] * len(sources)
            else:
                assert len(quality) == len(sources)
            img_requests = [
                build_image_request(src, q) for src, q in zip(sources, quality)
            ]
            image_responses = image_client.get_image_async(img_requests)
        elif pixel_format is not None:
            if isinstance(pixel_format, int):
                pixel_format = [pixel_format] * len(sources)
            else:
                assert len(pixel_format) == len(sources)
            img_requests = [
                build_image_request(src, pixel_format=pf)
                for src, pf in zip(sources, pixel_format)
            ]
            image_responses = image_client.get_image_async(img_requests)
        else:
            image_responses = image_client.get_image_from_sources_async(sources)

        return image_responses.result() if await_the_resp else image_responses

    def query_IK_reachability_of_gripper(self, se3querypose: SE3Pose) -> bool:
        """Check the reachability of a given pose of the gripper in the body frame"""

        # This is how you make the SE3 pose given x,y,z and rotation:
        # se3querypose = SE3Pose(*point_in_body, Quat(*gripper_pose_quat))
        task_T_desired_tool: SE3Pose = se3querypose

        # get this pose from the BD document
        wr1_T_tool: SE3Pose = SE3Pose(0, 0, 0, Quat())

        odom_T_ground_body: SE3Pose = get_a_tform_b(
            self.get_robot_state().kinematic_state.transforms_snapshot,
            ODOM_FRAME_NAME,
            "body",
        )

        # Now, construct a task frame as body frame
        odom_T_task: SE3Pose = odom_T_ground_body

        ik_request = InverseKinematicsRequest(
            root_frame_name=ODOM_FRAME_NAME,
            scene_tform_task=odom_T_task.to_proto(),
            wrist_mounted_tool=InverseKinematicsRequest.WristMountedTool(
                wrist_tform_tool=wr1_T_tool.to_proto()
            ),
            tool_pose_task=InverseKinematicsRequest.ToolPoseTask(
                task_tform_desired_tool=task_T_desired_tool.to_proto()
            ),
        )
        ik_response = self.ik_client.inverse_kinematics(ik_request)
        if ik_response.status == InverseKinematicsResponse.STATUS_OK:
            return True
        return False

    def make_arm_pose_command(
        self,
        point_in_body: np.ndarray,
        gripper_pose_quat: List[float],
        seconds: int = 5,
    ):
        """Make the arm pose command for the given point in the body frame and the given gripper pose, and
        it does not execute. This is a helper function for move_arm_to_point_with_body_follow"""
        hand_pos_rt_body = geometry_pb2.Vec3(
            x=point_in_body[0], y=point_in_body[1], z=point_in_body[-1]
        )

        # Rotation as a quaternion
        if isinstance(gripper_pose_quat, type(geometry.EulerZXY().to_quaternion())):
            qw, qx, qy, qz = (
                gripper_pose_quat.w,
                gripper_pose_quat.x,
                gripper_pose_quat.y,
                gripper_pose_quat.z,
            )
        else:
            qw, qx, qy, qz = gripper_pose_quat

        body_Q_hand = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)

        # Build the SE(3) pose of the desired hand position in the moving body frame.
        body_T_hand = geometry_pb2.SE3Pose(
            position=hand_pos_rt_body, rotation=body_Q_hand
        )

        # Transform the desired from the moving body frame to the odom frame.
        robot_state = self.get_robot_state()
        odom_T_body = get_a_tform_b(
            robot_state.kinematic_state.transforms_snapshot,
            ODOM_FRAME_NAME,
            GRAV_ALIGNED_BODY_FRAME_NAME,
        )
        odom_T_hand = odom_T_body * math_helpers.SE3Pose.from_proto(body_T_hand)

        # duration in seconds
        # Create the arm command.
        arm_command = RobotCommandBuilder.arm_pose_command(
            odom_T_hand.x,
            odom_T_hand.y,
            odom_T_hand.z,
            odom_T_hand.rot.w,
            odom_T_hand.rot.x,
            odom_T_hand.rot.y,
            odom_T_hand.rot.z,
            ODOM_FRAME_NAME,
            seconds,
        )
        return arm_command

    def move_arm_to_point_with_body_follow(
        self,
        points_in_body: List[np.ndarray],
        gripper_pose_quats: List[List[float]],
        seconds: int = 5,
        allow_body_follow: bool = True,
        body_offset_from_hand: List[float] = [0.55, 0, 0.25],
    ) -> bool:
        """Move the arm to the given point in the body frame and the given gripper poses, and allow
        the body to follow the arm."""

        arm_command_list = []
        for point_in_body, gripper_pose_quat in zip(points_in_body, gripper_pose_quats):
            arm_command_list.append(
                self.make_arm_pose_command(point_in_body, gripper_pose_quat)
            )

        if allow_body_follow and len(arm_command_list) == 1:
            # Tell the robot's body to follow the arm
            mobility_command = mobility_command_pb2.MobilityCommand.Request(
                follow_arm_request=basic_command_pb2.FollowArmCommand.Request(
                    body_offset_from_hand=Vec3(
                        x=body_offset_from_hand[0],
                        y=body_offset_from_hand[1],
                        z=body_offset_from_hand[2],
                    )
                )
            )
            synchronized_command = synchronized_command_pb2.SynchronizedCommand.Request(
                mobility_command=mobility_command
            )
            follow_arm_command = robot_command_pb2.RobotCommand(
                synchronized_command=synchronized_command
            )
            # Combine the arm and mobility commands into one synchronized command.
            command = RobotCommandBuilder.build_synchro_command(
                follow_arm_command, *arm_command_list
            )
        else:
            command = RobotCommandBuilder.build_synchro_command(*arm_command_list)

        # Send the request
        move_command_id = self.command_client.robot_command(command)
        self.robot.logger.info("Moving arm to position.")
        msg = self.block_until_arm_arrives(move_command_id, seconds + 1)

        # Set the robot base velocity to reset the base motion after calling body movement.
        # Without this, calling the move gripper function casues the base to move.
        self.set_base_velocity(x_vel=0, y_vel=0, ang_vel=0, vel_time=0.8)
        return msg

    def grasp_point_in_image_with_IK(
        self,
        point_in_gripper: np.ndarray,
        body_T_cam: mn.Matrix4,
        gripper_pose_quat: List[float] = None,
        solution_angles: np.ndarray = np.zeros((3,)),
        timeout=10,
        claw_gripper_control_parameters: List[Tuple] = [],
        visualize: Tuple[Any, np.ndarray] = None,
    ):
        """This is an alternative (grasp_point_in_image) to grasp the object in the image, which uses IK to move the arm to the point in the image."""

        point_in_body = np.array(
            body_T_cam.transform_point(mn.Vector3(*point_in_gripper))
        )
        print(f"Point in body {point_in_body}")

        bx, by, byaw = self.get_xy_yaw(False)
        current_gripper_pose = self.get_ee_quaternion_in_body_frame(
            frame_name=GRAV_ALIGNED_BODY_FRAME_NAME
        ).view((np.double, 4))
        up_thresh = 0.2
        point_in_body_uppeest = point_in_body.copy()
        point_in_body_uppeest[-1] += up_thresh
        is_point_reachable_without_mobility = self.query_IK_reachability_of_gripper(
            SE3Pose(*point_in_body, Quat(*gripper_pose_quat))
        )
        if not is_point_reachable_without_mobility:
            point_in_body_uppeest[0] += 0.05

        status = self.move_arm_to_point_with_body_follow(
            [point_in_body_uppeest],
            [gripper_pose_quat],
            allow_body_follow=not is_point_reachable_without_mobility,
        )

        pos = self.get_ee_pos_in_body_frame(GRAV_ALIGNED_BODY_FRAME_NAME)[0]
        pos[-1] -= up_thresh - 0.1
        current_gripper_pose = self.get_ee_quaternion_in_body_frame(
            GRAV_ALIGNED_BODY_FRAME_NAME
        ).view((np.double, 4))
        status = self.move_arm_to_point_with_body_follow(
            [pos], [current_gripper_pose], allow_body_follow=False
        )

        if visualize is not None:
            intrinsics, predicted_pixel, rgb_image = visualize
            pixel_uv = project_3d_to_pixel_uv(
                np.array(
                    body_T_cam.inverted().transform_point(mn.Vector3(*point_in_body))
                ).reshape(1, 3),
                intrinsics,
            )[0].tolist()
            pixel_uv = list(map(int, pixel_uv))
            predicted_pixel = list(map(int, predicted_pixel))
            rgb_image = cv2.circle(rgb_image, pixel_uv, 2, (0, 255, 0), 2)
            rgb_image = cv2.circle(rgb_image, predicted_pixel, 2, (0, 0, 255), 2)
            cv2.imwrite("grasp_point.png", rgb_image)

        print(f"Grasp reached to object ? {status}")

        n: int = len(claw_gripper_control_parameters)
        claw_gripper_command = None

        # Force control the gripper to ensure that the robot does not crush the object
        for claw_index, (claw_gripper_angle, max_torque) in enumerate(
            claw_gripper_control_parameters
        ):
            claw_gripper_command = (
                RobotCommandBuilder.claw_gripper_open_fraction_command(
                    claw_gripper_angle,
                    claw_gripper_command,
                    disable_force_on_contact=claw_index != n - 1,
                    max_torque=max_torque,
                )
            )
        self.command_client.robot_command(claw_gripper_command)
        nbx, nby, nbyaw = self.get_xy_yaw()

        current_position_of_gripper_in_body = self.get_ee_pos_in_body_frame()[0]
        current_position_of_gripper_in_body[-1] += 0.1

        roll, pitch, yaw = np.deg2rad(solution_angles).tolist()
        quat = geometry.EulerZXY(yaw=yaw, roll=roll, pitch=pitch).to_quaternion()

        self.move_arm_to_point_with_body_follow(
            [current_position_of_gripper_in_body] * 3,
            [current_gripper_pose, [1.0, 0.0, 0.0, 0.0], quat],
            1,
        )
        return status

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
            constraint.vector_alignment_with_tolerance.threshold_radians = 0.17

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

    def get_global_from_local_based_on_input_T(
        self, x_pos, y_pos, yaw, curr_x, curr_y, curr_yaw
    ):
        """Transform the point given the x,y,yaw location"""
        coors = np.array([x_pos, y_pos, 1.0])
        local_T_global = self._get_local_T_global(curr_x, curr_y, curr_yaw)
        x, y, w = local_T_global.dot(coors)
        global_x_pos, global_y_pos = x / w, y / w
        global_yaw = wrap_heading(curr_yaw + yaw)
        return global_x_pos, global_y_pos, global_yaw

    def get_global_from_local(self, x_pos, y_pos, yaw):
        """Local x, y, yaw locations given the base"""
        curr_x, curr_y, curr_yaw = self.get_xy_yaw(use_boot_origin=True)
        return self.get_global_from_local_based_on_input_T(
            x_pos, y_pos, yaw, curr_x, curr_y, curr_yaw
        )

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
        if relative:
            global_x_pos, global_y_pos, global_yaw = self.get_global_from_local(
                x_pos, y_pos, yaw
            )
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

    def home_robot(self, write_to_file: bool = False):
        print(f"Updating robot pose w.r.t home. write_to_file={write_to_file}")
        x, y, yaw = self.get_xy_yaw(use_boot_origin=True)
        local_T_global = self._get_local_T_global()
        self.global_T_home = np.linalg.inv(local_T_global)
        self.robot_recenter_yaw = yaw

        if write_to_file:
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

    def dock(self, dock_id: int = DOCK_ID, home_robot: bool = False) -> None:
        """
        Dock the robot to the specified dock
        `blocking_dock_robot` will also move the robot to the dock if the dock is in view
        otherwise it will look for the dock at its current location

        Args:
            dock_id: The dock to dock to
            home_robot: Whether to reset home the robot after docking
        """
        blocking_dock_robot(self.robot, dock_id)
        if home_robot:
            self.home_robot(write_to_file=True)

    def undock(self):
        blocking_undock(self.robot)

    def power_robot(self):
        """
        Power on the robot and undock/stand up
        """
        self.power_on()

        # Undock if docked, otherwise stand up
        try:
            self.undock()
        except Exception:
            print("Undocking failed: just standing up instead...")
            self.blocking_stand()

    def shutdown(self, should_dock: bool = False) -> None:
        """
        Stops the robot and docks it if should_dock is True else sits the robot down

        Args:
            should_dock: bool indicating whether to dock the robot or not
        """
        try:
            if should_dock:
                print("Executing automatic docking")
                dock_start_time = time.time()
                while time.time() - dock_start_time < 2:
                    try:
                        self.dock(dock_id=DOCK_ID, home_robot=True)
                    except Exception:
                        print("Dock not found... trying again")
                        time.sleep(0.1)
            else:
                print("Will sit down here")
                self.sit()
        finally:
            self.power_off()

    def select_hand_image(self, is_rgb=True, img_src: List[str] = []):
        """
        Gets hand raw rgb and depth, returns List[rgbimage, unscaleddepthimage] image object is BD source image object which has kinematic snapshot
        and camera intrinsics along with pixel data
        """
        img_src = (
            img_src
            if img_src
            else [SpotCamIds.HAND_COLOR, SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME]
        )  # default img_src to gripper

        pixel_format_rgb = (
            image_pb2.Image.PIXEL_FORMAT_RGB_U8
            if is_rgb
            else image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8
        )
        pixel_format_depth = image_pb2.Image.PIXEL_FORMAT_DEPTH_U16
        img_resp = self.get_image_responses(
            img_src, pixel_format=[pixel_format_rgb, pixel_format_depth]
        )
        return img_resp

    def get_hand_image(self, is_rgb=True, force_get_gripper=False):
        """
        Gets hand raw rgb & depth, returns List[rgbimage, unscaleddepthimage] image object is BD source image object which has kinematic snapshot & camera intrinsics along with pixel data
        If is_gripper_blocked is True then returns intel realsense images
        If hand_image_sources are passed then above condition is ignored & will send image & depth for each source
        Thus if you send hand_image_sources=["gripper", "intelrealsense"] then 4 image resps should be returned.
        In addition, the flag force_get_gripper allows you to get gripper images even if is_gripper_blocked is True.
        This is useful when you want to get gripper camera transformation when the gripper is blocked.
        """
        realsense_img_srcs: List[str] = [
            SpotCamIds.INTEL_REALSENSE_COLOR,
            SpotCamIds.INTEL_REALSENSE_DEPTH,
        ]

        if self.is_gripper_blocked and not force_get_gripper:
            return self.select_hand_image(img_src=realsense_img_srcs)
        else:
            return self.select_hand_image(is_rgb=is_rgb)

    def get_camera_intrinsics(
        self,
        sources: List[SpotCamIds],
        quality=None,
        pixel_format=None,
        as_3x3_matrix: bool = False,
    ) -> List[image_pb2.ImageSource.PinholeModel.CameraIntrinsics]:
        """Retrieve caliberation properties of stated Spot's cameras

        :param sources: list containing camera uuids
        :param quality: (Optional) either an int or a list specifying what quality each source
            should return its image with
        :param quality: (Optional) pixel format of response
        :param as_3x3_matrix: (Optional) indicating the response of transformation if it should
            be 3x3 np.ndarray or image_pb2.ImageSource.PinholeModel.CameraIntrinsics

        :return: list containing all inputs cameras' intrinsics either as 3x3 np.ndarray or
            as image_pb2.ImageSource.PinholeModel.CameraIntrinsics
        """
        image_responses = self.get_image_responses(
            sources, quality=quality, pixel_format=pixel_format
        )

        camera_intrinsics_list = (
            []
        )  # type: List[Any[image_pb2.ImageSource.PinholeModel.CameraIntrinsics, np.ndarray]]
        for image_response in image_responses:
            camera_intrinsics = image_response.source.pinhole.intrinsics
            if as_3x3_matrix:
                fx = camera_intrinsics.focal_length.x
                fy = camera_intrinsics.focal_length.y
                ppx = camera_intrinsics.principal_point.x
                ppy = camera_intrinsics.principal_point.y
                camera_intrinsics_list.append(
                    np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]])
                )
            else:
                camera_intrinsics_list.append(camera_intrinsics)
        return camera_intrinsics_list

    def get_camera_intrinsics_as_3x3(self, camera_intrinsics) -> np.ndarray:
        """
        Converts camera intrinsics BD object to 3X3 camera intrinsics matrix
        Args:
           camera_intrinsics : bosdyn.api.image_pb2.CameraIntrinsics object
        """
        fx = camera_intrinsics.focal_length.x
        fy = camera_intrinsics.focal_length.y
        ppx = camera_intrinsics.principal_point.x
        ppy = camera_intrinsics.principal_point.y
        intrinsics = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]])
        return intrinsics

    def get_ros_TransformStamped_vision_T_body(
        self, frame_tree_snapshot
    ) -> TransformStamped:
        """
        Generates vision_T_body transform as a ROS TransformStamped message from the FrameTreeSnapshot

        Args:
            frame_tree_snapshot (FrameTreeSnapshot): FrameTreeSnapshot from which to extract the vision_T_body transform

        Returns:
            ros_TransformStamped_vision_T_body (TransformStamped): ROS TransformStamped message containing the vision_T_body transform
        """
        vision_tform_body = get_vision_tform_body(frame_tree_snapshot)
        ros_TransformStamped_vision_T_body = bd_SE3Pose_to_ros_TransformStamped(
            bd_se3=vision_tform_body, parent_frame=rf.SPOT_WORLD, child_frame=rf.SPOT
        )
        return ros_TransformStamped_vision_T_body

    def get_ros_Pose_vision_T_body(self, frame_tree_snapshot) -> Pose:
        """
        Generates vision_T_body transform as a ROS Pose message from the FrameTreeSnapshot

        Args:
            frame_tree_snapshot (FrameTreeSnapshot): FrameTreeSnapshot from which to extract the vision_T_body transform

        Returns:
            ros_Pose_vision_T_body (Pose): ROS Pose message containing the vision_T_body transform
        """
        vision_tform_body = get_vision_tform_body(frame_tree_snapshot)
        ros_Pose_vision_T_body = bd_SE3Pose_to_ros_Pose(bd_se3=vision_tform_body)
        return ros_Pose_vision_T_body

    def get_magnum_Matrix4_spot_a_T_b(self, a: str, b: str, tree=None) -> mn.Matrix4:
        """
        Gets transformation from 'a' frame to 'b' frame such that a_T_b.
        `a` & `b` takes string values of the name of the frames
        tree is optional, its the kinematic transforms tree if none it will make new from robot's state
        image sources also give us kinematic transforms trees that can be sent here
        """
        frame_tree_snapshot = (
            self.get_robot_state().kinematic_state.transforms_snapshot
            if tree is None
            else tree
        )
        # BD api's function get_a_tform_b uses given transforms tree to make a_T_b
        se3_pose = get_a_tform_b(frame_tree_snapshot, a, b)
        # convert SE3Pose to magnum Matrix 4 transformation, seperate rotation & translation
        pos = se3_pose.get_translation()
        quat = se3_pose.rotation.normalize()
        quat = quaternion.quaternion(quat.w, quat.x, quat.y, quat.z)
        rotation_matrix = mn.Quaternion(quat.imag, quat.real).to_matrix()
        translation = mn.Vector3(*pos)
        mn_transformation = mn.Matrix4.from_(rotation_matrix, translation)
        return mn_transformation

    def get_sophus_SE3_spot_a_T_b(
        self, frame_tree_snapshot: dict, a: str, b: str
    ) -> sp.SE3:
        """
        Gets transformation from `a` frame to `b` frame as a_T_b.
        Takes in `a` & `b` as string names for the frames
        Frame tree snapshot SHOULD be passed for frames other than `body`, `vision` or `odom`
        Frame tree snapshot for any camera will be a part of image_response object received from that camera
        """
        if frame_tree_snapshot is None:
            frame_tree_snapshot = (
                self.get_robot_state().kinematic_state.transforms_snapshot
            )
        se3_pose = get_a_tform_b(frame_tree_snapshot, a, b)
        pos = se3_pose.get_translation()
        quat = se3_pose.rotation.normalize()
        return sp.SE3(quat.to_matrix(), pos)

    def get_ee_pos_in_body_frame(self, frame_name: str = "body"):
        """
        Return ee xyz position and roll, pitch, yaw
        """
        # Get transformation
        body_T_hand = self.get_ee_transform(frame_name)

        # Get rotation. BD API returns values with the order of yaw, pitch, roll.
        theta = math_helpers.quat_to_eulerZYX(body_T_hand.rotation)
        # Change the order to roll, pitch, yaw
        theta = np.array(theta)[::-1]

        # Get position x,y,z
        position = (
            self.robot_state_client.get_robot_state()
            .kinematic_state.transforms_snapshot.child_to_parent_edge_map["hand"]
            .parent_tform_child.position
        )

        return np.array([position.x, position.y, position.z]), theta

    def get_ee_transform(self, frame_name: str = "body"):
        """
        Get ee transformation from base (body) to hand frame
        """
        body_T_hand = get_a_tform_b(
            self.robot_state_client.get_robot_state().kinematic_state.transforms_snapshot,
            frame_name,
            "hand",
        )
        return body_T_hand

    def get_ee_transform_in_vision_frame(self):
        """
        Get ee transformation from vision (global) to hand frame
        """
        # Get the euler z,y,x
        vision_T_hand = get_a_tform_b(
            self.robot_state_client.get_robot_state().kinematic_state.transforms_snapshot,
            "vision",
            "hand",
        )
        return vision_T_hand

    def get_ee_quaternion_in_body_frame(self, frame_name="body"):
        """
        Get ee's quaternion
        """
        body_T_hand = self.get_ee_transform(frame_name)
        quat = body_T_hand.rotation
        quat = quaternion.quaternion(quat.w, quat.x, quat.y, quat.z)
        return quat

    def get_cur_ee_pose_offset(self):
        """Get the current ee pose offset"""
        # Get base to hand's transformation
        ee_transform = self.get_magnum_Matrix4_spot_a_T_b("vision", "hand")
        # Get the base transformation
        base_transform = self.get_magnum_Matrix4_spot_a_T_b("vision", "body")
        # Do offset
        base_to_arm_offset = 0.292
        base_transform.translation = base_transform.transform_point(
            mn.Vector3(base_to_arm_offset, 0, 0)
        )
        # Get ee relative to base
        ee_position = (base_transform.inverted() @ ee_transform).translation
        base_T_hand_yaw = get_angle_between_forward_and_target(ee_position)
        return base_T_hand_yaw

    def construct_cabinet_task(
        self,
        velocity_normalized,
        force_limit=40,
        target_angle=None,
        position_control=False,
        reset_estimator_bool=True,
    ):
        """Helper function for opening/closing cabinets

        params:
        + velocity_normalized: normalized task tangential velocity in range [-1.0, 1.0]
        In position mode, this normalized velocity is used as a velocity limit for the planned trajectory.
        + force_limit (optional): positive value denoting max force robot will exert along task dimension
        + target_angle: target angle displacement (rad) in task space. This is only used if position_control == True
        + position_control: if False will move the affordance in velocity control, if True will move by target_angle
        with a max velocity of velocity_limit
        + reset_estimator_bool: boolean that determines if the estimator should compute a task frame from scratch.
        Only set to False if you want to re-use the estimate from the last constrained manipulation action.

        Output:
        + command: api command object

        Notes:
        In this function, we assume the initial motion of the cabinet is
        along the x-axis of the hand (forward and backward). If the initial
        grasp is such that the initial motion needs to be something else,
        change the force direction.
        """
        angle_sign, angle_value, tangential_velocity = get_position_and_vel_values(
            target_angle, velocity_normalized, force_limit, position_control
        )

        frame_name = "hand"
        force_lim = force_limit
        # Setting a placeholder value that doesn't matter, since we don't
        # apply a pure torque in this task.
        torque_lim = 5.0
        force_direction = geometry_pb2.Vec3(x=angle_sign * -1.0, y=0.0, z=0.0)
        torque_direction = geometry_pb2.Vec3(x=0.0, y=0.0, z=0.0)
        init_wrench_dir = geometry_pb2.Wrench(
            force=force_direction, torque=torque_direction
        )
        task_type = (
            basic_command_pb2.ConstrainedManipulationCommand.Request.TASK_TYPE_R3_CIRCLE_FORCE
        )
        reset_estimator = wrappers_pb2.BoolValue(value=reset_estimator_bool)
        control_mode = POSITION_MODE if position_control else VELOCITY_MODE

        command = RobotCommandBuilder.constrained_manipulation_command(
            task_type=task_type,
            init_wrench_direction_in_frame_name=init_wrench_dir,
            force_limit=force_lim,
            torque_limit=torque_lim,
            tangential_speed=tangential_velocity,
            frame_name=frame_name,
            control_mode=control_mode,
            target_angle=angle_value,
            reset_estimator=reset_estimator,
        )
        return command


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
    if (
        image_response.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16
        and image_response.shot.image.format == image_pb2.Image.FORMAT_RAW
    ):
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
