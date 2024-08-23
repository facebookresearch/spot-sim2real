# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import magnum as mn
import numpy as np
import rospy
from bosdyn.client.frame_helpers import get_a_tform_b
from bosdyn.client.math_helpers import quat_to_eulerZYX
from spot_rl.envs.base_env import SpotBaseEnv
from spot_rl.utils.pixel_to_3d_conversion_utils import (
    get_3d_point,
    sample_patch_around_point,
)
from spot_rl.utils.utils import ros_topics as rt
from spot_wrapper.spot import Spot, wrap_heading
from std_msgs.msg import Float32, String

MAX_PUBLISH_FREQ = 20
MAX_DEPTH = 3.5
MAX_HAND_DEPTH = 1.7
DETECTIONS_BUFFER_LEN = 30
LEFT_CROP = 124
RIGHT_CROP = 60
NEW_WIDTH = 228
NEW_HEIGHT = 240
ORIG_WIDTH = 640
ORIG_HEIGHT = 480
WIDTH_SCALE = 0.5
HEIGHT_SCALE = 0.5


class SpotNavEnv(SpotBaseEnv):
    def __init__(self, config, spot: Spot):
        super().__init__(config, spot)
        self._goal_xy = None
        self._enable_nav_by_hand = False
        self._enable_dynamic_yaw = False
        self._enable_dynamic_goal_xy = False
        self.detections_location = []  # type: ignore
        self.goal_heading = None
        self.succ_distance = config.SUCCESS_DISTANCE
        self.succ_angle = np.deg2rad(config.SUCCESS_ANGLE_DIST)

        rospy.Subscriber(rt.OPEN_VOC_OBJECT_DETECTOR_TOPIC, String, self.bbox_cb)

    def enable_nav_by_hand(self):
        if not self._enable_nav_by_hand:
            self._enable_nav_by_hand = True
            print(
                f"{self.node_name} Enabling nav goal change get_nav_observation by base switched to get_nav_observation by hand fn"
            )
            self.backup_fn_of_get_nav_observation_that_operates_by_robot_base = (
                self.get_nav_observation
            )
            self.get_nav_observation = self.get_nav_observation_by_hand

    def bbox_cb(self, msg):
        # The example format of the msg: "cup,0,0,0;table,0,0,0"
        objects_detected = msg.data.split(";")
        self.objects_detected = []
        for object_detected in objects_detected:
            if object_detected == "":
                continue
            try:
                class_label, x, y, z = object_detected.split(",")
            except Exception as e:
                print(
                    f"Fail to split the object detected due to {e} with object_detected being {object_detected}."
                )
                continue
            self.objects_detected.append((class_label, [float(x), float(y), float(z)]))

    def disable_nav_by_hand(self):
        if self._enable_nav_by_hand:
            self.get_nav_observation = (
                self.backup_fn_of_get_nav_observation_that_operates_by_robot_base
            )
            self._enable_nav_by_hand = False
            print(
                f"{self.node_name} Disabling nav goal change get_nav_observation by base fn restored"
            )

    def reset(
        self,
        goal_xy,
        goal_heading,
        dynamic_yaw=False,
        dynamic_goal_xy=False,
        target_object=None,
    ):
        self._goal_xy = np.array(goal_xy, dtype=np.float32)
        self.goal_heading = goal_heading
        observations = super().reset()
        self.detections_location = []

        self._enable_dynamic_yaw = dynamic_yaw
        self._enable_dynamic_goal_xy = dynamic_goal_xy

        assert len(self._goal_xy) == 2

        self._cur_arm_depth = None

        if self._enable_dynamic_yaw or self._enable_dynamic_goal_xy:
            self.succ_distance = self.config.SUCCESS_DISTANCE_FOR_DYNAMIC_YAW_NAV
            self.succ_angle = np.deg2rad(
                self.config.SUCCESS_ANGLE_DIST_FOR_DYNAMIC_YAW_NAV
            )
        else:
            self.succ_distance = self.config.SUCCESS_DISTANCE
            self.succ_angle = np.deg2rad(self.config.SUCCESS_ANGLE_DIST)

        if self._enable_dynamic_goal_xy:
            rospy.set_param("enable_tracking", False)
            rospy.set_param("object_targets", target_object)
            rospy.set_param("object_target", target_object)

        self.dist_to_goal = float("inf")
        self.abs_goal_heading = float("inf")

        return observations

    def get_success(self, observations, succ_set_base=True):
        succ = self.get_nav_success(observations, self.succ_distance, self.succ_angle)
        if succ and succ_set_base:
            self.spot.set_base_velocity(0.0, 0.0, 0.0, 1 / self.ctrl_hz)
        return succ

    def get_hand_xy_theta(self, use_boot_origin=False):
        """
        Much like spot.get_xy_yaw(), this function returns x,y,yaw of the hand camera instead of base such as in spot.get_xy_yaw()
        Accepts the same parameter use_boot_origin of type bool like the function mentioned in above line, this determines whether the calculation is from the vision frame or robot'home
        If true, then the location is calculated from the vision frame else from home/dock
        Returns x,y,theta useful in head/hand based navigation used in Heurisitic Mobile Navigation
        """
        vision_T_hand = get_a_tform_b(
            self.spot.robot_state_client.get_robot_state().kinematic_state.transforms_snapshot,
            "vision",
            "hand",
        )
        theta = quat_to_eulerZYX(vision_T_hand.rotation)[0]
        point_in_global_2d = np.array([vision_T_hand.x, vision_T_hand.y])
        return (
            (point_in_global_2d[0], point_in_global_2d[1], theta)
            if use_boot_origin
            else self.spot.xy_yaw_global_to_home(
                point_in_global_2d[0], point_in_global_2d[1], theta
            )
        )

    def get_nav_observation_by_hand(self, goal_xy, goal_heading):

        observations = self.get_head_depth()

        # Get rho theta observation
        x, y, yaw = self.get_hand_xy_theta()
        curr_xy = np.array([x, y], dtype=np.float32)
        rho = np.linalg.norm(curr_xy - goal_xy)
        theta = wrap_heading(np.arctan2(goal_xy[1] - y, goal_xy[0] - x) - yaw)
        rho_theta = np.array([rho, theta], dtype=np.float32)

        # Get goal heading observation
        goal_heading_ = -np.array([wrap_heading(goal_heading - yaw)], dtype=np.float32)
        observations["target_point_goal_gps_and_compass_sensor"] = rho_theta
        observations["goal_heading"] = goal_heading_

        return observations

    def get_current_angle_for_target_facing(self):
        vector_robot_to_target = self._goal_xy - np.array([self.x, self.y])
        vector_robot_to_target = vector_robot_to_target / np.linalg.norm(
            vector_robot_to_target
        )
        vector_forward_robot = np.array(
            self.curr_transform.transform_vector(mn.Vector3(1, 0, 0))
        )[[0, 1]]
        vector_forward_robot = vector_forward_robot / np.linalg.norm(
            vector_forward_robot
        )

        return vector_robot_to_target, vector_forward_robot

    def affordance_prediction(
        self,
        depth_raw: np.ndarray,
        mask: np.ndarray,
        camera_intrinsics,
        center_pixel: np.ndarray,
    ) -> np.ndarray:
        """
        Accepts
        depth_raw: np.array HXW, 0.-2000.
        mask: HXW, bool mask
        camera_intrinsics:spot camera intrinsic object
        center_pixel: np.array of length 2
        Returns: Suitable point on object to grasp
        """

        mask = np.where(mask > 0, 1, 0).astype(depth_raw.dtype)
        depth_image_masked = depth_raw * mask[...].astype(depth_raw.dtype)

        non_zero_indices = np.nonzero(depth_image_masked)
        # Calculate the bounding box coordinates
        y_min, y_max = non_zero_indices[0].min(), non_zero_indices[0].max()
        x_min, x_max = non_zero_indices[1].min(), non_zero_indices[1].max()
        cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
        Z = float(sample_patch_around_point(int(cx), int(cy), depth_raw) * 1.0)
        point_in_gripper = get_3d_point(camera_intrinsics, center_pixel, Z)

        return point_in_gripper

    def get_observations(self):
        if self._enable_dynamic_yaw:
            # Modify the goal_heading here based on the current robot orientation
            (
                vector_robot_to_target,
                vector_forward_robot,
            ) = self.get_current_angle_for_target_facing()
            x1 = (
                vector_robot_to_target[1] * vector_forward_robot[0]
                - vector_robot_to_target[0] * vector_forward_robot[1]
            )
            x2 = (
                vector_robot_to_target[0] * vector_forward_robot[0]
                + vector_robot_to_target[1] * vector_forward_robot[1]
            )
            rotation_delta = np.arctan2(x1, x2)
            self.goal_heading = wrap_heading(self.yaw + rotation_delta)

        if (
            self._enable_dynamic_goal_xy
            and self.dist_to_goal < 1.5
            and self.abs_goal_heading < np.deg2rad(75)
        ):
            objects_detected = self.objects_detected.copy()
            if len(objects_detected) != 0:
                print("change goal!!!!!!")
                self._goal_xy = np.array(
                    [objects_detected[0][1][0], objects_detected[0][1][1]]
                )

        print(f"self._goal_xy: {self._goal_xy}")
        return self.get_nav_observation(self._goal_xy, self.goal_heading)

    def step(self, *args, **kwargs):
        observations, reward, done, info = super().step(*args, **kwargs)

        # Slow the base down if we are close to the nav target to slow down the the heading changes
        self.dist_to_goal, _ = observations["target_point_goal_gps_and_compass_sensor"]
        self.abs_goal_heading = abs(observations["goal_heading"][0])

        if self._enable_dynamic_yaw:
            if self.dist_to_goal < 1.5 and self.abs_goal_heading < np.rad2deg(45):
                self.slowdown_base = 0.5
            else:
                self.slowdown_base = -1

        return observations, reward, done, info
