# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import copy
import time
from typing import Any, Dict

import magnum as mn
import numpy as np
import quaternion
import rospy
from spot_rl.envs.base_env import SpotBaseEnv
from spot_rl.utils.geometry_utils import is_position_within_bounds
from spot_wrapper.spot import Spot


class SpotPlaceEnv(SpotBaseEnv):
    def __init__(self, config, spot: Spot):
        super().__init__(config, spot)
        self.place_target = None
        self.place_target_is_local = False

        self.ee_gripper_offset = mn.Vector3(config.EE_GRIPPER_OFFSET)
        self.placed = False

    def reset(self, place_target, target_is_local=False, *args, **kwargs):
        assert place_target is not None
        self.place_target = np.array(place_target)
        self.place_target_is_local = target_is_local

        self.reset_arm()

        observations = super().reset()
        self.placed = False
        return observations

    def step(self, action_dict: Dict[str, Any], *args, **kwargs):
        gripper_pos_in_base_frame = self.get_gripper_position_in_base_frame_hab()
        place_target_in_base_frame = self.get_base_frame_place_target_hab()
        place = is_position_within_bounds(
            gripper_pos_in_base_frame,
            place_target_in_base_frame,
            self.config.SUCC_XY_DIST,
            self.config.SUCC_Z_DIST,
            convention="habitat",
        )

        # Update the action_dict with place flag
        action_dict["place"] = place

        return super().step(action_dict=action_dict, *args, **kwargs)

    def get_success(self, observations):
        return self.place_attempted

    def get_observations(self):
        observations = {
            "joint": self.get_arm_joints(),
            "obj_start_sensor": self.get_place_sensor(),
        }

        return observations


class SpotSemanticPlaceEnv(SpotBaseEnv):
    def __init__(self, config, spot: Spot):
        config.INITIAL_ARM_JOINT_ANGLES = copy.deepcopy(
            config.INITIAL_ARM_JOINT_ANGLES_SEMANTIC_PLACE
        )
        super().__init__(config, spot)
        self.place_target = None
        self.place_target_is_local = False

        self.ee_gripper_offset = mn.Vector3(config.EE_GRIPPER_OFFSET)
        self.placed = False
        self._time_step = 0
        self.initial_ee_pose = None
        self.target_object_pose = None
        # Overwrite joint limits for semantic_place skills
        self.arm_lower_limits = np.deg2rad(config.ARM_LOWER_LIMITS_FOR_SEMANTIC_PLACE)
        self.arm_upper_limits = np.deg2rad(config.ARM_UPPER_LIMITS_FOR_SEMANTIC_PLACE)

    def get_angle_v2(self, x, y):
        if np.linalg.norm(x) != 0:
            x_norm = x / np.linalg.norm(x)
        else:
            x_norm = x

        if np.linalg.norm(y) != 0:
            y_norm = y / np.linalg.norm(y)
        else:
            y_norm = y

        return np.arccos(np.clip(np.dot(x_norm, y_norm), -1, 1))

    def get_angle(self, rel_pos):
        """Get angle"""
        forward = np.array([1.0, 0, 0])
        rel_pos = np.array(rel_pos)
        forward = forward[[0, 1]]
        rel_pos = rel_pos[[0, 1]]

        heading_angle = self.get_angle_v2(forward, rel_pos)
        c = np.cross(forward, rel_pos) < 0
        if not c:
            heading_angle = -1.0 * heading_angle
        return heading_angle

    def get_ee_target_orientation(self):
        """ee target orientation"""
        # Get base T
        base_T = self.spot.get_magnum_Matrix4_spot_a_T_b("vision", "body")
        height = base_T.translation[2]

        # Get ee T
        ee_T = self.spot.get_magnum_Matrix4_spot_a_T_b("vision", "hand")

        # Get the glocal location of the place target
        target = np.copy(self.place_target)
        # Offset when we register the point
        target[2] -= height
        obj_local_pos = base_T.inverted().transform_point(target)
        # Get the angle
        angle = self.get_angle(obj_local_pos)
        # Rotate the base by the angle
        base_T = base_T @ mn.Matrix4.rotation_z(mn.Rad(angle))

        base_T_ee = base_T.inverted() @ ee_T

        quat = quaternion.from_rotation_matrix(base_T_ee.rotation())

        return quat

    def get_cur_ee_pose_offset(self):
        # Get base to hand's transformation
        ee_transform = self.spot.get_magnum_Matrix4_spot_a_T_b("vision", "hand")
        # Get the base transformation
        base_transform = self.spot.get_magnum_Matrix4_spot_a_T_b("vision", "body")
        # Do offset
        base_transform.translation = base_transform.transform_point(
            mn.Vector3(0.292, 0, 0)
        )
        # Get ee relative to base
        ee_position = (base_transform.inverted() @ ee_transform).translation
        base_T_hand_yaw = self.get_angle(ee_position)
        return base_T_hand_yaw

    def reset(self, place_target, target_is_local=False, *args, **kwargs):
        assert place_target is not None
        self.place_target = np.array(place_target)
        self.place_target_is_local = target_is_local

        self._time_step = 0
        if not self.config.RUNNING_AFTER_GRASP_FOR_PLACE:
            self.reset_arm()
            # We wait for a second to let the arm in the placing
            # ready location
            time.sleep(1.5)
            # There is a bit of mistchmatch of joints after resetting
            # So we reset the arm again
            # ee_position, ee_orientation = self.spot.get_ee_pos_in_body_frame()
            # self.spot.move_gripper_to_point(ee_position, [np.pi / 2, 0, 0])

        # Set the initial ee pose
        self.initial_ee_pose = self.spot.get_ee_pos_in_body_frame_quat()
        # Set the target pose
        # self.target_object_pose = self.get_ee_target_orientation()
        self.target_object_pose = self.spot.get_ee_pos_in_body_frame_quat()
        # self.target_object_pose = quaternion.quaternion(
        #     0.709041893482208,
        #     0.704837739467621,
        #     -0.00589140923693776,
        #     -0.0207040011882782,
        # )
        rospy.set_param("is_gripper_blocked", 1)
        observations = super().reset()
        self.placed = False
        return observations

    def step(self, action_dict: Dict[str, Any], *args, **kwargs):
        # <= 0 for unsnap
        place = action_dict.get("grip_action", None) < 0.
        xyz = self.get_place_sensor(True)
        if (
            abs(xyz[2]) < 0.05
            and np.linalg.norm(np.array([xyz[0], xyz[1]])) < 0.25
            and self._time_step >= 50
        ):
            place = True
        if self._time_step >= 75:
            place = True

        self._time_step += 1
        print(
            "dis to goal:", np.linalg.norm(self.get_place_sensor(True)), self._time_step
        )
        print(f"Time step Internal {self._time_step}")
        print("place in base place env:", place)
        action_dict["place"] = place
        action_dict["travel_time_scale"] = 1.0 / 0.9 * 1.75
        # place = action_dict.get("grip_action", None) <= 0.0
        print("grip_action in sem place env:", action_dict.get("grip_action", None))
        action_dict["place"] = place
        action_dict["semantic_place"] = place
        return super().step(action_dict, *args, **kwargs)

    def get_success(self, observations):
        return self.place_attempted

    def get_observations(self):
        assert self.initial_ee_pose is not None
        assert self.target_object_pose is not None

        # Get the gaol sensor
        obj_goal_sensor = self.get_place_sensor(True)
        # obj_goal_sensor = self.get_place_sensor_norm()

        # Get the delta ee orientation
        current_gripper_orientation = self.spot.get_ee_pos_in_body_frame_quat()
        delta_ee = self.spot.angle_between_quat(
            self.initial_ee_pose, current_gripper_orientation
        )
        delta_ee = np.array([delta_ee], dtype=np.float32)

        # Get the delta object orientation
        delta_obj = self.spot.angle_between_quat(
            self.target_object_pose, current_gripper_orientation
        )
        # remove the offset from the base to object
        delta_obj = np.array(
            [delta_obj - abs(self.get_cur_ee_pose_offset())], dtype=np.float32
        )
        # Get the jaw image
        arm_depth, _ = self.get_gripper_images()

        print("rpy to init ee:", delta_ee)
        print("rpy to targ obj:", delta_obj)
        print(
            "xyz to targ obj:",
            obj_goal_sensor,
            self.initial_ee_pose,
            current_gripper_orientation,
        )
        # self.spot.move_gripper_to_point(np.array([1.35, 0.17, 0.35]),[0.0,0,0])

        observations = {
            "obj_goal_sensor": obj_goal_sensor,
            "relative_initial_ee_orientation": delta_ee,
            "relative_target_object_orientation": delta_obj,
            "articulated_agent_jaw_depth": arm_depth,
            "joint": self.get_arm_joints(semantic_place=True),
            "is_holding": np.ones((1,)),
        }
        print("self.get_cur_ee_pose_offset()", self.get_cur_ee_pose_offset())
        return observations
