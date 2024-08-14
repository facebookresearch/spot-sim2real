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
from spot_wrapper.utils import angle_between_quat


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
    """This is Spot semantic place class"""

    def __init__(self, config, spot: Spot):
        # We set the initial arm joints
        config.INITIAL_ARM_JOINT_ANGLES = copy.deepcopy(
            config.INITIAL_ARM_JOINT_ANGLES_SEMANTIC_PLACE
        )
        super().__init__(config, spot)

        # Define the place variables
        self.place_target = None
        self.place_target_is_local = False
        self.placed = False

        # Set gripper variables
        self.ee_gripper_offset = mn.Vector3(config.EE_GRIPPER_OFFSET)

        # Define the observation variables
        self.initial_ee_pose = None
        self.target_object_pose = None

        # Overwrite joint limits for semantic_place skills
        self.arm_lower_limits = np.deg2rad(config.ARM_LOWER_LIMITS_SEMANTIC_PLACE)
        self.arm_upper_limits = np.deg2rad(config.ARM_UPPER_LIMITS_SEMANTIC_PLACE)

        # Place steps
        self._time_step = 0

    def decide_init_arm_joint(self, ee_orientation_at_grasping):
        """Decide the place location"""

        # User does not set the gripper orientation
        if ee_orientation_at_grasping is None:
            self.initial_arm_joint_angles = np.deg2rad(
                self.config.INITIAL_ARM_JOINT_ANGLES_SEMANTIC_PLACE
            )
        else:
            # Get the pitch and yaw
            pitch = ee_orientation_at_grasping[1]
            yaw = ee_orientation_at_grasping[2]
            print("ee_orientation_at_grasping:", ee_orientation_at_grasping)
            if abs(pitch) <= 1.309:  # 75 degree in pitch
                if yaw > 0:  # gripper is in object's right hand side
                    self.initial_arm_joint_angles = np.deg2rad(
                        self.config.INITIAL_ARM_JOINT_ANGLES_SEMANTIC_PLACE
                    )
                else:  # gripper is in object's left hand side
                    self.initial_arm_joint_angles = np.deg2rad(
                        self.config.INITIAL_ARM_JOINT_ANGLES_SEMANTIC_PLACE_LEFT_HAND
                    )
            else:
                self.initial_arm_joint_angles = np.deg2rad(
                    self.config.INITIAL_ARM_JOINT_ANGLES_SEMANTIC_PLACE_TOP_DOWN
                )

    def reset(self, place_target, target_is_local=False, *args, **kwargs):
        assert place_target is not None
        self.place_target = np.array(place_target)
        self.place_target_is_local = target_is_local

        self._time_step = 0

        # Decide the reset arm angle and then reset the arm
        self.decide_init_arm_joint(kwargs["ee_orientation_at_grasping"])
        self.reset_arm()

        # We wait for a second to let the arm in the placing
        # ready location
        time.sleep(1.0)
        # Sometimes, there will be a bit of mistchmatch of joints after resetting
        # So we can reset the arm again here using the following
        # ee_position, ee_orientation = self.spot.get_ee_pos_in_body_frame()
        # self.spot.move_gripper_to_point(ee_position, [np.pi / 2, 0, 0])

        # Set the initial ee pose
        self.initial_ee_pose = self.spot.get_ee_quaternion_in_body_frame()
        # Set the target pose
        self.target_object_pose = self.spot.get_ee_quaternion_in_body_frame()
        # Automatically use intelrealsense camera
        rospy.set_param("is_gripper_blocked", 1)
        observations = super().reset()
        rospy.set_param("is_whiten_black", False)
        self.placed = False
        return observations

    def step(self, action_dict: Dict[str, Any], *args, **kwargs):

        place = False

        # Place command is issued if the place action is smaller than zero
        place = action_dict.get("grip_action", None) <= 0.0

        # If the time steps have been passed for 50 steps and gripper is in the desired place location
        cur_place_sensor_xyz = self.get_place_sensor(True)
        if (
            abs(cur_place_sensor_xyz[2]) < 0.05
            and np.linalg.norm(
                np.array([cur_place_sensor_xyz[0], cur_place_sensor_xyz[1]])
            )
            < 0.25
            and self._time_step >= 50
        ):
            place = True

        # If the time steps have been passed for 75 steps, we will just place the object
        if self._time_step >= 75:
            place = True

        self._time_step += 1

        # Write into action dict
        action_dict["place"] = place
        action_dict["semantic_place"] = place

        # Set the travel time scale so that the arm movement is smooth
        return super().step(
            action_dict, travel_time_scale=1.0 / 0.9 * 1.75, *args, **kwargs
        )

    def get_success(self, observations):
        return self.place_attempted

    def get_observations(self):
        assert self.initial_ee_pose is not None
        assert self.target_object_pose is not None

        # Get the gaol sensor
        obj_goal_sensor = self.get_place_sensor(True)

        # Get the delta ee orientation
        current_gripper_orientation = self.spot.get_ee_quaternion_in_body_frame()
        delta_ee = angle_between_quat(self.initial_ee_pose, current_gripper_orientation)
        delta_ee = np.array([delta_ee], dtype=np.float32)

        # Get the delta object orientation
        delta_obj = angle_between_quat(
            self.target_object_pose, current_gripper_orientation
        )
        # remove the offset from the base to object
        delta_obj = np.array(
            [delta_obj - abs(self.spot.get_cur_ee_pose_offset())], dtype=np.float32
        )
        # Get the jaw image
        arm_depth, _ = self.get_gripper_images()

        observations = {
            "obj_goal_sensor": obj_goal_sensor,
            "relative_initial_ee_orientation": delta_ee,
            "relative_target_object_orientation": delta_obj,
            "articulated_agent_jaw_depth": arm_depth,
            "joint": self.get_arm_joints(self.config.SEMANTIC_PLACE_JOINT_BLACKLIST),
            "is_holding": np.ones((1,)),
        }
        return observations


class SpotSemanticPlaceEEEnv(SpotBaseEnv):
    """This is Spot semantic place class"""

    def __init__(self, config, spot: Spot):
        # We set the initial arm joints
        config.INITIAL_ARM_JOINT_ANGLES = copy.deepcopy(
            config.INITIAL_ARM_JOINT_ANGLES_SEMANTIC_PLACE
        )
        super().__init__(config, spot)

        # Define the place variables
        self.place_target = None
        self.place_target_is_local = False
        self.placed = False

        # Set gripper variables
        self.ee_gripper_offset = mn.Vector3(config.EE_GRIPPER_OFFSET)

        # Define the observation variables
        self.initial_ee_pose = None
        self.target_object_pose = None

        # Overwrite joint limits for semantic_place skills
        self.arm_lower_limits = np.deg2rad(config.ARM_LOWER_LIMITS_SEMANTIC_PLACE)
        self.arm_upper_limits = np.deg2rad(config.ARM_UPPER_LIMITS_SEMANTIC_PLACE)

        # Place steps
        self._time_step = 0

    def decide_init_arm_joint(self, ee_orientation_at_grasping):
        """Decide the place location"""

        # User does not set the gripper orientation
        if ee_orientation_at_grasping is None:
            self.initial_arm_joint_angles = np.deg2rad(
                self.config.INITIAL_ARM_JOINT_ANGLES_SEMANTIC_PLACE
            )
        else:
            # Get the pitch and yaw
            pitch = ee_orientation_at_grasping[1]
            yaw = ee_orientation_at_grasping[2]
            print("ee_orientation_at_grasping:", ee_orientation_at_grasping)
            if abs(pitch) <= 1.309:  # 75 degree in pitch
                if yaw > 0:  # gripper is in object's right hand side
                    self.initial_arm_joint_angles = np.deg2rad(
                        self.config.INITIAL_ARM_JOINT_ANGLES_SEMANTIC_PLACE
                    )
                else:  # gripper is in object's left hand side
                    self.initial_arm_joint_angles = np.deg2rad(
                        self.config.INITIAL_ARM_JOINT_ANGLES_SEMANTIC_PLACE_LEFT_HAND
                    )
            else:
                self.initial_arm_joint_angles = np.deg2rad(
                    self.config.INITIAL_ARM_JOINT_ANGLES_SEMANTIC_PLACE_TOP_DOWN
                )

    def reset(self, place_target, target_is_local=False, *args, **kwargs):
        assert place_target is not None
        self.place_target = np.array(place_target)
        self.place_target_is_local = target_is_local

        self._time_step = 0

        # Decide the reset arm angle and then reset the arm
        self.decide_init_arm_joint(kwargs["ee_orientation_at_grasping"])
        self.reset_arm()

        # We wait for a second to let the arm in the placing
        # ready location
        time.sleep(1.0)
        # Sometimes, there will be a bit of mistchmatch of joints after resetting
        # So we can reset the arm again here using the following
        # ee_position, ee_orientation = self.spot.get_ee_pos_in_body_frame()
        # self.spot.move_gripper_to_point(ee_position, [np.pi / 2, 0, 0])

        # Set the initial ee pose
        self.initial_ee_pose = self.spot.get_ee_quaternion_in_body_frame()
        # Set the target pose
        self.target_object_pose = self.spot.get_ee_quaternion_in_body_frame()
        # Automatically use intelrealsense camera
        rospy.set_param("is_gripper_blocked", 1)
        observations = super().reset()
        rospy.set_param("is_whiten_black", False)
        self.placed = False
        return observations

    def step(self, action_dict: Dict[str, Any], *args, **kwargs):

        place = False

        # Place command is issued if the place action is smaller than zero
        # TODO: semantic place ee: check how grip_action behaves!
        place = action_dict.get("grip_action", None) <= 0.0

        # If the time steps have been passed for 50 steps and gripper is in the desired place location
        cur_place_sensor_xyz = self.get_place_sensor(True)
        if (
            abs(cur_place_sensor_xyz[2]) < 0.05
            and np.linalg.norm(
                np.array([cur_place_sensor_xyz[0], cur_place_sensor_xyz[1]])
            )
            < 0.25
            and self._time_step >= 50
        ):
            place = True

        # If the time steps have been passed for 75 steps, we will just place the object
        if self._time_step >= 75:
            place = True

        self._time_step += 1

        # Write into action dict
        action_dict["place"] = place
        action_dict["semantic_place"] = place

        # Set the travel time scale so that the arm movement is smooth
        return super().step(
            action_dict, travel_time_scale=1.0 / 0.9 * 1.75, *args, **kwargs
        )

    def get_success(self, observations):
        return self.place_attempted

    def get_observations(self):
        assert self.initial_ee_pose is not None
        assert self.target_object_pose is not None

        # Get the gaol sensor
        obj_goal_sensor = self.get_place_sensor(True)

        # Get the delta ee orientation
        current_gripper_orientation = self.spot.get_ee_quaternion_in_body_frame()
        delta_ee = angle_between_quat(self.initial_ee_pose, current_gripper_orientation)
        delta_ee = np.array([delta_ee], dtype=np.float32)

        # Get the delta object orientation
        delta_obj = angle_between_quat(
            self.target_object_pose, current_gripper_orientation
        )
        # remove the offset from the base to object
        delta_obj = np.array(
            [delta_obj - abs(self.spot.get_cur_ee_pose_offset())], dtype=np.float32
        )
        # Get the jaw image
        arm_depth, _ = self.get_gripper_images()

        # Get ee's pose -- x,y,z
        xyz, _ = self.spot.get_ee_pos_in_body_frame()
        observations = {
            "obj_goal_sensor": obj_goal_sensor,
            "relative_initial_ee_orientation": delta_ee,
            "relative_target_object_orientation": delta_obj,
            "articulated_agent_jaw_depth": arm_depth,
            "ee_pos": xyz,
            "is_holding": np.ones((1,)),
        }
        return observations
