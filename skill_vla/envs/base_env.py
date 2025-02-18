# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json

# mypy: ignore-errors
import os
import os.path as osp
import time
from typing import Any, Dict

import cv2
import gym
import numpy as np
import rospy
from spot_rl.utils.robot_subscriber import SpotRobotSubscriberMixin
from spot_rl.utils.utils import FixSizeOrderedDict, arr2str, object_id_to_object_name
from spot_rl.utils.utils import ros_topics as rt
from spot_wrapper.spot import Spot, wrap_heading

try:
    import magnum as mn
except Exception:
    pass

DETECTIONS_BUFFER_LEN = 30
LEFT_CROP = 124
RIGHT_CROP = 60
NEW_WIDTH = 228
NEW_HEIGHT = 240
ORIG_WIDTH = 640
ORIG_HEIGHT = 480
WIDTH_SCALE = 0.5
HEIGHT_SCALE = 0.5


def pad_action(action):
    """Pad action zero for the non-controllable indices of the arm."""
    # A special case for semantic place skills. The semantic skill
    # controls 5 joints. However, in the real world testing, we disable
    # the last joint control for better performance.
    return np.array([*action[:3], 0.0, action[3], 0.0])


def rescale_actions(actions, action_thresh=0.05, silence_only=False):
    actions = np.clip(actions, -1, 1)
    # Silence low actions
    actions[np.abs(actions) < action_thresh] = 0.0
    if silence_only:
        return actions

    # Remap action scaling to compensate for silenced values
    action_offsets = np.ones_like(actions) * action_thresh
    action_offsets[actions < 0] = -action_offsets[actions < 0]
    action_offsets[actions == 0] = 0
    actions = (actions - np.array(action_offsets)) / (1.0 - action_thresh)

    return actions


class SpotBaseEnv(SpotRobotSubscriberMixin, gym.Env):
    node_name = "spot_reality_gym"
    no_raw = True
    proprioception = True

    def __init__(
        self,
        config,
        spot: Spot,
    ):
        self.detections_buffer = {
            k: FixSizeOrderedDict(maxlen=DETECTIONS_BUFFER_LEN)
            for k in ["filtered_hand_rgb", "filtered_hand_depth", "viz"]
        }
        super().__init__(spot=spot)
        self.config = config
        self.spot = spot
        self._max_lin_dist_scale = self.config.MAX_LIN_DIST
        self._max_ang_dist_scale = self.config.MAX_ANG_DIST
        self._max_joint_movement_scale = self.config.MAX_JOINT_MOVEMENT
        self.ctrl_hz = self.config.CTRL_HZ
        self.max_episode_steps = self.config.MAX_EPISODE_STEPS
        self.arm_lower_limits = np.deg2rad(self.config.ARM_LOWER_LIMITS)
        self.arm_upper_limits = np.deg2rad(self.config.ARM_UPPER_LIMITS)
        self.prev_base_moved = False
        self.num_steps = 0
        self.should_end = False
        self.grasp_success = False

    def get_observations(self):
        raise NotImplementedError

    def get_success(self, observations):
        raise NotImplementedError

    def power_robot(self):
        self.spot.power_on()
        # self.say("Standing up")
        try:
            self.spot.undock()
        except:
            print("Undocking failed: just standing up instead...")
            self.spot.blocking_stand()

    def process_base_action(self, base_action):
        base_action = rescale_actions(base_action, silence_only=True)
        if np.count_nonzero(base_action) > 0:
            # Command velocities using the input action
            lin_dist, ang_dist = base_action

            # Scale the linear and angular velocities
            nav_velocity_scaling = rospy.get_param("nav_velocity_scaling", 1.0)
            lin_dist *= self._max_lin_dist_scale * nav_velocity_scaling
            ang_dist *= np.deg2rad(self._max_ang_dist_scale * nav_velocity_scaling)

            target_yaw = wrap_heading(self.yaw + ang_dist)
            # No horizontal velocity
            ctrl_period = 1 / self.ctrl_hz
            # Don't even bother moving if it's just for a bit of distance
            if abs(lin_dist) < 0.05 and abs(ang_dist) < np.deg2rad(3):
                base_action = None
                target_yaw = None
            else:
                base_action = [lin_dist / ctrl_period, 0, ang_dist / ctrl_period]
                self.prev_base_moved = True
        else:
            base_action = None
            self.prev_base_moved = False
        return base_action

    def process_arm_action(self, arm_action):
        # arm_action = rescale_actions(arm_action, action_thresh=0.000)
        arm_action *= self._max_joint_movement_scale
        arm_action = self.current_arm_pose + pad_action(arm_action)
        arm_action = np.clip(arm_action, self.arm_lower_limits, self.arm_upper_limits)
        return arm_action

    def pre_step(self, action_dict):
        # Update the action_dict with grasp and place flags
        base_action = action_dict.get("base_action", None)
        if base_action is not None:
            base_action = self.process_base_action(base_action)
        arm_action = action_dict.get("arm_action", None)
        arm_ee_action = action_dict.get("arm_ee_action", None)
        arm_action = self.process_arm_action(arm_action)

        grasp = action_dict.get("grasp", False)
        place = action_dict.get("place", False)

        return base_action, arm_action

    def post_step(self):
        observations = self.get_observations()
        self.num_steps += 1

        reward = None
        info = {"num_steps": self.num_steps}
        timeout = self.num_steps >= self.max_episode_steps
        done = timeout or self.get_success(observations) or self.should_end

        return observations, reward, done, info

    def step(
        self,
        action_dict: Dict[str, Any],
        nav_silence_only=True,
        disable_oa=None,
        travel_time_scale=1.0,
    ):
        assert self.reset_ran, ".reset() must be called first!"
        base_action, arm_action = self.pre_step()

        return self.post_step()

    @property
    def filtered_hand_depth(self):
        return self.msgs[rt.FILTERED_HAND_DEPTH]

    @property
    def filtered_head_depth(self):
        return self.msgs[rt.FILTERED_HEAD_DEPTH]

    @property
    def filtered_hand_rgb(self):
        return self.msgs[rt.HAND_RGB]

    def img_callback(self, topic, msg):
        super().img_callback(topic, msg)
        if topic == rt.HAND_RGB:
            self.detections_buffer["filtered_hand_rgb"][str(msg.header.stamp)] = msg
        if topic == rt.FILTERED_HAND_DEPTH:
            self.detections_buffer["filtered_hand_depth"][str(msg.header.stamp)] = msg

    def process_images(self, img):
        # Crop out black vertical bars on the left and right edges of aligned depth img
        img = img[:, LEFT_CROP:-RIGHT_CROP]
        img = cv2.resize(img, (NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_AREA)
        # img = img.reshape([*img.shape, 1])  # unsqueeze
        # img = np.float32(img) / 255.0

        return img

    def get_gripper_images(self, key="filtered_hand_rgb", save_image=False):
        if self.grasp_success:
            # Return blank images if the gripper is being blocked
            blank_img = np.zeros([NEW_HEIGHT, NEW_WIDTH, 1], dtype=np.float32)
            return blank_img, blank_img.copy()
        print('len images: ', len(self.detections_buffer[key]))
        timestamp, msg = self.detections_buffer[key].popitem(last=True)
        self.detections_buffer[key][str(timestamp)] = msg
        arm_img = self.msg_to_cv2(msg)
        if 'depth' in key:
            arm_img = self.process_images(arm_img)
        cv2.imwrite(f"imgs/{key}_before_{int(time.time()*10000)}.png", arm_img)

        return arm_img

    def get_arm_joints(self, joint_black_list=None):
        # Get proprioception inputs
        joint_black_list = (
            self.config.JOINT_BLACKLIST
            if joint_black_list is None
            else joint_black_list
        )
        joints = np.array(
            [
                j
                for idx, j in enumerate(self.current_arm_pose)
                if idx not in joint_black_list
            ],
            dtype=np.float32,
        )

        return joints

    def attempt_grasp(self, obj_center_pixel, graspmode="any"):
        print('trying to grasp!!')
        self.grasp_success = self.spot.grasp_hand_depth(
            obj_center_pixel,
            top_down_grasp=graspmode == "topdown",
            horizontal_grasp=graspmode == "side",
            timeout=10,
        )
        return
