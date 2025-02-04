# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json

# mypy: ignore-errors
import os
import os.path as osp

import cv2
import gym
import numpy as np
from spot_rl.utils.robot_subscriber import SpotRobotSubscriberMixin
from spot_rl.utils.utils import FixSizeOrderedDict, arr2str, object_id_to_object_name
from spot_rl.utils.utils import ros_topics as rt
from spot_wrapper.spot import Spot

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
            for k in ["detections", "filtered_depth", "viz"]
        }
        super().__init__(spot=spot)
        self.config = config
        self.spot = spot

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

    def process_images(self, img):
        # Crop out black vertical bars on the left and right edges of aligned depth img
        img = img[:, LEFT_CROP:-RIGHT_CROP]
        img = cv2.resize(img, (NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_AREA)
        img = img.reshape([*img.shape, 1])  # unsqueeze
        # img = np.float32(img) / 255.0

        return img

    def get_gripper_images(self, save_image=False):
        if self.grasp_attempted:
            # Return blank images if the gripper is being blocked
            blank_img = np.zeros([NEW_HEIGHT, NEW_WIDTH, 1], dtype=np.float32)
            return blank_img, blank_img.copy()
        arm_rgb = self.msg_to_cv2(self.detections_buffer["filtered_hand_rgb"][-1])
        arm_rgb = self.process_images(arm_rgb)

        return arm_rgb

    def get_arm_joints(self, joint_black_list=None):
        """Get the current arm joints. If it is semantic place skills,
        we will return one addition joints"""
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
