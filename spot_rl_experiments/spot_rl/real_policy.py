# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import pickle
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
from gym import spaces
from gym.spaces import Dict as SpaceDict
from habitat_baselines.rl.ddppo.policy.resnet_policy import PointNavResNetPolicy
from habitat_baselines.rl.ppo.moe import NavGazeMixtureOfExpertsMask
from habitat_baselines.rl.ppo.policy import PointNavBaselinePolicy
from habitat_baselines.utils.common import GaussianNet, batch_obs
from spot_rl.utils.construct_configs import construct_config
from torch import Size, Tensor
from yacs.config import CfgNode as CN


class RealPolicy:
    """
    RealPolicy is a baseclass for all the Policies (GazePolicy, NavPolicy, PlacePolicy, MobileGazePolicy, etc)
    Inputs :
    checkpoint_path: Str or Dict of strs denoting path to pytorch weights or torchscript model path
    observation_space: required arg for initializing underlying PolicyClass (required for non torchscript models)
    action_space: required arg for initializing underlying PolicyClass (required for non torchscript models)
    device: cpu or gpu
    policy_class: Name of the underlying PolicyClass, default=PointNavBaselinePolicy
    is_torchscript_policy: Boolean, True/False, deafult=False, Flags denote whether the undelying policy will be loaded as torchscript model or normal Pytorch class,
    More info on spot_rl_experiments/utils/README.md
    """

    def __init__(
        self,
        checkpoint_path,
        observation_space,
        action_space,
        device,
        policy_class=PointNavBaselinePolicy,
        config=CN(),
    ):
        print("Loading policy...")
        self.device = torch.device(device)
        # config = construct_config()

        self.is_torchscript_policy = False

        if (
            "torchscript" in str(config.WEIGHTS_TYPE).lower()
            and type(checkpoint_path) == str
        ):
            # map pytorch weight path to torchscript path based on basename
            checkpoint_path_without_ext = os.path.basename(checkpoint_path).split(".")[
                0
            ]
            checkpoint_path_ts = [
                torchscript_checkpoint_path
                for torchscript_checkpoint_path in list(
                    config.WEIGHTS_TORCHSCRIPT.values()
                )
                if checkpoint_path_without_ext in torchscript_checkpoint_path
            ]
            checkpoint_path_prefix = checkpoint_path.split("weights")[0]
            checkpoint_path = (
                os.path.join(checkpoint_path_prefix, checkpoint_path_ts[0])
                if len(checkpoint_path_ts) > 0
                else checkpoint_path
            )
            self.is_torchscript_policy = "torchscript" in checkpoint_path
        # print(policy_class, self.is_torchscript_policy)
        if self.is_torchscript_policy:
            # Hab3 policy loading
            # Load the policy using torch script
            extra_files = {"net_meta_dict.pkl": ""}
            self.policy = torch.jit.load(
                checkpoint_path, _extra_files=extra_files, map_location=self.device
            )
            self.network_meta_attributes = pickle.loads(
                extra_files["net_meta_dict.pkl"]
            )
            print(
                f"Extra files content loaded with torchscript_model {self.network_meta_attributes}"
            )
        else:
            if isinstance(checkpoint_path, str):
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            else:
                checkpoint = checkpoint_path

            # Load the config
            config = checkpoint["config"]

            """ Disable observation transforms for real world experiments """
            config.defrost()
            config.RL.POLICY.OBS_TRANSFORMS.ENABLED_TRANSFORMS = []
            config.freeze()
            config.RL.POLICY["init"] = False

            # Load the policy using policy class
            self.policy = policy_class.from_config(
                config=config,
                observation_space=observation_space,
                action_space=action_space,
            )
            # Move it to the device
            self.policy.to(self.device)

            # Load trained weights into the policy
            self.policy.load_state_dict(
                {
                    k[len("actor_critic.") :]: v
                    for k, v in checkpoint["state_dict"].items()
                }
            )
            self.config = config

        print("Loaded Actor-critic architecture")
        self.prev_actions = None
        self.test_recurrent_hidden_states = None
        self.not_done_masks = None

        self.num_actions = action_space.shape[0]
        self.reset_ran = False
        print("Policy loaded.")

    def reset(self):
        self.reset_ran = True
        if self.is_torchscript_policy:
            self.test_recurrent_hidden_states = torch.zeros(
                1,  # The number of environments. Just one for real world.
                self.network_meta_attributes.get("num_recurrent_layers", 4),
                self.network_meta_attributes.get("RL_PPO_HIDDEN_SIZE", 512),
                device=self.device,
            )
        else:
            self.test_recurrent_hidden_states = torch.zeros(
                1,  # The number of environments. Just one for real world.
                self.policy.net.num_recurrent_layers,
                self.config.RL.PPO.hidden_size,
                device=self.device,
            )

        # We start an episode with 'done' being True (0 for 'not_done')
        self.not_done_masks = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
        self.prev_actions = torch.zeros(1, self.num_actions, device=self.device)

    def act(self, observations):
        assert self.reset_ran, "You need to call .reset() on the policy first."
        batch = batch_obs([observations], device=self.device)
        with torch.no_grad():
            if self.is_torchscript_policy:
                # Using torch script to load the model
                (actions, self.test_recurrent_hidden_states) = self.policy(
                    batch,
                    self.test_recurrent_hidden_states,
                    self.prev_actions,
                    self.not_done_masks,
                )
            else:
                _, actions, _, self.test_recurrent_hidden_states = self.policy.act(
                    batch,
                    self.test_recurrent_hidden_states,
                    self.prev_actions,
                    self.not_done_masks,
                    deterministic=True,
                    actions_only=True,
                )

        self.prev_actions.copy_(actions)
        self.not_done_masks = torch.ones(1, 1, dtype=torch.bool, device=self.device)

        # GPU/CPU torch tensor -> numpy
        actions = actions.squeeze().cpu().numpy()

        return actions


class GazePolicy(RealPolicy):
    def __init__(self, checkpoint_path, device, config: CN = CN()):
        observation_space = SpaceDict(
            {
                "arm_depth": spaces.Box(
                    low=0.0, high=1.0, shape=(240, 228, 1), dtype=np.float32
                ),
                "arm_depth_bbox": spaces.Box(
                    low=0.0, high=1.0, shape=(240, 228, 1), dtype=np.float32
                ),
                "joint": spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),
                "is_holding": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
            }
        )
        action_space = spaces.Box(
            -1.0, 1.0, (config.get("GAZE_ACTION_SPACE_LENGTH", 4),)
        )
        super().__init__(
            checkpoint_path, observation_space, action_space, device, config=config
        )


class MobileGazePolicy(RealPolicy):
    def __init__(self, checkpoint_path, device, config: CN = CN()):
        observation_space = SpaceDict(
            {
                "arm_depth_bbox_sensor": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(240, 228, 1),
                    dtype=np.float32,
                ),
                "articulated_agent_arm_depth": spaces.Box(
                    low=0.0, high=1.0, shape=(240, 228, 1), dtype=np.float32
                ),
                "joint": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(4,),
                    dtype=np.float32,
                ),
            }
        )
        action_space = spaces.Box(
            -1.0, 1.0, (config.get("MOBILE_GAZE_ACTION_SPACE_LENGTH", 7),)
        )
        super().__init__(
            checkpoint_path, observation_space, action_space, device, config=config
        )


class MobileGazeEEPolicy(RealPolicy):
    def __init__(self, checkpoint_path, device, config: CN = CN()):

        observation_space = SpaceDict(
            {
                "arm_depth_bbox_sensor": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(240, 228, 1),
                    dtype=np.float32,
                ),
                "articulated_agent_arm_depth": spaces.Box(
                    low=0.0, high=1.0, shape=(240, 228, 1), dtype=np.float32
                ),
                "ee_pose": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(6,),
                    dtype=np.float32,
                ),
            }
        )
        action_space = spaces.Box(-1.0, 1.0, (9,))
        super().__init__(
            checkpoint_path, observation_space, action_space, device, config=config
        )


class SemanticGazePolicy(RealPolicy):
    def __init__(self, checkpoint_path, device, config: CN = CN()):
        observation_space = SpaceDict(
            {
                "arm_depth_bbox_sensor": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(240, 228, 1),
                    dtype=np.float32,
                ),
                "articulated_agent_arm_depth": spaces.Box(
                    low=0.0, high=1.0, shape=(240, 228, 1), dtype=np.float32
                ),
                "topdown_or_side_grasping": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "joint": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(4,),
                    dtype=np.float32,
                ),
            }
        )
        action_space = spaces.Box(
            -1.0, 1.0, (config.get("MOBILE_GAZE_ACTION_SPACE_LENGTH", 7),)
        )
        super().__init__(
            checkpoint_path, observation_space, action_space, device, config=config
        )


class PlacePolicy(RealPolicy):
    def __init__(self, checkpoint_path, device, config: CN = CN()):
        observation_space = SpaceDict(
            {
                "joint": spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),
                "obj_start_sensor": spaces.Box(
                    low=0.0, high=1.0, shape=(3,), dtype=np.float32
                ),
            }
        )
        action_space = spaces.Box(
            -1.0, 1.0, (config.get("PLACE_ACTION_SPACE_LENGTH", 4),)
        )
        super().__init__(
            checkpoint_path, observation_space, action_space, device, config=config
        )


class SemanticPlacePolicy(RealPolicy):
    def __init__(self, checkpoint_path, device, config: CN = CN()):
        observation_space = SpaceDict(
            {
                "obj_goal_sensor": spaces.Box(
                    shape=[
                        3,
                    ],
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                ),
                "relative_initial_ee_orientation": spaces.Box(
                    shape=[
                        1,
                    ],
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                ),
                "relative_target_object_orientation": spaces.Box(
                    shape=[
                        1,
                    ],
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                ),
                "articulated_agent_jaw_depth": spaces.Box(
                    shape=[240, 228, 1], low=0.0, high=1.0, dtype=np.float32
                ),
                "joint": spaces.Box(
                    shape=[
                        5,
                    ],
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                ),
                "is_holding": spaces.Box(
                    shape=[
                        1,
                    ],
                    low=0,
                    high=1,
                    dtype=np.float32,
                ),
            }
        )
        action_space = spaces.Box(
            -1.0, 1.0, (config.get("SEMANTIC_PLACE_ACTION_SPACE_LENGTH", 9),)
        )
        super().__init__(
            checkpoint_path, observation_space, action_space, device, config=config
        )


class SemanticPlaceEEPolicy(RealPolicy):
    def __init__(self, checkpoint_path, device, config: CN = CN()):
        observation_space = SpaceDict(
            {
                "obj_goal_sensor": spaces.Box(
                    shape=[
                        3,
                    ],
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                ),
                "relative_initial_ee_orientation": spaces.Box(
                    shape=[
                        1,
                    ],
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                ),
                "relative_target_object_orientation": spaces.Box(
                    shape=[
                        1,
                    ],
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                ),
                "articulated_agent_jaw_depth": spaces.Box(
                    shape=[240, 228, 1], low=0.0, high=1.0, dtype=np.float32
                ),
                "ee_pose": spaces.Box(
                    shape=[
                        6,
                    ],
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                ),
                "is_holding": spaces.Box(
                    shape=[
                        1,
                    ],
                    low=0,
                    high=1,
                    dtype=np.float32,
                ),
            }
        )
        action_space = spaces.Box(
            -1.0, 1.0, (config.get("SEMANTIC_PLACE_EE_ACTION_SPACE_LENGTH", 10),)
        )
        super().__init__(
            checkpoint_path, observation_space, action_space, device, config=config
        )


class NavPolicy(RealPolicy):
    def __init__(self, checkpoint_path, device, config: CN = CN()):
        observation_space = SpaceDict(
            {
                "spot_left_depth": spaces.Box(
                    low=0.0, high=1.0, shape=(212, 120, 1), dtype=np.float32
                ),
                "spot_right_depth": spaces.Box(
                    low=0.0, high=1.0, shape=(212, 120, 1), dtype=np.float32
                ),
                "goal_heading": spaces.Box(
                    low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32
                ),
                "target_point_goal_gps_and_compass_sensor": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )
        # Linear, angular, and horizontal velocity (in that order)
        action_space = spaces.Box(
            -1.0, 1.0, (config.get("NAV_ACTION_SPACE_LENGTH", 2),)
        )
        super().__init__(
            checkpoint_path, observation_space, action_space, device, config=config
        )


class OpenCloseDrawerPolicy(RealPolicy):
    def __init__(self, checkpoint_path, device, config: CN = CN()):
        observation_space = SpaceDict(
            {
                "handle_bbox": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(240, 228, 1),
                    dtype=np.float32,
                ),
                "articulated_agent_arm_depth": spaces.Box(
                    low=0.0, high=1.0, shape=(240, 228, 1), dtype=np.float32
                ),
                "joint": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(5,),
                    dtype=np.float32,
                ),
                "ee_pos": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(3,),
                    dtype=np.float32,
                ),
                "art_pose_delta_sensor": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "is_holding": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
            }
        )
        action_space = spaces.Box(
            -1.0, 1.0, (config.get("OPEN_CLOSE_DRAWER_ACTION_SPACE_LENGTH", 8),)
        )
        super().__init__(
            checkpoint_path, observation_space, action_space, device, config=config
        )


class MixerPolicy(RealPolicy):
    def __init__(
        self,
        mixer_checkpoint_path,
        nav_checkpoint_path,
        gaze_checkpoint_path,
        place_checkpoint_path,
        device,
        config,
    ):
        observation_space = SpaceDict(
            {
                "spot_left_depth": spaces.Box(
                    low=0.0, high=1.0, shape=(212, 120, 1), dtype=np.float32
                ),
                "spot_right_depth": spaces.Box(
                    low=0.0, high=1.0, shape=(212, 120, 1), dtype=np.float32
                ),
                "goal_heading": spaces.Box(
                    low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32
                ),
                "target_point_goal_gps_and_compass_sensor": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(2,),
                    dtype=np.float32,
                ),
                "arm_depth": spaces.Box(
                    low=0.0, high=1.0, shape=(240, 228, 1), dtype=np.float32
                ),
                "arm_depth_bbox": spaces.Box(
                    low=0.0, high=1.0, shape=(240, 228, 1), dtype=np.float32
                ),
                "joint": spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),
                "is_holding": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "obj_start_sensor": spaces.Box(
                    low=0.0, high=1.0, shape=(3,), dtype=np.float32
                ),
                "visual_features": spaces.Box(
                    low=0.0, high=1.0, shape=(1024,), dtype=np.float32
                ),
            }
        )
        checkpoint = torch.load(mixer_checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint["config"].RL.POLICY["nav_checkpoint_path"] = nav_checkpoint_path
        checkpoint["config"].RL.POLICY["gaze_checkpoint_path"] = gaze_checkpoint_path
        checkpoint["config"].RL.POLICY["place_checkpoint_path"] = place_checkpoint_path
        # checkpoint["config"].RL.POLICY["use_residuals"] = False
        checkpoint["config"]["NUM_ENVIRONMENTS"] = 1
        action_space = spaces.Box(-1.0, 1.0, (6 + 3,))
        super().__init__(
            checkpoint,
            observation_space,
            action_space,
            device,
            policy_class=NavGazeMixtureOfExpertsMask,
            config=config,
        )
        self.not_done = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
        self.moe_actions = None
        self.policy.deterministic_nav = True
        self.policy.deterministic_gaze = True
        self.policy.deterministic_place = True
        self.nav_silence_only = True
        self.test_recurrent_hidden_states = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            1,
            512 * 3,
            device=self.device,
        )

    def reset(self):
        self.not_done = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
        self.test_recurrent_hidden_states = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            1,
            512 * 3,
            device=self.device,
        )

    def act(self, observations, expert=None):
        transformed_obs = self.policy.transform_obs([observations], self.not_done)
        batch = batch_obs(transformed_obs, device=self.device)
        with torch.no_grad():
            _, actions, _, self.test_recurrent_hidden_states = self.policy.act(
                batch,
                self.test_recurrent_hidden_states,
                None,
                self.not_done,
                deterministic=False,
                # deterministic=True,
                actions_only=True,
            )

        # GPU/CPU torch tensor -> numpy
        self.not_done = torch.ones(1, 1, dtype=torch.bool, device=self.device)
        actions = actions.squeeze().cpu().numpy()

        activated_experts = []
        corrective_actions = OrderedDict()
        corrective_actions["arm"] = actions[:4]
        corrective_actions["base"] = actions[4:6]
        if actions[-3] > 0:
            activated_experts.append("nav")
            corrective_actions.pop("base")
            self.nav_silence_only = True
        else:
            self.nav_silence_only = False
        if actions[-2] > 0:
            activated_experts.append("gaze")
            corrective_actions.pop("arm")
        if actions[-1] > 0:
            activated_experts.append("place")
            corrective_actions.pop("arm")
        corrective_actions_list = []
        for v in corrective_actions.values():
            for vv in v:
                corrective_actions_list.append(f"{vv:.3f}")
        print(
            f"gater: {', '.join(activated_experts)}\t"
            f"corrective: {', '.join(corrective_actions_list)}"
        )

        self.moe_actions = actions
        action_dict = self.policy.action_to_dict(actions, 0, use_residuals=False)
        step_action = action_dict["action"]["action"].numpy()
        arm_action, base_action = np.split(step_action, [4])

        return base_action, arm_action


if __name__ == "__main__":
    config = construct_config()
    gaze_policy = GazePolicy(
        config.WEIGHTS.GAZE,
        device="cuda",
    )
    gaze_policy.reset()
    observations = {
        "arm_depth": np.zeros([240, 320, 1], dtype=np.float32),
        "arm_depth_bbox": np.zeros([240, 320, 1], dtype=np.float32),
        "joint": np.zeros(4, dtype=np.float32),
        "is_holding": np.zeros(1, dtype=np.float32),
    }
    actions = gaze_policy.act(observations)
    print("actions:", actions)
