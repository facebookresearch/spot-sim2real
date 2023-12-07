# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from collections import OrderedDict

import numpy as np
import torch
from gym import spaces
from gym.spaces import Dict as SpaceDict
from habitat_baselines.rl.ddppo.policy.resnet_policy import PointNavResNetPolicy
from habitat_baselines.rl.ppo.moe import NavGazeMixtureOfExpertsMask
from habitat_baselines.rl.ppo.policy import PointNavBaselinePolicy
from habitat_baselines.utils.common import GaussianNet, batch_obs
from torch import Size, Tensor


# Turn numpy observations into torch tensors for consumption by policy
def to_tensor(v):
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)


class CustomNormal(torch.distributions.normal.Normal):
    def sample(self, sample_shape: Size = torch.Size()) -> Tensor:  # noqa: B008
        return self.rsample(sample_shape)

    def log_probs(self, actions) -> Tensor:
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self) -> Tensor:
        return super().entropy().sum(-1, keepdim=True)


class RealPolicy:
    def __init__(
        self,
        checkpoint_path,
        observation_space,
        action_space,
        device,
        policy_class=PointNavBaselinePolicy,
        is_hab3_policy=False,
    ):
        print("Loading policy...")
        self.device = torch.device(device)
        if is_hab3_policy:
            # Hab3 policy loading
            # Load the policy using torch script
            self.policy = {}
            if self.device.type == "cuda":
                self.policy["net"] = torch.jit.load(checkpoint_path["net"]).cuda()
                self.policy["action_dist"] = torch.jit.load(
                    checkpoint_path["action_dis"]
                ).cuda()
                self.policy["std"] = torch.load(checkpoint_path["std"]).cuda()
            else:
                self.policy["net"] = torch.jit.load(checkpoint_path["net"])
                self.policy["action_dist"] = torch.jit.load(
                    checkpoint_path["action_dis"]
                )
                self.policy["std"] = torch.load(checkpoint_path["std"])
        else:
            if isinstance(checkpoint_path, str):
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
            else:
                checkpoint = checkpoint_path

        # Load the config
        if not is_hab3_policy:
            config = checkpoint["config"]

        """ Disable observation transforms for real world experiments """
        if not is_hab3_policy:
            config.defrost()
            config.RL.POLICY.OBS_TRANSFORMS.ENABLED_TRANSFORMS = []
            config.freeze()
            config.RL.POLICY["init"] = False

            self.policy = policy_class.from_config(
                config=config,
                observation_space=observation_space,
                action_space=action_space,
            )
            print("Actor-critic architecture:", self.policy)
            # Move it to the device
            self.policy.to(self.device)

            # Load trained weights into the policy
            self.policy.load_state_dict(
                {
                    k[len("actor_critic.") :]: v
                    for k, v in checkpoint["state_dict"].items()
                }
            )

        self.prev_actions = None
        self.test_recurrent_hidden_states = None
        self.not_done_masks = None
        if is_hab3_policy:
            self.config = None
        else:
            self.config = config
        self.num_actions = action_space.shape[0]
        self.reset_ran = False
        self.is_hab3_policy = is_hab3_policy
        print("Policy loaded.")

    def reset(self):
        self.reset_ran = True
        if not self.is_hab3_policy:
            self.test_recurrent_hidden_states = torch.zeros(
                1,  # The number of environments. Just one for real world.
                self.policy.net.num_recurrent_layers,
                self.config.RL.PPO.hidden_size,
                device=self.device,
            )
        else:
            self.test_recurrent_hidden_states = torch.zeros(
                1,  # The number of environments. Just one for real world.
                4,
                512,
                device=self.device,
            )

        # We start an episode with 'done' being True (0 for 'not_done')
        self.not_done_masks = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
        self.prev_actions = torch.zeros(1, self.num_actions, device=self.device)

    def get_action(self, mu_maybe_std, std):
        """The final transformation of the action given inputs"""
        mu_maybe_std = mu_maybe_std.float()
        mu = mu_maybe_std

        mu = torch.tanh(mu)
        std = torch.clamp(std, -5, 2)
        std = torch.exp(std)

        return CustomNormal(mu, std, validate_args=False)

    def act(self, observations):
        assert self.reset_ran, "You need to call .reset() on the policy first."
        batch = batch_obs([observations], device=self.device)
        with torch.no_grad():
            if not self.is_hab3_policy:
                _, actions, _, self.test_recurrent_hidden_states = self.policy.act(
                    batch,
                    self.test_recurrent_hidden_states,
                    self.prev_actions,
                    self.not_done_masks,
                    deterministic=True,
                    actions_only=True,
                )
            else:
                # Using torch script to save the model
                with torch.no_grad():
                    output_model = self.policy["net"](
                        batch,
                        self.test_recurrent_hidden_states,
                        self.prev_actions,
                        self.not_done_masks,
                    )

                features = output_model[0]
                self.test_recurrent_hidden_states = output_model[1]

                with torch.no_grad():
                    raw_actions = self.get_action(
                        self.policy["action_dist"](features), self.policy["std"]
                    )
                    actions = raw_actions.mean

        self.prev_actions.copy_(actions)
        self.not_done_masks = torch.ones(1, 1, dtype=torch.bool, device=self.device)

        # GPU/CPU torch tensor -> numpy
        actions = actions.squeeze().cpu().numpy()

        return actions


class GazePolicy(RealPolicy):
    def __init__(self, checkpoint_path, device):
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
        action_space = spaces.Box(-1.0, 1.0, (4,))
        super().__init__(
            checkpoint_path,
            observation_space,
            action_space,
            device,
        )


class MobileGazePolicy(RealPolicy):
    def __init__(self, checkpoint_path, device):
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
        action_space = spaces.Box(-1.0, 1.0, (7,))
        super().__init__(
            checkpoint_path,
            observation_space,
            action_space,
            device,
            is_hab3_policy=True,
        )


class PlacePolicy(RealPolicy):
    def __init__(self, checkpoint_path, device):
        observation_space = SpaceDict(
            {
                "joint": spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),
                "obj_start_sensor": spaces.Box(
                    low=0.0, high=1.0, shape=(3,), dtype=np.float32
                ),
            }
        )
        action_space = spaces.Box(-1.0, 1.0, (4,))
        super().__init__(checkpoint_path, observation_space, action_space, device)


class NavPolicy(RealPolicy):
    def __init__(self, checkpoint_path, device):
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
        action_space = spaces.Box(-1.0, 1.0, (2,))
        super().__init__(checkpoint_path, observation_space, action_space, device)


class SocialNavPolicy(RealPolicy):
    def __init__(self, checkpoint_path, device):
        observation_space = SpaceDict(
            {
                "articulated_agent_arm_depth": spaces.Box(
                    low=0.0, high=1.0, shape=(240, 228, 1), dtype=np.float32
                ),
                "spot_head_stereo_depth_sensor": spaces.Box(
                    low=0.0, high=1.0, shape=(240, 228, 1), dtype=np.float32
                ),
                "humanoid_detector_sensor": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(240, 228, 1),
                    dtype=np.float32,
                ),
            }
        )
        # Linear, angular, and horizontal velocity (in that order)
        action_space = spaces.Box(-1.0, 1.0, (2,))
        super().__init__(checkpoint_path, observation_space, action_space, device)


class MixerPolicy(RealPolicy):
    def __init__(
        self,
        mixer_checkpoint_path,
        nav_checkpoint_path,
        gaze_checkpoint_path,
        place_checkpoint_path,
        device,
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
        checkpoint = torch.load(mixer_checkpoint_path, map_location="cpu")
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
    # Loading Hab3 policy
    ckpt_dict = {}
    ckpt_dict[
        "net"
    ] = "/home/jimmytyyang/research/spot-sim2real/spot_rl_experiments/weights/mobile_gaze/mg3ns_4_latest_ts.pth"
    ckpt_dict[
        "action_dis"
    ] = "/home/jimmytyyang/research/spot-sim2real/spot_rl_experiments/weights/mobile_gaze/mg3ns_4_latest_ad_ts.pth"
    ckpt_dict[
        "std"
    ] = "/home/jimmytyyang/research/spot-sim2real/spot_rl_experiments/weights/mobile_gaze/mg3ns_4_latest_std.pth"
    mobile_gaze_policy = MobileGazePolicy(
        ckpt_dict,
        device="cpu",
    )
    mobile_gaze_policy.reset()
    observations = {
        "arm_depth_bbox_sensor": np.zeros([240, 228, 1], dtype=np.float32),
        "articulated_agent_arm_depth": np.zeros([240, 228, 1], dtype=np.float32),
        "joint": np.zeros(4, dtype=np.float32),
    }
    actions = mobile_gaze_policy.act(observations)
    print("actions:", actions)

    # Loading the old policy
    gaze_policy = GazePolicy(
        "/home/jimmytyyang/research/spot-sim2real/spot_rl_experiments/weights/final_paper/gaze_normal_32_seed100_1649708902_ckpt.38.pth",
        device="cpu",
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
