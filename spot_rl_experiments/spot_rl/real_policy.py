# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from collections import OrderedDict

import numpy as np
import torch
from gym import spaces
from gym.spaces import Dict as SpaceDict
from torch import Size, Tensor

try:
    # The origin spot sim2real env
    from habitat_baselines.rl.ppo.moe import NavGazeMixtureOfExpertsMask
    from habitat_baselines.rl.ppo.policy import PointNavBaselinePolicy

except Exception:
    # Based on the new habitat
    import sys

    sys.path.append("/Users/jimmytyyang/research/habitat-lab/habitat-lab")
    sys.path.append("/Users/jimmytyyang/research/habitat-lab/habitat-baselines")
    from habitat_baselines.rl.ddppo.policy.resnet_policy import PointNavResNetPolicy
    from habitat_baselines.utils.common import GaussianNet, batch_obs


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
        policy_class=None,  # PointNavBaselinePolicy,
        config_path=None,
    ):
        print("Loading policy...")
        self.device = torch.device(device)
        if isinstance(checkpoint_path, str):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        else:
            checkpoint = checkpoint_path

        # Load the config
        if config_path is not None:
            config = torch.load(config_path, map_location="cpu")
            config = config["config"]
        else:
            config = checkpoint["config"]

        self.config = config

        """ Disable observation transforms for real world experiments """
        try:
            config.defrost()
            config.RL.POLICY.OBS_TRANSFORMS.ENABLED_TRANSFORMS = []
            config.freeze()
            config.RL.POLICY["init"] = False
        except Exception:
            print("Does not support the old config")

        self.policy = policy_class.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        print("Actor-critic architecture:", self.policy)
        # Move it to the device
        self.policy.to(self.device)

        # Load trained weights into the policy
        try:
            self.policy.load_state_dict(
                {
                    k[len("actor_critic.") :]: v
                    for k, v in checkpoint["state_dict"].items()
                }
            )
        except Exception:
            self.policy.load_state_dict(checkpoint["state_dict"])

        self.prev_actions = None
        self.test_recurrent_hidden_states = None
        self.not_done_masks = None
        self.config = config
        self.num_actions = action_space.shape[0]
        self.reset_ran = False
        print("Policy loaded.")

    def reset(self):
        self.reset_ran = True
        try:
            self.test_recurrent_hidden_states = torch.zeros(
                1,  # The number of environments. Just one for real world.
                self.policy.net.num_recurrent_layers,
                self.config.RL.PPO.hidden_size,
                device=self.device,
            )
        except Exception:
            self.test_recurrent_hidden_states = torch.zeros(
                1,  # The number of environments. Just one for real world.
                self.policy.net.num_recurrent_layers,
                self.config.habitat_baselines.rl.ppo.hidden_size,
                device=self.device,
            )

        # We start an episode with 'done' being True (0 for 'not_done')
        self.not_done_masks = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
        self.prev_actions = torch.zeros(1, self.num_actions, device=self.device)

    def act(self, observations):
        assert self.reset_ran, "You need to call .reset() on the policy first."
        batch = batch_obs([observations], device=self.device)
        with torch.no_grad():
            try:
                _, actions, _, self.test_recurrent_hidden_states = self.policy.act(
                    batch,
                    self.test_recurrent_hidden_states,
                    self.prev_actions,
                    self.not_done_masks,
                    deterministic=True,
                    actions_only=True,
                )
            except Exception:

                # Using torch script to save the model
                traced_cell = torch.jit.trace(
                    self.policy.net,
                    (
                        batch,
                        self.test_recurrent_hidden_states,
                        self.prev_actions,
                        self.not_done_masks,
                    ),
                    strict=False,
                )
                torch.jit.save(
                    traced_cell,
                    "/Users/jimmytyyang/Downloads/mobile_gaze_ckpt_1101/mg3ns_4_latest_ts.pth",
                )
                # Using torch script to save action distribution
                traced_cell_ad = torch.jit.trace(
                    self.policy.action_distribution.mu_maybe_std,
                    (torch.tensor(torch.zeros((1, 512)))),
                    strict=False,
                )
                torch.jit.save(
                    traced_cell_ad,
                    "/Users/jimmytyyang/Downloads/mobile_gaze_ckpt_1101/mg3ns_4_latest_ad_ts.pth",
                )
                # Using torch script to save std
                torch.save(
                    self.policy.action_distribution.std,
                    "/Users/jimmytyyang/Downloads/mobile_gaze_ckpt_1101/mg3ns_4_latest_std.pth",
                )

                PolicyActionData = self.policy.act(
                    batch,
                    self.test_recurrent_hidden_states,
                    self.prev_actions,
                    self.not_done_masks,
                    deterministic=True,
                )
                actions = PolicyActionData.actions
                self.test_recurrent_hidden_states = PolicyActionData.rnn_hidden_states
                print("original hidden state:", PolicyActionData.rnn_hidden_states)

        self.prev_actions.copy_(actions)
        self.not_done_masks = torch.ones(1, 1, dtype=torch.bool, device=self.device)

        # GPU/CPU torch tensor -> numpy
        actions = actions.squeeze().cpu().numpy()

        return actions

    def get_action(self, mu_maybe_std, std):
        mu_maybe_std = mu_maybe_std.float()
        mu = mu_maybe_std

        mu = torch.tanh(mu)
        std = torch.clamp(std, -5, 2)
        std = torch.exp(std)

        return CustomNormal(mu, std, validate_args=False)

    def act_ts(self, observation):
        """Using torchscript to run the model"""
        # Using torch script to save the model
        batch = batch_obs([observations], device=self.device)
        loaded_trace = torch.jit.load(
            "/Users/jimmytyyang/Downloads/mobile_gaze_ckpt_1101/mg3ns_4_latest_ts.pth"
        )
        loaded_trace_ad = torch.jit.load(
            "/Users/jimmytyyang/Downloads/mobile_gaze_ckpt_1101/mg3ns_4_latest_ad_ts.pth"
        )
        load_std = torch.load(
            "/Users/jimmytyyang/Downloads/mobile_gaze_ckpt_1101/mg3ns_4_latest_std.pth"
        )
        with torch.no_grad():
            output_model = loaded_trace(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
            )

        features = output_model[0]
        rnn_hidden_states = output_model[1]

        with torch.no_grad():
            features_ad = loaded_trace_ad(features)

            action = self.get_action(features_ad, load_std)
            action = action.mean

        # action_distribution = GaussianNet(512,7, self.config["habitat_baselines"]["rl"]["policy"]["main_agent"]["action_dist"],)
        # distribution = action_distribution(features)
        # action = distribution.mean
        # self.policy.action_distribution(features).mean
        return action, rnn_hidden_states


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
            PointNavBaselinePolicy,
        )


class MobileGazePolicy(RealPolicy):
    def __init__(self, checkpoint_path, config_path, device):
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
            PointNavResNetPolicy,
            config_path,
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
    mobile_gaze_policy = MobileGazePolicy(
        "/Users/jimmytyyang/Downloads/mobile_gaze_ckpt_1101/mg3ns_4_latest_only_state_dict.pth",
        "/Users/jimmytyyang/Downloads/mobile_gaze_ckpt_1101/mg3ns_4_latest_only_config.pth",
        device="cpu",
    )
    mobile_gaze_policy.reset()
    observations = {
        "arm_depth_bbox_sensor": np.zeros([240, 228, 1], dtype=np.float32),
        "articulated_agent_arm_depth": np.zeros([240, 228, 1], dtype=np.float32),
        "joint": np.zeros(4, dtype=np.float32),
    }
    actions = mobile_gaze_policy.act(observations)
    original_recurrent_hidden_state = mobile_gaze_policy.test_recurrent_hidden_states

    print("actions:", actions)
    print("Torch script method")

    mobile_gaze_policy.reset()
    ts_output = mobile_gaze_policy.act_ts(observations)
    actions_ts = ts_output[0]
    rnn_hidden_states = ts_output[1]

    print("action:", actions_ts)

    print("=====Comparsion=====")
    print(
        torch.sum(original_recurrent_hidden_state - rnn_hidden_states),
        "<-should be zero!",
    )
    print(torch.sum(abs(actions_ts - actions)), "<-should be zero!")
    #  out, rnn_hidden_states, aux_loss_state
    # from habitat_baselines.utils.common import GaussianNet
    # self.action_distribution = GaussianNet(
    #     512, #self.net.output_size,
    #     7,
    #     config["habitat_baselines"]["rl"]["policy"]["main_agent"]["action_dist"],
    # )

    # gaze_policy = GazePolicy(
    #     "weights/bbox_mask_5thresh_autograsp_shortrange_seed1_36.pth",
    #     device="cpu",
    # )
    # gaze_policy.reset()
    # observations = {
    #     "arm_depth": np.zeros([240, 320, 1], dtype=np.float32),
    #     "arm_depth_bbox": np.zeros([240, 320, 1], dtype=np.float32),
    #     "joint": np.zeros(4, dtype=np.float32),
    #     "is_holding": np.zeros(1, dtype=np.float32),
    # }
    # actions = gaze_policy.act(observations)
    # print("actions:", actions)
