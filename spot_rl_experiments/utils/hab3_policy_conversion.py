# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This is the script that loads a pretrained policy from a checkpoint and convert it to torch script format

import sys
from collections import OrderedDict

import numpy as np
import torch
from gym import spaces
from gym.spaces import Dict as SpaceDict
from habitat_baselines.rl.ddppo.policy.resnet_policy import PointNavResNetPolicy
from habitat_baselines.utils.common import GaussianNet, batch_obs
from spot_rl.utils.utils import get_default_parser
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
    """CustomNoraml needed to load the policy"""

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
        policy_class,
    ):
        print("Loading policy...")
        self.device = torch.device(device)

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Load the config
        config = checkpoint["config"]
        self.config = config

        # Init the policy class
        self.policy = policy_class.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )

        print("Actor-critic architecture:", self.policy)
        # Move it to the device
        self.policy.to(self.device)

        # Load trained weights into the policy
        self.policy.load_state_dict(checkpoint["state_dict"])

        self.prev_actions = None
        self.test_recurrent_hidden_states = None
        self.not_done_masks = None
        self.config = config
        self.num_actions = action_space.shape[0]
        self.reset_ran = False
        print("Policy loaded.")

        self.ts_net = None
        self.ts_action_dis = None
        self.std = None

    def reset(self):
        self.reset_ran = True
        self.test_recurrent_hidden_states = torch.zeros(
            1,  # The number of environments. Just one for real world.
            self.policy.net.num_recurrent_layers,
            self.config.habitat_baselines.rl.ppo.hidden_size,
            device=self.device,
        )

        # We start an episode with 'done' being True (0 for 'not_done')
        self.not_done_masks = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
        self.prev_actions = torch.zeros(1, self.num_actions, device=self.device)

    def convert_from_hab3_via_torchscript(self, save_path, observations):
        """This is the function that converts hab3 trained policy using the torchscript"""
        # Using torch script to save the model
        batch = batch_obs([observations], device=self.device)
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
        torch.jit.save(traced_cell, save_path["net"])

        # Using torch script to save action distribution
        traced_cell_ad = torch.jit.trace(
            self.policy.action_distribution.mu_maybe_std,
            (torch.tensor(torch.zeros((1, 512)))),
            strict=False,
        )
        torch.jit.save(traced_cell_ad, save_path["action_distribution"])

        # Using torch script to save std
        torch.save(self.policy.action_distribution.std, save_path["std"])

        print("Save (1) net (2) action distribution (3) std")

        self.ts_net = torch.jit.load(save_path["net"])
        self.ts_action_dis = torch.jit.load(save_path["action_distribution"])
        self.std = torch.load(save_path["std"])
        print("Load the torchscrip policy successfully")

    def act(self, observations):
        """Use the noraml way to get action"""
        assert self.reset_ran, "You need to call .reset() on the policy first."
        batch = batch_obs([observations], device=self.device)
        with torch.no_grad():
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
        """Small wrapper to get the final action"""
        mu_maybe_std = mu_maybe_std.float()
        mu = mu_maybe_std

        mu = torch.tanh(mu)
        std = torch.clamp(std, -5, 2)
        std = torch.exp(std)

        return CustomNormal(mu, std, validate_args=False)

    def act_ts(self, observation):
        """Using torchscript to run the model"""
        assert self.ts_net is not None, "Please load the torchscript policy first!"

        # Using torch script to save the model
        batch = batch_obs([observations], device=self.device)
        with torch.no_grad():
            output_model = self.ts_net(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
            )

        features = output_model[0]
        rnn_hidden_states = output_model[1]

        with torch.no_grad():
            action = self.get_action(self.ts_action_dis(features), self.std)
            action = action.mean
        return action, rnn_hidden_states


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
            PointNavResNetPolicy,
        )


def parse_arguments(args=sys.argv[1:]):
    parser = get_default_parser()
    parser.add_argument(
        "-t", "--target-hab3-policy", type=str, help="name of the target hab3 policy"
    )
    parser.add_argument(
        "-n",
        "--net",
        type=str,
        help="where to save torch script file for the net",
    )
    parser.add_argument(
        "-a",
        "--action-distribution",
        type=str,
        help="where to save torch script file for the action distribution",
    )
    parser.add_argument(
        "-s",
        "--std",
        type=str,
        help="where to save torch script file for the std",
    )
    args = parser.parse_args(args=args)

    return args


if __name__ == "__main__":
    """Script for loading the hab3-trained policy and convert it into a torchscript file.
    To run this script, you have to in hab3 conda enviornment with a latest habitat-sim"""
    args = parse_arguments()
    # The save location of the policy
    save_path = {}
    # This should be your target hab3 policy
    save_path["target_hab3_policy"] = args.target_hab3_policy
    # The following should be the target save path in the local disk
    save_path["net"] = args.net
    save_path["action_distribution"] = args.action_distribution
    save_path["std"] = args.std

    # The originl hab3 trained policy
    mobile_gaze_policy = MobileGazePolicy(
        save_path["target_hab3_policy"],
        device="cpu",
    )
    mobile_gaze_policy.reset()

    # Get the observation space
    observations = {
        "arm_depth_bbox_sensor": np.zeros([240, 228, 1], dtype=np.float32),
        "articulated_agent_arm_depth": np.zeros([240, 228, 1], dtype=np.float32),
        "joint": np.zeros(4, dtype=np.float32),
    }

    # Noraml way in hab3 to get the action
    actions = mobile_gaze_policy.act(observations)
    recurrent_hidden_state = mobile_gaze_policy.test_recurrent_hidden_states

    print("actions:", actions)
    print("recurrent_hidden_state:", recurrent_hidden_state)

    print("Torch script method...")
    mobile_gaze_policy.convert_from_hab3_via_torchscript(save_path, observations)
    mobile_gaze_policy.reset()
    ts_output = mobile_gaze_policy.act_ts(observations)
    ts_actions = ts_output[0]
    ts_rnn_hidden_states = ts_output[1]

    print("actions_ts:", ts_actions)
    print("recurrent_hidden_state:", ts_rnn_hidden_states)

    print("Please make sure the two methods are the same!")
