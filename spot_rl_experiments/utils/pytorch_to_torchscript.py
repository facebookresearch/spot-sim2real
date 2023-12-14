# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This is the script that loads a pretrained policy from a checkpoint and convert it to torch script format

import argparse
import importlib
import sys
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
from gym import spaces
from gym.spaces import Dict as SpaceDict

# from habitat_baselines.rl.ddppo.policy.resnet_policy import PointNavResNetPolicy
from habitat_baselines.utils.common import batch_obs
from torch import Size, Tensor
from yacs.config import CfgNode as CN


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


class PolicyConverter:
    """
    This is PolicyConverter class which will load the given policyclass in Pytorch & convert to torchscript model
    checkpoint_path: weights checkpoint .pth file path
    observation_space
    action_space
    device: cpu/gpu
    policy_class: PolicyClass that needs to be initialized & converted
    """

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
        checkpoint = torch.load(checkpoint_path, map_location=str(self.device))

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

    def convert_from_pytorch_to_torchscript(self, save_path, observations):
        """This is the function that converts hab3 trained policy using the torchscript
        Uses torch.jit.trace method to trace the forward pass of the Pytorch model & saves the corresponding torchscript model at given save_path
        """
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
            (torch.tensor(torch.zeros((1, 512))).to(self.device)),
            strict=False,
        )
        torch.jit.save(traced_cell_ad, save_path["action_distribution"])

        # Using torch script to save std
        torch.save(self.policy.action_distribution.std, save_path["std"])

        self.ts_net = torch.jit.load(save_path["net"], map_location=str(self.device))
        self.ts_action_dis = torch.jit.load(
            save_path["action_distribution"], map_location=str(self.device)
        )
        self.std = torch.load(save_path["std"], map_location=str(self.device))
        print("Load the torchscript policy successfully")

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
            # print("original hidden state:", PolicyActionData.rnn_hidden_states)

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

    def act_ts(self, observations):
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


def convert_from_pytorch_to_torchscript(conversion_params_yaml_path: str):
    """
    Accepts path to params yaml file
    Converts given pytorch policy to torchscript
    Checks if both the models produces same output
    """
    config = CN()
    config.set_new_allowed(True)
    config.merge_from_file(conversion_params_yaml_path)

    # The save location of the policy
    save_path = {}
    # This should be your target hab3 policy
    save_path["target_hab3_policy"] = config.TARGET_HAB3_POLICY_PATH
    # The following should be the target save path in the local disk
    save_path["net"] = config.OUTPUT_NET_SAVE_PATH
    save_path["action_distribution"] = config.OUTPUT_ACTION_DST_SAVE_PATH
    save_path["std"] = config.OUTPUT_ACTION_STD_SAVE_PATH

    # import the Habitat Policy
    module_path: Any = str(config.MODEL_CLASS_NAME).split(".")
    policy_class_name, module_path = module_path[-1], ".".join(module_path[:-1])
    module = importlib.import_module(module_path)
    policy_class = getattr(module, policy_class_name)

    # create observations & observation space from OBSERVATIONS_DICT
    observation_space: dict = {}
    observations: dict = {}

    for key, shape in dict(config.OBSERVATIONS_DICT).items():
        # if normal case don't include spot_head_stereo_depth_sensor key
        if not config.USE_STEREO_PAIR_CAMERA and "stereo" in key:
            continue
        if (
            "spot_head_stereo_depth_sensor" in key
            or "articulated_agent_arm_depth" in key
        ):
            low, high = 0.0, 1.0
        else:
            low, high = np.finfo(np.float32).min, np.finfo(np.float32).max
        observation_space[key] = spaces.Box(
            low=low,
            high=high,
            shape=shape,
            dtype=np.float32,
        )
        observations[key] = np.zeros(shape, dtype=np.float32)
    # print({key:value.shape for key, value in observations.items()})
    observation_space = SpaceDict(observation_space)
    action_space = spaces.Box(-1.0, 1.0, (config.ACTION_SPACE_LENGTH,))

    # The originl hab3 trained policy initialized in Pytorch
    pytorchpolicyconverter = PolicyConverter(
        checkpoint_path=save_path["target_hab3_policy"],
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        policy_class=policy_class,
    )
    pytorchpolicyconverter.reset()

    # Normal way in hab3 to get the action
    # save Pytorch actions & reccurent hidden states to compare with torchscript actions & hidden states
    actions_pytorch = pytorchpolicyconverter.act(observations)
    recurrent_hidden_state_pytorch = pytorchpolicyconverter.test_recurrent_hidden_states

    print("actions from pytorch model:", actions_pytorch)
    print("recurrent_hidden_state from pytorch model:", recurrent_hidden_state_pytorch)

    # Convert to torchscript model
    pytorchpolicyconverter.convert_from_pytorch_to_torchscript(save_path, observations)
    pytorchpolicyconverter.reset()

    # Extract actions & hidden states from the torchscript model
    (
        actions_torchscript,
        recurrent_hidden_state_torchscript,
    ) = pytorchpolicyconverter.act_ts(observations)
    print("actions from torchscript model:", actions_torchscript)
    print(
        "recurrent_hidden_state from torchscript model:",
        recurrent_hidden_state_torchscript,
    )
    # Compare actions & rnn states from pytorch & torchscript model they should be pretty close
    is_same_actions = torch.allclose(
        torch.from_numpy(actions_pytorch).to(device), actions_torchscript
    )
    is_same_rnn_states = torch.allclose(
        recurrent_hidden_state_pytorch, recurrent_hidden_state_torchscript
    )
    print(
        f"Are actions from Pytorch & torchscript model same ? {is_same_actions}, are recurrent hidden states same ? {is_same_rnn_states}",
    )
    assert (
        is_same_actions and is_same_rnn_states
    ), f"Actions & RNN states were supposed to be same but found actions to be same? {is_same_actions} & rnn states to be same?{is_same_rnn_states}"
    print(
        f"Torchscipt net model saved at {save_path['net']}, action distribution saved at {save_path['action_distribution']} and std saved at {save_path['std']}"
    )


def parse_arguments(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--conversion_params_yaml_path",
        type=str,
        required=True,
        help="Path to conversion parameters.yaml file",
    )
    args = parser.parse_args(args=args)
    return args


if __name__ == "__main__":
    """Script for loading the hab3-trained policy and convert it into a torchscript file.
    To run this script, you have to in hab3 conda enviornment with a latest habitat-sim"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    args = parse_arguments()
    convert_from_pytorch_to_torchscript(args.conversion_params_yaml_path)
