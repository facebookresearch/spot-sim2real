# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Code inspired from https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html
# This is the script that loads a pretrained policy from a checkpoint and convert it to torch script format

import argparse
import importlib
import os
import pickle
import sys
from collections import OrderedDict
from typing import Any, Tuple

import numpy as np
import torch
from gym import spaces
from gym.spaces import Dict as SpaceDict
from habitat_baselines.common.tensor_dict import TensorDict

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


class FinalTorchscriptModel(torch.nn.Module):
    def __init__(
        self,
        net_ts,
        action_distribution_ts,
        NEW_HABITAT_LAB_POLICY_OR_OLD,
        std,
        min_log_std,
        max_log_std,
    ):
        super(FinalTorchscriptModel, self).__init__()
        self.net_ts = net_ts
        self.action_dist_ts = action_distribution_ts
        self.NEW_HABITAT_LAB_POLICY_OR_OLD = NEW_HABITAT_LAB_POLICY_OR_OLD
        self.std = std
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def get_action(self, mu_maybe_std, std):
        """Small wrapper to get the final action"""
        if self.NEW_HABITAT_LAB_POLICY_OR_OLD == "old":
            return CustomNormal(*mu_maybe_std, validate_args=False)
        mu_maybe_std = mu_maybe_std.float()
        mu = mu_maybe_std
        mu = torch.tanh(mu)
        std = torch.clamp(std, self.min_log_std, self.max_log_std)
        std = torch.exp(std)
        return CustomNormal(mu, std, validate_args=False)

    def forward(
        self,
        batch_obs: TensorDict,
        test_recurrent_hidden_states: Tensor,
        prev_actions: Tensor,
        not_done_masks: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        output_of_model = self.net_ts(
            batch_obs,
            test_recurrent_hidden_states,
            prev_actions,
            not_done_masks,
        )
        features, rnn_hidden_states = output_of_model[0], output_of_model[1]
        action = self.get_action(self.action_dist_ts(features), self.std)
        action = action.mean
        return action, rnn_hidden_states


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
        conversion_params,
    ):
        print("Loading policy...")
        self.device = torch.device(device)

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Load the config
        config = checkpoint["config"]
        self.config = config
        self.conversion_params = conversion_params

        if self.conversion_params.NEW_HABITAT_LAB_POLICY_OR_OLD == "old":
            # load policy as seen in real_policy.py
            self.config.defrost()
            self.config.RL.POLICY.OBS_TRANSFORMS.ENABLED_TRANSFORMS = []
            self.config.freeze()
            self.config.RL.POLICY["init"] = False

        # Init the policy class
        self.policy = policy_class.from_config(
            config=self.config,
            observation_space=observation_space,
            action_space=action_space,
        )

        print("Actor-critic architecture:", self.policy)

        if self.conversion_params.NEW_HABITAT_LAB_POLICY_OR_OLD == "old":
            # Load trained weights into the policy
            self.policy.load_state_dict(
                {
                    k[len("actor_critic.") :]: v
                    for k, v in checkpoint["state_dict"].items()
                }
            )
        else:
            # Load trained weights into the policy
            self.policy.load_state_dict(checkpoint["state_dict"])

        # Move it to the device
        self.policy.to(self.device)

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
        self.set_required_parameters()

    def reset(self):
        self.reset_ran = True
        self.test_recurrent_hidden_states = torch.zeros(
            1,  # The number of environments. Just one for real world.
            self.policy.net.num_recurrent_layers,
            self.config.habitat_baselines.rl.ppo.hidden_size
            if self.conversion_params.NEW_HABITAT_LAB_POLICY_OR_OLD == "new"
            else self.config.RL.PPO.hidden_size,
            device=self.device,
        )

        # We start an episode with 'done' being True (0 for 'not_done')
        self.not_done_masks = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
        self.prev_actions = torch.zeros(1, self.num_actions, device=self.device)

    def set_required_parameters(self):
        self.rl_ppo_hidden_size = (
            self.config.habitat_baselines.rl.ppo.hidden_size
            if self.conversion_params.NEW_HABITAT_LAB_POLICY_OR_OLD == "new"
            else self.config.RL.PPO.hidden_size
        )
        self.min_log_std = (
            self.config.habitat_baselines.rl.policy.main_agent.action_dist.min_log_std
            if self.conversion_params.NEW_HABITAT_LAB_POLICY_OR_OLD == "new"
            else None
        )
        self.max_log_std = (
            self.config.habitat_baselines.rl.policy.main_agent.action_dist.max_log_std
            if self.conversion_params.NEW_HABITAT_LAB_POLICY_OR_OLD == "new"
            else None
        )
        self.std = (
            self.policy.action_distribution.std
            if self.conversion_params.NEW_HABITAT_LAB_POLICY_OR_OLD == "new"
            else torch.zeros((1,))
        )
        self.num_recurrent_layers = self.policy.net.num_recurrent_layers

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
        # Using torch script to save action distribution
        print("Action Distribtion net input size ", self.policy.net.output_size)
        action_dist_input_size = (
            512
            if self.conversion_params.NEW_HABITAT_LAB_POLICY_OR_OLD == "new"
            else self.policy.net.output_size
        )
        traced_cell_ad = torch.jit.trace(
            self.policy.action_distribution.mu_maybe_std
            if self.conversion_params.NEW_HABITAT_LAB_POLICY_OR_OLD == "new"
            else self.policy.action_distribution,
            (torch.ones((1, action_dist_input_size)).to(self.device)),
            strict=False,
        )
        # Create a Wrapper Network
        wrapper_network = FinalTorchscriptModel(
            traced_cell,
            traced_cell_ad,
            self.conversion_params.NEW_HABITAT_LAB_POLICY_OR_OLD,
            self.std,
            self.min_log_std,
            self.max_log_std,
        )
        wrapper_network.to(self.device)
        wrapper_network = torch.jit.trace(
            wrapper_network,
            (
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
            ),
        )
        # Extra Flags to save
        networkmeta: dict = {
            "num_recurrent_layers": self.num_recurrent_layers,
            "RL_PPO_HIDDEN_SIZE": self.rl_ppo_hidden_size,
        }
        networkmeta_pkl_data = pickle.dumps(networkmeta)
        torch.jit.save(
            wrapper_network,
            save_path["full_net"],
            _extra_files={"net_meta_dict.pkl": networkmeta_pkl_data},
        )

        # Load the saved network
        extra_files = {"net_meta_dict.pkl": ""}
        self.ts_net = torch.jit.load(
            save_path["full_net"],
            _extra_files=extra_files,
            map_location=str(self.device),
        )
        print("Extra file contents", pickle.loads(extra_files["net_meta_dict.pkl"]))
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
            actions = (
                PolicyActionData.actions
                if self.conversion_params.NEW_HABITAT_LAB_POLICY_OR_OLD == "new"
                else PolicyActionData[1]
            )
            self.test_recurrent_hidden_states = (
                PolicyActionData.rnn_hidden_states
                if self.conversion_params.NEW_HABITAT_LAB_POLICY_OR_OLD == "new"
                else PolicyActionData[-1]
            )
            # print("original hidden state:", PolicyActionData.rnn_hidden_states)

        self.prev_actions.copy_(actions)
        self.not_done_masks = torch.ones(1, 1, dtype=torch.bool, device=self.device)

        # GPU/CPU torch tensor -> numpy
        actions = actions.squeeze().cpu().numpy()

        return actions

    def act_ts(self, observations):
        """Using torchscript to run the model"""
        assert self.ts_net is not None, "Please load the torchscript policy first!"

        # Using torch script to save the model
        batch = batch_obs([observations], device=self.device)
        with torch.no_grad():
            (action, rnn_hidden_states) = self.ts_net(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
            )
            action = action.squeeze().cpu().numpy()
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
    save_path["full_net"] = config.OUTPUT_COMBINED_NET_SAVE_PATH

    # create output folder path if it doesn't exists
    dir_name = os.path.dirname(save_path["full_net"])
    os.makedirs(dir_name, exist_ok=True)

    # import the Habitat Policy
    module_path: Any = str(config.MODEL_CLASS_NAME).split(".")
    policy_class_name, module_path = module_path[-1], ".".join(module_path[:-1])
    module = importlib.import_module(module_path)
    policy_class = getattr(module, policy_class_name)

    # create observations & observation space from OBSERVATIONS_DICT
    observation_space: dict = {}
    observations: dict = {}

    for key, value in dict(config.OBSERVATIONS_DICT).items():
        # if normal case don't include spot_head_stereo_depth_sensor key
        if not config.USE_STEREO_PAIR_CAMERA and "stereo" in key:
            continue
        shape, low, high, dtype = value
        low, high, dtype = str(low), str(high), str(dtype)
        observation_space[key] = spaces.Box(
            low=eval(low),
            high=eval(high),
            shape=shape,
            dtype=eval(dtype),
        )
        observations[key] = np.zeros(shape, dtype=eval(dtype))
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
        conversion_params=config,
    )
    pytorchpolicyconverter.reset()

    # Normal way in hab3 to get the action
    # save Pytorch actions & reccurent hidden states to compare with torchscript actions & hidden states
    actions_pytorch = pytorchpolicyconverter.act(observations)
    recurrent_hidden_state_pytorch = pytorchpolicyconverter.test_recurrent_hidden_states

    print("actions from pytorch model:", actions_pytorch)

    # Convert to torchscript model
    pytorchpolicyconverter.convert_from_pytorch_to_torchscript(save_path, observations)
    pytorchpolicyconverter.reset()

    # Extract actions & hidden states from the torchscript model
    (
        actions_torchscript,
        recurrent_hidden_state_torchscript,
    ) = pytorchpolicyconverter.act_ts(observations)
    print("actions from torchscript model:", np.round(actions_torchscript, 4))

    # Compare actions & rnn states from pytorch & torchscript model they should be pretty close
    is_same_actions = np.allclose(
        np.round(actions_pytorch, 4), np.round(actions_torchscript, 4)
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
    print(f"Torchscipt net model saved at {save_path['full_net']}")


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
