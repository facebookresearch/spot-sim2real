#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc

import numpy as np
import torch
from gym import spaces
from torch import nn as nn

from habitat.config import Config
from habitat.core.spaces import ActionSpace
from habitat.tasks.nav.nav import (
    ImageGoalSensor,
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.models.simple_cnn import (
    ARM_VISION_KEYS,
    HEAD_VISION_KEYS,
    SimpleCNN,
)
from habitat_baselines.utils.common import (
    CategoricalNet,
    GaussianCategoricalNet,
    GaussianNet,
    initialized_linear,
)


class Policy(nn.Module, metaclass=abc.ABCMeta):
    def __init__(
        self,
        net,
        action_space,
        gaussian_categorical=False,
        init=True,
        **kwargs,
    ):
        super().__init__()
        self.net = net
        if net is not None:
            if isinstance(action_space, ActionSpace):
                self.action_distribution = CategoricalNet(
                    self.net.output_size, action_space.n
                )
            else:
                if gaussian_categorical:
                    self.action_distribution = GaussianCategoricalNet(
                        self.net.output_size, **kwargs
                    )
                else:
                    self.action_distribution = GaussianNet(
                        self.net.output_size, action_space.shape[0], init=init
                    )

            self.critic = CriticHead(self.net.output_size, init=init)

        self.distribution = None

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
        actions_only=False,
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()
        if actions_only:
            return None, action, None, rnn_hidden_states

        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)

        # Save for use in behavioral cloning the mean and std
        self.distribution = distribution

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config, observation_space, action_space):
        pass


class CriticHead(nn.Module):
    def __init__(self, input_size, init=True):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        if init:
            nn.init.orthogonal_(self.fc.weight)
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


@baseline_registry.register_policy
class PointNavBaselinePolicy(Policy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        **kwargs,
    ):
        super().__init__(
            PointNavBaselineNet(  # type: ignore
                observation_space=observation_space,
                hidden_size=hidden_size,
                **kwargs,
            ),
            action_space,
            **kwargs,
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: spaces.Dict, action_space
    ):
        goal_hidden_size = config.RL.PPO.get("goal_hidden_size", 0)
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=config.RL.PPO.hidden_size,
            goal_hidden_size=goal_hidden_size,
            fuse_states=config.RL.POLICY.fuse_states,
            force_blind=config.RL.POLICY.force_blind,
            init=config.RL.POLICY.get("init", True),
        )


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class PointNavBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_size: int,
        goal_hidden_size,
        fuse_states,
        force_blind,
        init=True,
    ):
        super().__init__()

        self.fuse_states = fuse_states
        self._n_input_goal = sum(
            [observation_space.spaces[n].shape[0] for n in self.fuse_states]
        )

        # Construct CNNs
        o_keys = observation_space.spaces.keys()
        head_visual_inputs = len([k for k in o_keys if k in HEAD_VISION_KEYS])
        arm_visual_inputs = len([k for k in o_keys if k in ARM_VISION_KEYS])
        self.num_cnns = min(1, head_visual_inputs) + min(1, arm_visual_inputs)

        self._hidden_size = hidden_size

        if self.num_cnns <= 1:
            if self.num_cnns == 0:
                force_blind = True
            self.visual_encoder = SimpleCNN(
                observation_space, hidden_size, force_blind, init=init
            )
        elif self.num_cnns == 2:
            # We are using both cameras; make a CNN for each
            head_obs_space, arm_obs_space = [
                spaces.Dict(
                    {
                        k: v
                        for k, v in observation_space.spaces.items()
                        if k not in obs_key_blacklist
                    }
                )
                for obs_key_blacklist in [ARM_VISION_KEYS, HEAD_VISION_KEYS]
            ]
            # Head CNN
            self.visual_encoder = SimpleCNN(
                head_obs_space,
                hidden_size,
                force_blind,
                head_only=True,
                init=init,
            )
            # Arm CNN
            self.visual_encoder2 = SimpleCNN(
                arm_obs_space,
                hidden_size,
                force_blind,
                arm_only=True,
                init=init,
            )
        else:
            raise RuntimeError(
                f"Only supports 1 or 2 CNNs not {self.num_cnns}"
            )

        # 2-layer MLP for non-visual inputs
        self._goal_hidden_size = goal_hidden_size
        if self._goal_hidden_size != 0:
            self.goal_encoder = nn.Sequential(
                initialized_linear(
                    self._n_input_goal,
                    self._goal_hidden_size,
                    gain=np.sqrt(2),
                    init=init,
                ),
                nn.ReLU(),
                initialized_linear(
                    self._goal_hidden_size,
                    self._goal_hidden_size,
                    gain=np.sqrt(2),
                    init=init,
                ),
                nn.ReLU(),
            )

        # Final RNN layer
        visual_size = 0 if self.is_blind else hidden_size * self.num_cnns
        self.state_encoder = build_rnn_state_encoder(
            visual_size + self._goal_hidden_size, self._hidden_size, init=init
        )

        self.pred_visual_features = None
        self.vis_feats_queued = False

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_vis_feats(self, observations):
        # Convert double to float if found
        for k, v in observations.items():
            if v.dtype is torch.float64:
                observations[k] = v.type(torch.float32)

        x = []

        # Visual observations
        if not self.is_blind:
            x.append(self.visual_encoder(observations))
            if self.num_cnns == 2:
                x.append(self.visual_encoder2(observations))
        elif "visual_features" in observations:
            x.append(observations["visual_features"])

        # Save visual features for use by other policies (mixer policy)
        self.pred_visual_features = torch.cat(x, dim=1) if x else None
        self.vis_feats_queued = True
        return self.pred_visual_features

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        x = []
        if not self.is_blind:
            if self.vis_feats_queued:
                x.append(self.pred_visual_features)
            else:
                x.append(self.get_vis_feats(observations))
            self.vis_feats_queued = False

        # Non-visual observations
        if len(self.fuse_states) > 0:
            non_vis_obs = [observations[k] for k in self.fuse_states]
            x.append(self.goal_encoder(torch.cat(non_vis_obs, dim=-1)))

        # Final RNN layer
        x_out = torch.cat(x, dim=1)
        x_out, rnn_hidden_states = self.state_encoder(
            x_out, rnn_hidden_states, masks
        )

        return x_out, rnn_hidden_states
