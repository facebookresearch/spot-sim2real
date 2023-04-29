#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
from torch import nn as nn

from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.utils.common import GaussianNet, initialized_linear


def construct_mlp_base(input_size, hidden_size, num_layers=3, init=True):
    """Returns 3-layer MLP as a list of layers"""
    layers = []
    prev_size = input_size
    for _ in range(num_layers):
        layers.append(
            initialized_linear(
                int(prev_size), int(hidden_size), gain=np.sqrt(2), init=init
            )
        )
        layers.append(nn.ReLU())
        prev_size = hidden_size
    return layers


class MoePolicy(Policy, nn.Module):
    """
    Need 3 networks:
    A Net->Gaussian Gating Network (action shape == num_experts)
    A Net->Gaussian Residual Network (action shape == 2 base + 4 joint actions)
    A Net->Critic Network (used for RL training, unused for BC or test time)
    """

    def __init__(
        self,
        observation_space,
        fuse_states,
        num_gates,
        num_actions,
        use_rnn=False,
        blackout_gater=False,
        init=True,
    ):
        nn.Module.__init__(self)
        hidden_size = 512
        self.use_rnn = use_rnn
        self.num_gates = num_gates
        self.num_actions = num_actions
        self.fuse_states = fuse_states
        self.blackout_gater = blackout_gater
        self.blackout_obs = None
        spaces = observation_space.spaces
        input_size = sum([spaces[n].shape[0] for n in self.fuse_states])

        # Residual actor
        if self.use_rnn:
            self.residual_actor = RNNActorCritic(
                nn.Sequential(
                    *construct_mlp_base(input_size, hidden_size, init=init)
                ),
                GaussianNet(hidden_size, num_actions, init=init),
                init=init,
            )
        else:
            self.residual_actor = nn.Sequential(
                *construct_mlp_base(input_size, hidden_size, init=init),
                GaussianNet(hidden_size, num_actions, init=init),
            )

        # Gating actor

        if self.use_rnn:
            self.gating_actor = RNNActorCritic(
                nn.Sequential(
                    *construct_mlp_base(input_size, hidden_size, init=init)
                ),
                GaussianNet(hidden_size, num_gates, init=init),
                init=init,
            )
        else:
            self.gating_actor = nn.Sequential(
                *construct_mlp_base(input_size, hidden_size, init=init),
                GaussianNet(hidden_size, num_gates, init=init),
            )

        # Critic
        if self.use_rnn:
            self.critic = RNNActorCritic(
                nn.Sequential(
                    *construct_mlp_base(input_size, hidden_size, init=init)
                ),
                initialized_linear(hidden_size, 1, gain=np.sqrt(2), init=init),
                init=init,
            )
        else:
            self.critic = nn.Sequential(
                *construct_mlp_base(input_size, hidden_size, init=init),
                initialized_linear(hidden_size, 1, gain=np.sqrt(2), init=init),
            )

    def obs_to_tensor(self, observations, exclude=()):
        # Convert double to float if found
        for k, v in observations.items():
            if v.dtype is torch.float64:
                observations[k] = v.type(torch.float32)
        obs_keys = [k for k in self.fuse_states if k not in exclude]

        if self.blackout_gater:
            # Mask out visual features where corresponding image was zeros
            num_envs = observations["goal_heading"].shape[0]

            def am_close(env_idx):
                dist = observations[
                    "target_point_goal_gps_and_compass_sensor"
                ][env_idx][0]
                heading = observations["goal_heading"][env_idx][0]
                return dist < 0.3 and abs(heading) < 0.174533

            blind_inds = [i for i in range(num_envs) if am_close(i)]
            # Two assumptions: nav features are first, and are same length as
            # gaze features
            obs_copy = observations.copy()
            visual_features = obs_copy["visual_features"]
            visual_feat_len = visual_features.shape[1]
            nav_feats = visual_features[:, visual_feat_len // 2]
            feats_mask = torch.ones_like(nav_feats)
            feats_mask[blind_inds] = 0.0
            visual_features[:, visual_feat_len // 2] = nav_feats * feats_mask
            obs_copy["visual_features"] = visual_features
            self.blackout_obs = torch.cat(
                [obs_copy[k] for k in obs_keys], dim=1
            )

        return torch.cat([observations[k] for k in obs_keys], dim=1)

    def act(
        self,
        observations,
        rnn_hidden_states,  # don't use RNNs for now
        prev_actions,  # don't use prev_actions for now
        masks,  # don't use RNNs for now
        deterministic=False,
        actions_only=False,
    ):
        if self.use_rnn:
            (
                residual_distribution,
                gating_distribution,
                value,
                rnn_hidden_states,
            ) = self.compute_actions_and_value(
                observations,
                rnn_hidden_states,
                masks,
                actions_only=actions_only,
            )
        else:
            (
                residual_distribution,
                gating_distribution,
                value,
            ) = self.compute_actions_and_value(
                observations, actions_only=actions_only
            )

        action_and_log_probs = []
        for d in [residual_distribution, gating_distribution]:
            act = d.mode() if deterministic else d.sample()
            if actions_only:
                log_probs = None
            else:
                log_probs = d.log_probs(act)
            action_and_log_probs.extend([act, log_probs])

        res_act, res_log_p, gate_act, gate_log_p = action_and_log_probs

        action = torch.cat([res_act, gate_act], dim=1)

        if actions_only:
            action_log_probs = None
        else:
            action_log_probs = res_log_p + gate_log_p

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        if self.use_rnn:
            return self.critic(
                self.obs_to_tensor(observations),
                rnn_hidden_states[:, :, 512 * 2 :],
                masks,
            )[0]
        return self.critic(self.obs_to_tensor(observations))

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        if self.use_rnn:
            (
                residual_distribution,
                gating_distribution,
                value,
                rnn_hidden_states,
            ) = self.compute_actions_and_value(
                observations, rnn_hidden_states, masks
            )
        else:
            (
                residual_distribution,
                gating_distribution,
                value,
            ) = self.compute_actions_and_value(observations)

        res_act, gate_act = torch.split(
            action, [self.num_actions, self.num_gates], dim=1
        )

        action_log_probs = gating_distribution.log_probs(
            gate_act
        ) + residual_distribution.log_probs(res_act)
        distribution_entropy = (
            residual_distribution.entropy() + gating_distribution.entropy()
        )

        return value, action_log_probs, distribution_entropy, rnn_hidden_states

    def compute_actions_and_value(
        self,
        observations,
        rnn_hidden_states=None,
        masks=None,
        actions_only=False,
    ):
        observations_tensor = self.obs_to_tensor(observations)
        if self.use_rnn:
            assert rnn_hidden_states is not None
            assert masks is not None
            hx_1 = rnn_hidden_states[:, :, :512]
            hx_2 = rnn_hidden_states[:, :, 512 : 512 * 2]
            hx_3 = rnn_hidden_states[:, :, 512 * 2 :]
            residual_distribution, hx_1 = self.residual_actor(
                observations_tensor, hx_1, masks
            )

            if self.blackout_gater:
                gating_in = self.blackout_obs
            else:
                gating_in = observations_tensor
            gating_distribution, hx_2 = self.gating_actor(
                gating_in, hx_2, masks
            )
            if actions_only:
                value = None
            else:
                value, hx_3 = self.critic(observations_tensor, hx_3, masks)
            rnn_hidden_states = torch.cat([hx_1, hx_2, hx_3], dim=2)

            return (
                residual_distribution,
                gating_distribution,
                value,
                rnn_hidden_states,
            )
        else:
            residual_distribution = self.residual_actor(observations_tensor)
            gating_distribution = self.gating_actor(observations_tensor)
            if actions_only:
                value = None
            else:
                value = self.critic(observations_tensor)
            return residual_distribution, gating_distribution, value

    def forward(self, *x):
        raise NotImplementedError


class RNNActorCritic(nn.Module):
    def __init__(
        self, before_module, after_module, hidden_size=512, init=True
    ):
        super().__init__()
        self.before_module = before_module
        self.after_module = after_module
        self.rnn = build_rnn_state_encoder(hidden_size, hidden_size, init=init)

    def forward(self, observations, rnn_hidden_states, masks):
        x_out = self.before_module(observations)
        x_out, rnn_hidden_states = self.rnn(x_out, rnn_hidden_states, masks)
        x_out = self.after_module(x_out)

        return x_out, rnn_hidden_states
