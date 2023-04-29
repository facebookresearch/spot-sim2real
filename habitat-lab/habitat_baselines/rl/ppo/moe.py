import numpy as np
import torch
from gym.spaces import Box, Dict

from habitat.config import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.rl.ppo.moe_v2 import MoePolicy
from habitat_baselines.rl.ppo.sequential import (
    ckpt_to_policy,
    get_blank_params,
)
from habitat_baselines.utils.common import ObservationBatchingCache, batch_obs

ARM_ACTIONS = 4  # 4 controllable joints
BASE_ACTIONS = 2  # linear and angular vel
EXPERT_NAV_UUID = "expert_nav"
EXPERT_GAZE_UUID = "expert_gaze"
EXPERT_PLACE_UUID = "expert_place"
EXPERT_MASKS_UUID = "expert_masks"
VISUAL_FEATURES_UUID = "visual_features"
EXPERT_ACTIONS_UUID = "expert_actions"
CORRECTIONS_UUID = "corrections"


@baseline_registry.register_policy
class NavGazeMixtureOfExpertsRes(MoePolicy):
    def __init__(
        self, observation_space: Dict, action_space, config, *args, **kwargs
    ):
        # First, immediately extract values from config for clarity
        num_envs = config.NUM_ENVIRONMENTS
        hidden_size = config.RL.PPO.hidden_size
        nav_ckpt_path = config.RL.POLICY.nav_checkpoint_path
        gaze_ckpt_path = config.RL.POLICY.gaze_checkpoint_path
        place_ckpt_path = config.RL.POLICY.place_checkpoint_path
        freeze_keys = config.RL.POLICY.freeze_keys
        self.num_experts = 2 if place_ckpt_path == "" else 3
        self.fuse_states = config.RL.POLICY.fuse_states
        self.blind = config.RL.POLICY.force_blind
        self.obs_expert_actions = config.RL.POLICY.get(
            "obs_teacher_actions", False
        )

        # Determines if we use visual observations or features from experts
        observation_space_copy = Dict(observation_space.spaces.copy())
        if self.blind:
            vis_keys = [
                k for k in observation_space.spaces.keys() if "depth" in k
            ]
            for k in vis_keys:
                observation_space.spaces.pop(k)
            # Add visual features to the observation space
            observation_space.spaces[VISUAL_FEATURES_UUID] = Box(
                -np.inf, np.inf, (hidden_size * 2,)
            )
            self.fuse_states = [VISUAL_FEATURES_UUID] + self.fuse_states

        if self.obs_expert_actions:
            # Add expert actions to the` observation space and fuse_states
            observation_space.spaces[EXPERT_NAV_UUID] = Box(
                -1.0, 1.0, (BASE_ACTIONS,)
            )
            observation_space.spaces[EXPERT_GAZE_UUID] = Box(
                -1.0, 1.0, (ARM_ACTIONS,)
            )
            self.fuse_states.extend([EXPERT_NAV_UUID, EXPERT_GAZE_UUID])

        # Instantiate MoE's own policy
        super().__init__(
            observation_space,
            fuse_states=self.fuse_states,
            num_gates=self.num_experts,
            num_actions=BASE_ACTIONS + ARM_ACTIONS,
            use_rnn=config.RL.POLICY.get("use_rnn", False),
            blackout_gater=config.RL.POLICY.get("blackout_gater", False),
            init=config.RL.POLICY.init,
        )

        # For RolloutStorage in ppo_trainer.py
        self.policy_action_space = action_space

        # Get model params before experts get loaded in
        self.model_params = [p for p in self.parameters() if p.requires_grad]

        # Freeze weights that contain any freeze keys
        for key in freeze_keys:
            if key == "None":
                continue
            for name, param in self.named_parameters():
                if key in name:
                    param.requires_grad = False
                    print("Freezing:", name)

        # Store tensors into CPU for now
        self.device = torch.device("cpu")

        # Load pre-trained experts
        def get_ckpt_policy(ckpt_path):
            if ckpt_path == "":
                return None, None
            ckpt = torch.load(ckpt_path, map_location=self.device)
            ckpt["config"].RL.POLICY["init"] = False
            policy = ckpt_to_policy(ckpt, observation_space_copy)
            policy.eval()
            return ckpt, policy

        nav_ckpt, self.expert_nav_policy = get_ckpt_policy(nav_ckpt_path)
        gaze_ckpt, self.expert_gaze_policy = get_ckpt_policy(gaze_ckpt_path)
        place_ckpt, self.expert_place_policy = get_ckpt_policy(place_ckpt_path)

        # Freeze expert weights
        for name, param in self.named_parameters():
            if "expert" in name:
                param.requires_grad = False

        def get_hx_prev_actions(ckpt, policy):
            if ckpt is None:
                return None, None
            hx, _, prev_actions = get_blank_params(
                ckpt["config"], policy, torch.device("cpu"), num_envs=num_envs
            )
            return hx, prev_actions

        self.nav_rnn_hx, self.nav_prev_actions = get_hx_prev_actions(
            nav_ckpt, self.expert_nav_policy
        )
        self.gaze_rnn_hx, self.gaze_prev_actions = get_hx_prev_actions(
            gaze_ckpt, self.expert_gaze_policy
        )
        self.place_rnn_hx, self.place_prev_actions = get_hx_prev_actions(
            place_ckpt, self.expert_place_policy
        )

        (
            self.nav_masks,
            self.gaze_masks,
            self.place_masks,
            self.prev_nav_masks,
            self.prev_gaze_masks,
            self.prev_place_masks,
        ) = [torch.ones(num_envs, 1, dtype=torch.bool) for _ in range(6)]

        self.nav_action = torch.zeros(
            num_envs, BASE_ACTIONS, device=self.device
        )
        self.gaze_action = torch.zeros(
            num_envs, ARM_ACTIONS, device=self.device
        )
        self.place_action = torch.zeros(
            num_envs, ARM_ACTIONS, device=self.device
        )
        self._obs_batching_cache = ObservationBatchingCache()
        self.active_envs = list(range(num_envs))
        self.deterministic_nav = False
        self.deterministic_gaze = False
        self.deterministic_place = False

    @classmethod
    def from_config(
        cls, config: Config, observation_space: Dict, action_space
    ):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            config=config,
        )

    def to(self, device, *args):
        super().to(device, *args)
        self.device = device
        self.nav_rnn_hx = self.nav_rnn_hx.to(device)
        self.gaze_rnn_hx = self.gaze_rnn_hx.to(device)
        self.nav_prev_actions = self.nav_prev_actions.to(device)
        self.gaze_prev_actions = self.gaze_prev_actions.to(device)
        self.expert_nav_policy = self.expert_nav_policy.to(device)
        self.expert_gaze_policy = self.expert_gaze_policy.to(device)
        self.nav_masks = self.nav_masks.to(device)
        self.gaze_masks = self.gaze_masks.to(device)
        self.prev_nav_masks = self.prev_nav_masks.to(device)
        self.prev_gaze_masks = self.prev_gaze_masks.to(device)

        if self.expert_place_policy is not None:
            self.place_rnn_hx = self.place_rnn_hx.to(device)
            self.place_prev_actions = self.place_prev_actions.to(device)
            self.expert_place_policy = self.expert_place_policy.to(device)
            self.place_masks = self.place_masks.to(device)
            self.prev_place_masks = self.prev_place_masks.to(device)

    def get_expert_actions(self, batch, masks, gates=None):
        # Determine if one of the VectorEnvs were paused
        key = list(batch.keys())[0]
        if batch[key].shape[0] < len(self.active_envs):
            # Determine which environment ID was removed
            # TODO: Finish implementing this
            raise RuntimeError("Number of Vector Envs decreased!")
        elif batch[key].shape[0] > len(self.active_envs):
            raise RuntimeError("Number of Vector Envs increased!")

        masks_device = (
            masks.to(self.device) if masks.device != self.device else masks
        )
        nav_masks = torch.logical_and(masks_device, self.nav_masks)
        gaze_masks = torch.logical_and(masks_device, self.gaze_masks)
        if self.expert_place_policy is not None:
            place_masks = torch.logical_and(masks_device, self.place_masks)

        num_envs = masks.shape[0]
        if num_envs == 1 and gates is None:
            gates = torch.ones(num_envs, 3)

        with torch.no_grad():
            if num_envs > 1 or gates[0, 0] > 0:
                (
                    _,
                    self.nav_action,
                    _,
                    self.nav_rnn_hx,
                ) = self.expert_nav_policy.act(
                    batch,
                    self.nav_rnn_hx,
                    self.nav_prev_actions,
                    nav_masks,
                    deterministic=self.deterministic_nav,
                    actions_only=True,
                )

            if num_envs > 1 or gates[0, 1] > 0:
                (
                    _,
                    self.gaze_action,
                    _,
                    self.gaze_rnn_hx,
                ) = self.expert_gaze_policy.act(
                    batch,
                    self.gaze_rnn_hx,
                    self.gaze_prev_actions,
                    gaze_masks,
                    deterministic=self.deterministic_gaze,
                    actions_only=True,
                )
            if self.expert_place_policy is not None and (
                num_envs > 1 or gates[0, 2] > 0
            ):
                self.expert_place_policy.net.speak = True
                (
                    _,
                    self.place_action,
                    _,
                    self.place_rnn_hx,
                ) = self.expert_place_policy.act(
                    batch,
                    self.place_rnn_hx,
                    self.place_prev_actions,
                    place_masks,
                    deterministic=self.deterministic_place,
                    actions_only=True,
                )

        self.nav_prev_actions.copy_(self.nav_action)
        self.gaze_prev_actions.copy_(self.gaze_action)
        if self.expert_place_policy is not None:
            self.place_prev_actions.copy_(self.place_action)

        # Move expert actions to CPU (for observation/action_arg insertion)
        self.nav_action = self.nav_action.detach().cpu()
        self.gaze_action = self.gaze_action.detach().cpu()
        if self.expert_place_policy is not None:
            self.place_action = self.place_action.detach().cpu()

    def transform_obs(self, observations, masks, obs_transforms=None):
        """
        Inserts expert actions into the observations

        :param observations: list of dictionaries
        :param masks: torch tensor.bool
        :return: observations with expert actions
        """
        # If we observe expert actions, compute them now
        if self.obs_expert_actions:
            batch = batch_obs(
                observations,
                device=self.device,
                cache=self._obs_batching_cache,
            )
            if obs_transforms is not None:
                batch = apply_obs_transforms_batch(batch, obs_transforms)
            self.get_expert_actions(batch, masks)
        elif VISUAL_FEATURES_UUID in self.fuse_states:
            # Even if we don't use the expert actions as observations, we still
            # need to forward pass through them to get their visual features
            batch = batch_obs(
                observations,
                device=self.device,
                cache=self._obs_batching_cache,
            )
            if obs_transforms is not None:
                batch = apply_obs_transforms_batch(batch, obs_transforms)
            # Place expert does not have a visual encoder
            self.expert_nav_policy.net.get_vis_feats(batch)
            self.expert_gaze_policy.net.get_vis_feats(batch)

        # Insert into each observation; batch_obs loads these into gpu later
        num_envs = len(observations)
        for index_env in range(num_envs):
            if self.obs_expert_actions:
                observations[index_env][EXPERT_NAV_UUID] = self.nav_action[
                    index_env
                ].numpy()
                observations[index_env][EXPERT_GAZE_UUID] = self.gaze_action[
                    index_env
                ].numpy()
                if self.expert_place_policy is not None:
                    observations[index_env][
                        EXPERT_PLACE_UUID
                    ] = self.place_action[index_env].numpy()

            if self.blind:
                # Remove visual observations
                vis_keys = [
                    k for k in observations[index_env].keys() if "depth" in k
                ]
                for k in vis_keys:
                    observations[index_env].pop(k)

                # Stitch and add visual features from the experts
                visual_features = [
                    p.net.pred_visual_features[index_env]
                    for p in [self.expert_nav_policy, self.expert_gaze_policy]
                ]
                observations[index_env][VISUAL_FEATURES_UUID] = (
                    torch.cat(visual_features).cpu().numpy()
                )

        return observations

    def action_to_dict(self, action, index_env, **kwargs):
        # Merge mixer's actions with experts'
        gaze_action = self.gaze_action[index_env]
        nav_action = self.nav_action[index_env]
        if self.expert_place_policy is not None:
            place_action = self.place_action[index_env]
        step_action = action.to(torch.device("cpu"))

        # Add expert actions as action_args for reward calculation in RLEnv
        expert_args = {
            EXPERT_GAZE_UUID: gaze_action,
            EXPERT_NAV_UUID: nav_action,
        }
        if self.expert_place_policy is not None:
            expert_args[EXPERT_PLACE_UUID] = place_action
        step_data = {
            "action": {"action": step_action, **expert_args, **kwargs}
        }

        return step_data

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
        actions_only=False,
    ):
        value, action, action_log_probs, rnn_hidden_states = super().act(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            deterministic=deterministic,
            actions_only=actions_only,
        )

        # If expert actions were not observed, update them now
        if not self.obs_expert_actions:
            gates = action[:, ARM_ACTIONS + BASE_ACTIONS :]
            self.get_expert_actions(observations, masks, gates=gates)

        return value, action, action_log_probs, rnn_hidden_states


@baseline_registry.register_policy
class NavGazeMixtureOfExpertsMask(NavGazeMixtureOfExpertsRes):
    """
    This policy will signal which expert it thinks is most appropriate, use it
    to form a 'base' action composed of that expert's actions and 0s for the
    actions it does not control, and finally will also output actions that will
    be added to this base action.

    Unfortunately this causes the action space of the policy to be larger than
    that of the environment. We must ensure that RolloutStorage uses the
    policy's action space rather than the environment's, or else log_probs
    cannot be generated in the ppo.py script.
    """

    def __init__(
        self, observation_space, action_space, config, *args, **kwargs
    ):
        """Add actions for masking experts"""
        num_experts = 2 if config.RL.POLICY.place_checkpoint_path == "" else 3
        actual_action_space = Box(
            -1.0, 1.0, (action_space.shape[0] + num_experts,)
        )
        super().__init__(
            observation_space, actual_action_space, config, *args, **kwargs
        )
        self.residuals_on_inactive = config.RL.POLICY.residuals_on_inactive
        self.use_residuals = False #config.RL.POLICY.use_residuals
        self.dont_digitize = config.RL.POLICY.dont_digitize
        self.nav_action_mask = None
        self.gaze_action_mask = None
        self.place_action_mask = None
        self.arm_action_mask = None
        self.num_masks = self.num_experts

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
        update_masks=True,
        actions_only=False,
    ):
        """
        The residual actions need to be masked by the mask outputs in order for
        the log_probs to encourage the unused residuals to go towards zero if
        deemed a good decision, or away from zero if not.

        The last E actions are taken to be the mask actions, where E == # of
        experts. The mask order is Nav, Gaze, Place. The action order is Arm,
        Base.
        """
        value, action, action_log_probs, rnn_hidden_states = super().act(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            deterministic,
            actions_only=actions_only,
        )

        (
            residual_arm_actions,
            residual_base_actions,
            expert_masks,
        ) = torch.split(
            action, [ARM_ACTIONS, BASE_ACTIONS, self.num_masks], dim=1
        )

        # Update_masks could be False for teacher-forcing
        if update_masks:
            # Generate arm and base action masks
            self.get_action_masks(expert_masks)

        if self.residuals_on_inactive:
            # Residuals for NOT-selected (inactive) experts are ZERO-ED here
            residual_arm_actions = residual_arm_actions * (
                1.0 - self.arm_action_mask
            )
            residual_base_actions = residual_base_actions * (
                1.0 - self.nav_action_mask
            )

        action = torch.cat(
            [residual_arm_actions, residual_base_actions, expert_masks], dim=1
        )

        return value, action, action_log_probs, rnn_hidden_states

    def get_action_masks(self, expert_masks):
        if self.dont_digitize:
            activation_mask = (torch.clip(expert_masks, -1, 1) + 1) / 2
            activation_mask[activation_mask < 0.05] = 0.0
        else:
            activation_mask = expert_masks.detach().clone()
            if self.num_experts == 3:
                # We need to deactivate gaze or place in indices at which the
                # other arm expert has a higher value
                activation_mask[:, 1][
                    activation_mask[:, 1] < activation_mask[:, 2]
                ] = -1
                activation_mask[:, 2][
                    activation_mask[:, 2] <= activation_mask[:, 1]
                ] = -1
            activation_mask = torch.where(activation_mask > 0, 1.0, 0.0)
        activation_mask = torch.split(
            activation_mask, [1] * self.num_experts, dim=1
        )

        def reset_hx(prev, curr):
            # We want to return False when prev is False and curr is True
            reset = torch.logical_and(torch.eq(prev, 0), ~torch.eq(curr, 0))
            return torch.logical_not(reset)

        if self.num_experts == 2:
            nav_masks, gaze_masks = activation_mask
            self.nav_action_mask, self.gaze_action_mask = [
                m.repeat(1, num_actions)
                for m, num_actions in zip(
                    activation_mask, [BASE_ACTIONS, ARM_ACTIONS]
                )
            ]
            self.arm_action_mask = self.gaze_action_mask
        elif self.num_experts == 3:
            nav_masks, gaze_masks, place_masks = activation_mask
            (
                self.nav_action_mask,
                self.gaze_action_mask,
                self.place_action_mask,
            ) = [
                m.repeat(1, num_actions)
                for m, num_actions in zip(
                    activation_mask, [BASE_ACTIONS, ARM_ACTIONS, ARM_ACTIONS]
                )
            ]
            # Arm mask is the union of the gaze and place masks
            self.arm_action_mask = torch.clip(
                self.gaze_action_mask + self.place_action_mask, 0.0, 1.0
            )
        else:
            raise NotImplementedError

        # Zero out mask values indicating reselection of an expert
        self.nav_masks = reset_hx(self.prev_nav_masks, nav_masks)
        self.prev_nav_masks = nav_masks
        self.gaze_masks = reset_hx(self.prev_gaze_masks, gaze_masks)
        self.prev_gaze_masks = gaze_masks
        if self.expert_place_policy is not None:
            self.place_masks = reset_hx(self.prev_place_masks, place_masks)
            self.prev_place_masks = place_masks

    def action_to_dict(self, action, index_env, use_residuals=True, **kwargs):
        if self.use_residuals is not None:
            use_residuals = self.use_residuals

        # Merge mixer's actions with experts'
        gaze_action = self.gaze_action[index_env]
        gaze_action_mask = self.gaze_action_mask[index_env].detach().cpu()
        nav_action = self.nav_action[index_env]
        nav_action_mask = self.nav_action_mask[index_env].detach().cpu()
        if self.num_experts == 3:
            place_action = self.place_action[index_env]
            place_action_mask = (
                self.place_action_mask[index_env].detach().cpu()
            )
        else:
            place_action_mask = None

        experts_action_arg = torch.cat([gaze_action, nav_action])

        # Compile an action based on the actions of the selected experts
        base_action = nav_action * nav_action_mask
        if self.expert_place_policy is None:
            arm_action = gaze_action * gaze_action_mask
        else:
            assert max(gaze_action_mask + place_action_mask) <= 1
            arm_action = (
                gaze_action * gaze_action_mask
                + place_action * place_action_mask
            )
        experts_action = torch.cat([arm_action, base_action])

        if use_residuals:
            residual_action = action[: ARM_ACTIONS + BASE_ACTIONS]
            step_action = experts_action + residual_action
        else:
            step_action = experts_action

        # Add expert actions as action_args for reward calculation in RLEnv
        masks_arg = [
            float(nav_action_mask[0]),
            float(gaze_action_mask[0]),
            0.0 if place_action_mask is None else float(place_action_mask[0]),
        ]
        expert_args = {
            EXPERT_GAZE_UUID: gaze_action,
            EXPERT_NAV_UUID: nav_action,
            EXPERT_MASKS_UUID: masks_arg,
            EXPERT_ACTIONS_UUID: experts_action_arg,
            CORRECTIONS_UUID: residual_action
            if use_residuals
            else torch.zeros(1),
        }
        step_data = {
            "action": {"action": step_action, **expert_args, **kwargs}
        }

        return step_data


@baseline_registry.register_policy
class NavGazeMixtureOfExpertsMaskSingle(NavGazeMixtureOfExpertsMask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_masks = 1

    @classmethod
    def from_config(
        cls, config: Config, observation_space: Dict, action_space
    ):
        actual_action_space = Box(-1.0, 1.0, (action_space.shape[0] + 1,))

        return cls(
            observation_space=observation_space,
            action_space=actual_action_space,
            config=config,
        )

    def get_action_masks(self, expert_masks):
        if self.num_experts == 2:
            nav_masks = torch.where(expert_masks < 0, 1.0, -1.0)
            gaze_masks = torch.where(expert_masks > 0, 1.0, -1.0)
            expert_masks = [nav_masks, gaze_masks]
        elif self.num_experts == 3:
            nav_masks = torch.where(expert_masks < -0.25, 1.0, -1.0)
            gaze_masks = torch.where(
                torch.logical_and(-0.25 < expert_masks, expert_masks < 0.25),
                1.0,
                -1.0,
            )
            place_masks = torch.where(expert_masks > 0.25, 1.0, -1.0)
            expert_masks = [nav_masks, gaze_masks, place_masks]
        else:
            raise NotImplementedError
        expert_masks = torch.cat(expert_masks, dim=1)
        super().get_action_masks(expert_masks)


@baseline_registry.register_policy
class NavGazeMixtureOfExpertsMaskCombo(NavGazeMixtureOfExpertsMask):
    """
    The difference between Combo and Single is that the final action, the
    experts mask, is not the output of a Gaussian action distribution (-1 - 1),
    but rather is the output of a Categorical action distribution (0 - 2) or
    (0 - 4)
    """

    def __init__(
        self, observation_space, action_space, config, *args, **kwargs
    ):
        num_combos = config.RL.POLICY.num_combos
        num_actions = action_space.shape[0]
        actual_action_space = Box(-1.0, 1.0, (num_actions + 1,))
        super().__init__(
            observation_space, actual_action_space, config, *args, **kwargs
        )

        self.num_masks = 1
        self.num_combos = num_combos

        """Nav, Gaze, NavGaze | Place, NavPlace"""
        filter_valid_combos = lambda x: [i for i in x if i < num_combos]
        self.nav_combo_ids = filter_valid_combos([0, 2, 4])
        self.gaze_combo_ids = filter_valid_combos([1, 2])
        self.place_combo_ids = filter_valid_combos([3, 4])

    def get_action_masks(self, expert_masks):
        cmbs = [self.nav_combo_ids, self.gaze_combo_ids, self.place_combo_ids]

        expert_masks_out = []
        for combo_ids in cmbs:
            if len(combo_ids) == 0:
                continue
            expert_mask = torch.zeros_like(expert_masks, dtype=torch.bool)
            for combo_id in combo_ids:
                combo_id_mask = expert_masks == combo_id
                expert_mask = torch.logical_or(combo_id_mask, expert_mask)
            expert_mask_out = torch.where(expert_mask, 1.0, -1.0)
            expert_masks_out.append(expert_mask_out)

        expert_masks_out = torch.cat(expert_masks_out, dim=1)

        super().get_action_masks(expert_masks_out)

    def action_to_dict(self, action, index_env, **kwargs):
        return super().action_to_dict(
            action,
            index_env,
            use_residuals=not self.freeze_residuals,
            **kwargs,
        )
