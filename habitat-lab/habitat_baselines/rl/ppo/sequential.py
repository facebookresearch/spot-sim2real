from collections import OrderedDict

import torch
from gym import spaces
from gym.spaces import Box

from habitat.config import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo.policy import PointNavBaselinePolicy

"""
Assumption: we only deal with nav, pick, place, navpick, and navplace

Inputs: policy checkpoints, including what kind of env they are for.

- Need a function that decides who is next when one policy declares it's done
- How does each policy declare that it is done? Depends on:
    - Policy type (easy)
    - Timeout (easy)
    - State of the environment (hard)
    -
- Completely disregard prev_action and hnn input
    - use the one saved from last time
- Need a way to update the target
    - This may become a bigger problem later

Agent starts either in nav or manipulation. Nav seems better.

Master observation and action spaces are decided by the

"""

NAVPICK = "navpick"
NAVPLACE = "navplace"
# PICK = "rearrang_pick_analysis_bbox"
PICK = "spot_gaze"
PLACE = "rearrang_place_analysis"
NAV = "nav_v2"

# Denotes what visual observations an expert did NOT use during training
SKILL2VISUAL_BLACKLIST = {
    NAV: ["arm_depth", "arm_depth_bbox"],
    NAVPLACE: ["arm_depth", "arm_depth_bbox"],
    PICK: ["depth", "spot_left_depth", "spot_right_depth"],
    PLACE: [
        "depth",
        "arm_depth",
        "arm_depth_bbox",
        "spot_left_depth",
        "spot_right_depth",
    ],
}


def ckpt_to_policy(checkpoint, observation_space):
    config = checkpoint["config"]
    state_dict = checkpoint["state_dict"]

    # Determine the action space directly from the saved weights
    num_actions = state_dict["actor_critic.action_distribution.mu.bias"].shape[
        0
    ]
    action_space = Box(-1.0, 1.0, (num_actions,))

    # Filter out visual input keys that the policy didn't train with
    skill_type = config.hab_env_config
    policy_observation_space = spaces.Dict(
        {
            k: v
            for k, v in observation_space.spaces.items()
            if k not in SKILL2VISUAL_BLACKLIST[skill_type]
        }
    )

    if skill_type == PLACE:
        config.defrost()
        config.RL.POLICY.force_blind = True
        config.freeze()
    policy = PointNavBaselinePolicy.from_config(
        config=config,
        observation_space=policy_observation_space,
        action_space=action_space,
    )

    # Load weights
    policy.load_state_dict(
        {
            k[len("actor_critic.") :]: torch.tensor(v)
            for k, v in state_dict.items()
            if k.startswith("actor_critic.")
        }
    )

    return policy


def get_blank_params(config, policy, device, num_envs=1):
    hidden_state = torch.zeros(
        num_envs,
        1,  # num_recurrent_layers. SimpleCNN uses 1.
        config.RL.PPO.hidden_size,  # ppo_cfg.hidden_size,
        device=device,
    )

    masks = torch.zeros(
        num_envs,
        1,  # Just need one boolean.
        dtype=torch.bool,
        device=device,
    )

    if hasattr(policy, "action_distribution"):
        num_actions = policy.action_distribution.mu.out_features
    else:
        # Assume this is a MoE v2
        num_actions = policy.num_actions + policy.num_gates
    prev_actions = torch.zeros(
        num_envs,
        num_actions,
        device=device,
    )

    return hidden_state, masks, prev_actions


# For evaluation only!
@baseline_registry.register_policy
class SequentialExperts(PointNavBaselinePolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        experts,  # paths to .pth checkpoints
    ):
        # We just need this so things don't break...
        super().__init__(
            observation_space,
            action_space,
            hidden_size=512,
            goal_hidden_size=512,
            fuse_states=[],
            force_blind=False,
        )

        # Maps expert type (name of env used to train) to policies
        self.expert_skills = OrderedDict()
        experts = [i for i in experts if i != ""]
        for expert in experts:
            checkpoint = torch.load(expert, map_location="cpu")
            config = checkpoint["config"]
            skill_type = config.hab_env_config
            self.expert_skills[skill_type] = {
                "policy": ckpt_to_policy(checkpoint, observation_space),
                "config": config,
                "skill_type": skill_type,
            }

        # Load things to CPU for now
        self.device = torch.device("cpu")

        # Assume first checkpoint given corresponds to the first policy to use
        self.current_skill = list(self.expert_skills.values())[0]
        self.hidden_state, self.masks, self.prev_actions = get_blank_params(
            self.current_skill["config"],
            self.current_skill["policy"],
            self.device,
        )

        self.num_steps = 0
        self.num_transitions = 0
        self.next_skill_type = ""

    def reset(self):
        print("Resetting SequentialExperts...")
        # Assume first checkpoint given corresponds to the first policy to use
        self.current_skill = list(self.expert_skills.values())[0]
        self.hidden_state, self.masks, self.prev_actions = get_blank_params(
            self.current_skill["config"],
            self.current_skill["policy"],
            self.device,
        )

        self.num_steps = 0
        self.num_transitions = 0
        self.next_skill_type = ""

    @property
    def current_skill_type(self):
        return self.current_skill["skill_type"]

    # Overload .to() method
    def to(self, device, *args):
        super().to(device, *args)
        for skill_name in self.expert_skills.keys():
            self.expert_skills[skill_name]["policy"] = self.expert_skills[
                skill_name
            ]["policy"].to(device)
        self.hidden_state = self.hidden_state.to(device)
        self.masks = self.masks.to(device)
        self.prev_actions = self.prev_actions.to(device)
        self.device = device

    @classmethod
    def from_config(
        cls, config: Config, observation_space: spaces.Dict, action_space
    ):
        assert (
            config.NUM_PROCESSES == 1
        ), "SequentialExperts only works with 1 environment"
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            experts=[
                config.RL.POLICY.nav_checkpoint_path,
                config.RL.POLICY.gaze_checkpoint_path,
                config.RL.POLICY.place_checkpoint_path,
            ],
        )

    def update_current_policy(self, next_skill_type):
        """Baton pass if observations reflect that current policy is done"""

        assert next_skill_type in self.expert_skills, (
            "SequentialExperts does not have the requested skill of "
            f"'{next_skill_type}'!"
        )

        self.current_skill = self.expert_skills[next_skill_type]

        self.hidden_state, self.masks, self.prev_actions = get_blank_params(
            self.current_skill["config"],
            self.current_skill["policy"],
            self.device,
        )

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        if masks[0] == 0.0:
            self.reset()
        elif self.next_skill_type != "":
            print(
                f"SequentialExperts changing from {self.current_skill_type}"
                f" to {self.next_skill_type}!"
            )
            self.update_current_policy(self.next_skill_type)
            self.next_skill_type = ""

        _, action, _, self.hidden_state = self.current_skill["policy"].act(
            observations,
            self.hidden_state,
            self.prev_actions,
            self.masks,
            deterministic=False,
        )
        self.prev_actions = action
        self.masks = torch.ones(
            1,  # num_envs
            1,  # Just need one boolean.
            dtype=torch.bool,
            device=self.device,
        )

        # Pad expert actions to match the full action shape
        if self.current_skill_type in [PICK, PLACE]:
            z = torch.zeros(1, 2, device=self.device)
            action = torch.cat([action, z], dim=1)
        elif self.current_skill_type == NAV:
            z = torch.zeros(1, 4, device=self.device)
            action = torch.cat([z, action], dim=1)
        else:
            raise NotImplementedError

        # We don't use these, but need to return them
        value, action_log_probs = None, None

        return value, action, action_log_probs, rnn_hidden_states
