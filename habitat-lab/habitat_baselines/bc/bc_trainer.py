import glob
import os
import os.path as osp
from collections import deque
from contextlib import ExitStack

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from habitat import logger
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.rl.ppo.sequential import get_blank_params
from habitat_baselines.utils.common import batch_obs

EXPERT_NAV_UUID = "expert_nav"
EXPERT_GAZE_UUID = "expert_gaze"
EXPERT_PLACE_UUID = "expert_place"
EXPERT_NULL_UUID = "undetermined"  # for when there is no applicable expert
EXPERT_UUIDS = [
    EXPERT_NAV_UUID,
    EXPERT_GAZE_UUID,
    EXPERT_PLACE_UUID,
    EXPERT_NULL_UUID,
]


@baseline_registry.register_trainer(name="bc")
class BehavioralCloningMoe(BaseRLTrainer):
    def __init__(self, config, *args, **kwargs):
        logger.add_filehandler(config.LOG_FILE)
        # logger.info(f"Full config:\n{config}")

        self.config = config
        self.device = torch.device("cuda", 0)

        self.moe = None
        self.prev_actions = None
        self.masks = None
        self.num_actions = None
        self.envs = None
        self.success_deq = deque(maxlen=50)
        self.action_mse_deq = deque(maxlen=50)
        self.frames = []

        # Extract params from config
        self.batches_per_save = config.BATCHES_PER_CHECKPOINT
        self._batch_length = config.BATCH_LENGTH
        self.sl_lr = config.SL_LR
        self.policy_name = config.RL.POLICY.name
        self.total_num_steps = config.TOTAL_NUM_STEPS
        self.checkpoint_folder = osp.join(
            config.CHECKPOINT_FOLDER, config.PREFIX
        )
        self.tb_dir = config.TENSORBOARD_DIR
        self.bc_loss_type = config.BC_LOSS_TYPE
        self.load_weights = config.RL.DDPPO.pretrained
        self.teacher_forcing = config.TEACHER_FORCING
        self.num_envs = config.NUM_PROCESSES

        if not os.path.isdir(self.checkpoint_folder):
            os.makedirs(self.checkpoint_folder)

    def setup_teacher_student(self, del_envs=False):
        # Envs MUST be instantiated first
        observation_space = self.envs.observation_spaces[0]
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        # MoE and its experts are loaded here
        policy_cls = baseline_registry.get_policy(self.policy_name)
        self.moe = policy_cls.from_config(
            self.config, observation_space, self.envs.action_spaces[0]
        )

        # Load pretrained weights if provided
        if self.load_weights:
            pretrained_state = torch.load(
                self.config.RL.DDPPO.pretrained_weights, map_location="cpu"
            )
            orig_state_dict = self.moe.state_dict()
            self.moe.load_state_dict(
                {
                    k: v if "expert" not in k else orig_state_dict[k]
                    for k, v in pretrained_state["state_dict"].items()
                }
            )
        print("Actor-critic architecture:\n", self.moe)
        if hasattr(self.moe, "fuse_states"):
            fuse_states_list = "\n - ".join(self.moe.fuse_states)
            print("Fuse states:\n -", fuse_states_list)
        elif hasattr(self.moe.net, "fuse_states"):
            fuse_states_list = "\n - ".join(self.moe.net.fuse_states)
            print("Fuse states:\n -", fuse_states_list)
        if del_envs:
            self.envs.close()
            del self.envs
        self.moe.to(self.device)

        # Setup prev_actions, masks, and recurrent hidden states
        (
            self.rnn_hidden_states,
            self.masks,
            self.prev_actions,
        ) = get_blank_params(
            self.config, self.moe, self.device, num_envs=self.num_envs
        )
        self.num_actions = self.prev_actions.shape[1]
        if self.config.RL.POLICY.get("use_rnn", False):
            self.rnn_hidden_states = torch.zeros(
                self.num_envs,
                1,  # num_recurrent_layers. SimpleCNN uses 1.
                self.config.RL.PPO.hidden_size * 3,  # ppo_cfg.hidden_size,
                device=self.device,
            )

    def get_model_params(self):
        return self.moe.model_params

    def get_action_and_loss(self, batch):
        teacher_labels = self.get_teacher_labels(batch)
        if self.bc_loss_type == "log_prob":
            actions, action_loss = self.log_prob_loss(batch, teacher_labels)
        elif self.bc_loss_type == "mse":
            actions, action_loss = self.mse_loss(batch, teacher_labels)
        elif self.bc_loss_type == "mse_gaussian":
            actions, action_loss = self.mse_gaussian_loss(batch)
        else:
            raise NotImplementedError(f"Loss {self.bc_loss_type} unsupported!")

        if not self.bc_loss_type == "mse":
            mse_loss = F.mse_loss(actions, teacher_labels)
            self.action_mse_deq.append(mse_loss.detach().cpu().item())
        step_actions = self.stepify_actions(actions)

        return step_actions, action_loss

    def get_teacher_labels(self, batch, label_type="action"):
        # Extract teacher actions from the observations
        teacher_labels = []
        for idx, correct_skill_idx in enumerate(batch["correct_skill_idx"]):
            correct_label = torch.zeros(
                self.num_actions, dtype=torch.float32, device=self.device
            )
            correct_skill = EXPERT_UUIDS[int(correct_skill_idx)]
            if correct_skill == EXPERT_NULL_UUID:
                # Null action when an expert cannot be determined
                teacher_labels.append(correct_label)
                continue

            assert label_type in ["action", "mu", "std"]
            if label_type == "action":
                correct_label_arg = batch[correct_skill][idx]
            else:
                if correct_skill == EXPERT_NAV_UUID:
                    dist = self.moe.expert_nav_policy.distribution
                else:
                    # TODO add support for place expert
                    dist = self.moe.expert_gaze_policy.distribution
                if label_type == "mean":
                    correct_label_arg = dist.mean
                else:
                    correct_label_arg = dist.stddev

            # For MoE_res, correct action is ZEROS for correct expert
            if correct_skill == EXPERT_NAV_UUID:
                correct_label[-2:] = correct_label_arg
            else:
                correct_label[: len(correct_label_arg)] = correct_label_arg
            teacher_labels.append(correct_label)
        teacher_labels = torch.cat(
            [t.reshape(1, self.num_actions) for t in teacher_labels], dim=0
        )
        return teacher_labels

    def get_student_actions(self, batch, no_grad=False):
        conditional_grad = torch.no_grad if no_grad else ExitStack
        with conditional_grad():
            _, actions, _, self.rnn_hidden_states = self.moe.act(
                batch,
                self.rnn_hidden_states,
                self.prev_actions,
                self.masks,
                deterministic=True,
                update_masks=not self.teacher_forcing,
            )
        self.prev_actions.copy_(actions)
        if not no_grad:
            self.prev_actions = self.prev_actions.detach()
            self.rnn_hidden_states = self.rnn_hidden_states.detach()

        return actions

    def stepify_actions(self, actions, **kwargs):
        # Convert student actions into a dictionary for stepping envs
        step_actions = [
            self.moe.action_to_dict(act, index_env, **kwargs)
            for index_env, act in enumerate(actions.detach().cpu().unbind(0))
        ]

        return step_actions

    def log_prob_loss(self, batch, teacher_labels):
        _, action_log_probs, _, _ = self.moe.evaluate_actions(
            batch,
            self.rnn_hidden_states,
            self.prev_actions,
            self.masks,
            teacher_labels,
        )
        action_loss = -action_log_probs
        actions = self.get_student_actions(batch, no_grad=True)

        return actions, action_loss

    def mse_loss(self, batch, teacher_labels):
        actions = self.get_student_actions(batch, no_grad=False)

        mse_loss = F.mse_loss(actions, teacher_labels)
        self.action_mse_deq.append(mse_loss.detach().cpu().item())
        action_loss = mse_loss

        return actions, action_loss

    def mse_gaussian_loss(self, batch):
        actions = self.get_student_actions(batch, no_grad=False)
        student_mu = self.moe.distribution.mean
        student_std = self.moe.distribution.stddev

        # Extract teacher mu std from the observations
        teacher_mu = self.get_teacher_labels(batch, label_type="mu")
        teacher_std = self.get_teacher_labels(batch, label_type="std")

        mu_mse_loss = F.mse_loss(student_mu, teacher_mu.detach())
        std_mse_loss = F.mse_loss(student_std, teacher_std.detach())
        action_loss = mu_mse_loss + std_mse_loss

        return actions, action_loss

    def transform_observations(self, observations, masks):
        return self.moe.transform_obs(
            observations, masks, obs_transforms=self.obs_transforms
        )

    def init_envs(self, config):
        # Andrew's code for VectorEnvs
        import sys

        sys.path.insert(0, "./")
        from method.orp_policy_adapter import HabPolicy
        from orp_env_adapter import get_hab_args, get_hab_envs

        policy = baseline_registry.get_policy(self.policy_name)
        if issubclass(policy, HabPolicy):
            policy = policy(config)
        else:
            policy = None
        self.envs, _ = get_hab_envs(
            config,
            "./config.yaml",
            False,  # is_eval
            spec_gpu=config.TORCH_GPU_ID,
            setup_policy=policy,
        )

    def train(self):
        # HACK: Memory error when envs are loaded before policy
        tmp_config = self.config.clone()
        tmp_config.defrost()
        tmp_config.NUM_ENVIRONMENTS = 1
        tmp_config.NUM_PROCESSES = 1
        tmp_config.freeze()
        self.init_envs(tmp_config)

        # Set up policies
        self.setup_teacher_student(del_envs=True)
        # HACK: Memory error when envs are loaded before policy
        self.init_envs(self.config)

        # Set up optimizer
        optimizer = optim.Adam(self.get_model_params(), lr=self.sl_lr)

        # Set up tensorboard
        if self.tb_dir != "":
            print(f"Creating tensorboard at {self.tb_dir}...")
            os.makedirs(self.tb_dir, exist_ok=True)
            writer = SummaryWriter(self.tb_dir)
        else:
            writer = None

        # Start training
        observations = self.envs.reset()
        observations = self.transform_observations(observations, self.masks)
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        batch_num = 0
        action_loss = 0
        iterations = int(self.total_num_steps // self.envs.num_envs)
        for iteration in range(1, iterations + 1):
            # Step environment using *student* actions
            step_actions, loss = self.get_action_and_loss(batch)
            outputs = self.envs.step(step_actions)

            # Format consequent observations for the next iteration
            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            self.masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device=self.device,
            )
            observations = self.transform_observations(
                observations, self.masks
            )
            batch = batch_obs(observations, device=self.device)

            # Accumulate loss across batch
            action_loss += loss

            if self.config.DEBUG_BAD_EPS:
                frame = self.envs.render(mode="rgb_array")[0]
                self.frames.append(frame)

            # Get episode stats
            for idx, done in enumerate(dones):
                if done:
                    self.success_deq.append(infos[idx]["ep_success"])
                    if self.config.DEBUG_BAD_EPS:
                        if infos[idx]["ep_success"] < 1.0:
                            ep_txts = sorted(
                                glob.glob("/nethome/nyokoyama3/delme/*txt"),
                                key=osp.getmtime,
                            )
                            ep_id = int(
                                osp.basename(ep_txts[-2]).split(".")[0]
                            )
                            vid_name = f"/nethome/nyokoyama3/delme/{ep_id}.mp4"
                            vid = imageio.get_writer(vid_name, fps=30)
                            for im in self.frames:
                                vid.append_data(im)
                            vid.close()
                            logger.info(f"Video created: {vid_name}")
                        self.frames = []

            if iteration % self._batch_length == 0:
                # Run backpropagation using accumulated loss across batch
                optimizer.zero_grad()
                action_loss = action_loss.mean() / float(self._batch_length)
                if not torch.isnan(action_loss).any():
                    action_loss.backward()
                    optimizer.step()

                # Print stats
                batch_num += 1
                mean_succ = (
                    0 if not self.success_deq else np.mean(self.success_deq)
                )
                succ_deque_len = len(self.success_deq)
                logger.info(
                    f"iter: {iteration}\t"
                    f"batch_num: {batch_num}\t"
                    f"act_l: {action_loss.item():.4f}\t"
                    f"act_mse: {np.mean(self.action_mse_deq):.4f}\t"
                    f"mean_succ: {mean_succ:.4f} ({succ_deque_len})\t"
                )

                # Update tensorboard
                if writer is not None:
                    metrics_data = {"ep_success": mean_succ}
                    loss_data = {"action_loss": action_loss}
                    writer.add_scalars("metrics", metrics_data, batch_num)
                    writer.add_scalars("loss", loss_data, batch_num)

                # Reset loss
                action_loss = 0.0

                if batch_num % self.batches_per_save == 0:
                    # Save checkpoint
                    checkpoint = {
                        "state_dict": self.moe.state_dict(),
                        "config": self.config,
                        "iteration": iteration,
                        "batch": batch,
                    }
                    ckpt_id = int(batch_num / self.batches_per_save) - 1
                    filename = f"ckpt.{ckpt_id}_{batch_num}.pth"
                    ckpt_path = osp.join(self.checkpoint_folder, filename)
                    torch.save(checkpoint, ckpt_path)
                    print("Saved checkpoint:", ckpt_path)
                    if ckpt_id >= 50:
                        break

        self.envs.close()


@baseline_registry.register_trainer(name="bc_mask")
class BehavioralCloningMoeMask(BehavioralCloningMoe):
    def get_action_and_loss(self, batch):
        teacher_mask_labels = self.get_teacher_labels(batch)
        if self.teacher_forcing:
            # Use teacher masks
            self.moe.get_action_masks(teacher_mask_labels)
        actions = self.get_student_actions(batch)

        # Calculate loss. We only care about the mask outputs for behavioral
        # cloning, not the residuals.
        mask_actions = actions[:, -self.moe.num_masks :]
        action_loss = F.mse_loss(mask_actions, teacher_mask_labels)

        step_actions = self.stepify_actions(actions, use_residuals=False)

        # Not applicable for MoeMask; just say -1
        self.action_mse_deq.append(-1)

        return step_actions, action_loss

    def get_teacher_labels(self, batch, label_type="action"):
        """We only need to supervise 2 or 3 actions of the student: the masks
        that it outputs for each of the 2-3 experts"""

        expert_masks = {uuid: [] for uuid in EXPERT_UUIDS}
        # Iterates over each environment
        for idx, correct_skill_idx in enumerate(batch["correct_skill_idx"]):
            correct_skill = EXPERT_UUIDS[int(correct_skill_idx)]
            for uuid in expert_masks.keys():
                expert_masks[uuid].append(1 if uuid == correct_skill else -1)

        def get_mask_tensor(uuid):
            return torch.tensor(
                expert_masks[uuid], dtype=torch.float32
            ).reshape(self.envs.num_envs, 1)

        nav_masks = get_mask_tensor(EXPERT_NAV_UUID)
        gaze_masks = get_mask_tensor(EXPERT_GAZE_UUID)
        if self.moe.num_experts == 2:
            teacher_mask_labels = [nav_masks, gaze_masks]
        elif self.moe.num_experts == 3:
            place_masks = get_mask_tensor(EXPERT_PLACE_UUID)
            teacher_mask_labels = [nav_masks, gaze_masks, place_masks]
        else:
            raise NotImplementedError
        teacher_mask_labels = torch.cat(teacher_mask_labels, dim=1)
        teacher_mask_labels = teacher_mask_labels.to(self.device)

        return teacher_mask_labels


@baseline_registry.register_trainer(name="bc_mask_single")
class BehavioralCloningMoeMaskSingle(BehavioralCloningMoeMask):
    def get_teacher_labels(self, batch, label_type="action"):
        teacher_mask_labels = []
        uuid2bin = {
            EXPERT_NAV_UUID: -1.0,
            EXPERT_GAZE_UUID: 0.0 if self.moe.num_experts == 3 else 1.0,
            EXPERT_PLACE_UUID: 1.0,
            EXPERT_NULL_UUID: 0.0,  # Iffy
        }
        # Iterates over each environment
        for idx, correct_skill_idx in enumerate(batch["correct_skill_idx"]):
            correct_skill = EXPERT_UUIDS[int(correct_skill_idx)]
            teacher_mask_labels.append(uuid2bin[correct_skill])

        teacher_mask_labels = torch.tensor(
            teacher_mask_labels,
            dtype=torch.float32,
            device=self.device,
        ).reshape(self.envs.num_envs, 1)

        return teacher_mask_labels
