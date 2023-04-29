#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import time
from typing import Any, ClassVar, Dict, List, Tuple, Union

import torch
from numpy import ndarray
from torch import Tensor

from habitat import Config, logger
from habitat.core.env import Env, RLEnv
from habitat.core.vector_env import VectorEnv
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ddppo.ddp_utils import SAVE_STATE, is_slurm_batch_job
from habitat_baselines.utils.common import (
    get_checkpoint_id,
    poll_checkpoint_folder,
)

sys.path.insert(0, "./")
import numpy as np

try:
    from orp.controllers.base_ctrls import *
    from orp.dataset import OrpNavDatasetV0
    from orp.env_aux import *
    from orp.sim.simulator import OrpSim
except:
    pass
import math
import subprocess


def get_logger(config, args, flush_secs):
    import sys

    sys.path.insert(0, "./")
    from method.orp_log_adapter import CustomLogger

    if config.write_tb:
        real_tb_dir = os.path.join(config.TENSORBOARD_DIR, args.prefix)
        config.defrost()
        # Inject the prefix into all of the filepaths
        config.VIDEO_DIR = os.path.join(config.VIDEO_DIR, args.prefix)
        config.CHECKPOINT_FOLDER = os.path.join(
            config.CHECKPOINT_FOLDER, args.prefix
        )
        if not os.path.exists(config.VIDEO_DIR):
            os.makedirs(config.VIDEO_DIR, exist_ok=True)
        if not os.path.exists(config.CHECKPOINT_FOLDER):
            os.makedirs(config.CHECKPOINT_FOLDER, exist_ok=True)

        config.freeze()
        if not os.path.exists(real_tb_dir):
            os.makedirs(real_tb_dir, exist_ok=True)

        ret = TensorboardWriter(real_tb_dir, flush_secs=flush_secs)
    else:
        # ret = CustomLogger(not config.no_wb, args, config)
        ret = CustomLogger(config.no_wb, args, config)
    out_cfg_path = os.path.join(config.CHECKPOINT_FOLDER, "cfg.txt")
    print("out path is ", out_cfg_path)
    with open(out_cfg_path, "w") as f:
        f.write(str(config))
        f.write("\n")
        f.write(str(args))
        try:
            out = subprocess.check_output(
                ["git", "-C", "../habitat-sim/.git", "rev-parse", "HEAD"]
            )
            print(f"Using HabSim version {out}")
            f.write("hab sim version " + str(out) + "\n")
        except:
            print("Could not find HabSim version")
    return ret


class BaseTrainer:
    r"""Generic trainer class that serves as a base template for more
    specific trainer classes like RL trainer, SLAM or imitation learner.
    Includes only the most basic functionality.
    """

    supported_tasks: ClassVar[List[str]]

    def train(self) -> None:
        raise NotImplementedError

    def _setup_eval_config(self, checkpoint_config: Config) -> Config:
        r"""Sets up and returns a merged config for evaluation. Config
            object saved from checkpoint is merged into config file specified
            at evaluation time with the following overwrite priority:
                  eval_opts > ckpt_opts > eval_cfg > ckpt_cfg
            If the saved config is outdated, only the eval config is returned.

        Args:
            checkpoint_config: saved config from checkpoint.

        Returns:
            Config: merged config for eval.
        """

        config = self.config.clone()

        ckpt_cmd_opts = checkpoint_config.CMD_TRAILING_OPTS
        eval_cmd_opts = config.CMD_TRAILING_OPTS

        try:
            config.merge_from_other_cfg(checkpoint_config)
            config.merge_from_other_cfg(self.config)
            config.merge_from_list(ckpt_cmd_opts)
            config.merge_from_list(eval_cmd_opts)
        except KeyError:
            logger.info("Saved config is outdated, using solely eval config")
            config = self.config.clone()
            config.merge_from_list(eval_cmd_opts)
        config.defrost()
        if config.TASK_CONFIG.DATASET.SPLIT == "train":
            config.TASK_CONFIG.DATASET.SPLIT = "val"
        config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = self.config.SENSORS
        config.freeze()

        return config

    def eval(self) -> None:
        r"""Main method of trainer evaluation. Calls _eval_checkpoint() that
        is specified in Trainer class that inherits from BaseRLTrainer
        or BaseILTrainer

        Returns:
            None
        """
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if "tensorboard" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.TENSORBOARD_DIR) > 0
            ), "Must specify a tensorboard directory for video display"
            os.makedirs(self.config.TENSORBOARD_DIR, exist_ok=True)
        if "disk" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.VIDEO_DIR) > 0
            ), "Must specify a directory for storing videos on disk"

        import sys

        sys.path.insert(0, "./")
        from method.orp_log_adapter import CustomLogger
        from orp_env_adapter import get_hab_args

        args = get_hab_args(self.config, "./config.yaml")

        if self.config.EVAL_CONCUR:
            found_f = None
            # 1 hour
            timeout_seconds = 60 * 60
            i = 0
            look_prefix = self.config.PREFIX.split("-")[-1]
            while found_f is None and i < timeout_seconds:
                for f in os.listdir(self.config.EVAL_CKPT_PATH_DIR):
                    if look_prefix in f and "eval" not in f:
                        found_f = f
                        break
                print("Could not find", look_prefix, "waiting...")
                time.sleep(2)
                i += 1
            if found_f is None:
                raise ValueError(
                    "Timed out waiting for checkpoint directory to be created"
                )
            self.config.defrost()
            self.config.EVAL_CKPT_PATH_DIR = os.path.join(
                self.config.EVAL_CKPT_PATH_DIR, found_f
            )
            self.config.freeze()
            print("Found out folder ", self.config.EVAL_CKPT_PATH_DIR)

        if self.config.EVAL.EMPTY:
            self._eval_checkpoint_nodes(
                self.config.EVAL_CKPT_PATH_DIR, checkpoint_index=1, args=args
            )
        elif os.path.isfile(self.config.EVAL_CKPT_PATH_DIR):
            # evaluate singe checkpoint
            proposed_index = get_checkpoint_id(self.config.EVAL_CKPT_PATH_DIR)
            if proposed_index is not None:
                ckpt_idx = proposed_index
            else:
                ckpt_idx = 1
            self._eval_checkpoint_nodes(
                self.config.EVAL_CKPT_PATH_DIR,
                checkpoint_index=ckpt_idx,
                args=args,
            )
        else:
            with get_logger(self.config, args, self.flush_secs) as writer:
                # evaluate multiple checkpoints in order
                prev_ckpt_ind = -1
                while True:
                    current_ckpt = None
                    while current_ckpt is None:
                        current_ckpt = poll_checkpoint_folder(
                            self.config.EVAL_CKPT_PATH_DIR, prev_ckpt_ind
                        )
                        if current_ckpt is not None and current_ckpt.endswith(
                            ".txt"
                        ):
                            prev_ckpt_ind += 1
                            current_ckpt = None
                        time.sleep(2)  # sleep for 2 secs before polling again

                    logger.info(f"=======current_ckpt: {current_ckpt}=======")

                    prev_ckpt_ind += 1
                    self._eval_checkpoint(
                        checkpoint_path=current_ckpt,
                        writer=writer,
                        checkpoint_index=prev_ckpt_ind,
                    )

    # pylint: disable=access-member-before-definition
    def _eval_checkpoint_nodes(self, checkpoint_path, checkpoint_index, args):
        import random
        import string

        from method.orp_log_adapter import CustomLogger

        if "EVAL_NODE" in self.config:
            if isinstance(self.config.EVAL_NODE, str):
                eval_nodes = eval(self.config.EVAL_NODE)
                assert isinstance(
                    eval_nodes, list
                ), "Eval nodes must be a list"
            elif isinstance(self.config.EVAL_NODE, list):
                eval_nodes = self.config.EVAL_NODE
            else:
                eval_nodes = [self.config.EVAL_NODE]
        else:
            eval_nodes = [None]

        orig_hab_set = self.config.hab_set
        orig_config = self.config.clone()

        # Compute before the main loop so no random seed affects this.
        rnd = random.Random(None)
        rnd_ident = "".join(
            rnd.sample(string.ascii_uppercase + string.digits, k=4)
        )

        base_prefix = args.prefix
        for eval_node in eval_nodes:
            if eval_node is not None and base_prefix != "debug":
                args.prefix = (
                    base_prefix + "_" + rnd_ident + "_" + str(eval_node)
                )
                print("Assigning eval prefix", args.prefix)
            config_copy = orig_config.clone()
            self.config = config_copy
            with get_logger(self.config, args, self.flush_secs) as writer:
                if eval_node is not None:
                    self.config.defrost()
                    hab_sets = orig_hab_set.split(",")
                    if len(hab_sets) == 0:
                        hab_sets = []
                    hab_sets.append("TASK_CONFIG.EVAL_NODE=%i" % eval_node)
                    self.config.hab_set = ",".join(hab_sets)
                    self.config.freeze()
                if (
                    "rlt_name" in self.config.RL.POLICY
                    and self.config.RL.POLICY.rlt_name == "NnHighLevelPolicy"
                    and os.path.isdir(self.config.pick_nn)
                ):
                    # Evaluate across all checkpoints in each of the
                    # directories
                    self._eval_rlt_multi_checkpoint(
                        checkpoint_path, writer, checkpoint_index
                    )
                else:
                    self._eval_checkpoint(
                        checkpoint_path, writer, checkpoint_index
                    )

    def _eval_rlt_multi_checkpoint(
        self, checkpoint_path, writer, checkpoint_index
    ):
        all_ckpts_dirs = {
            "pick_nn": self.config.pick_nn,
            "place_nn": self.config.place_nn,
            "open_fridge_nn": self.config.open_fridge_nn,
            "close_fridge_nn": self.config.close_fridge_nn,
            "open_cab_nn": self.config.open_cab_nn,
            "close_cab_nn": self.config.close_cab_nn,
            "nav_nn": self.config.nav_nn,
        }
        all_ckpts = {k: os.listdir(v) for k, v in all_ckpts_dirs.items()}

        # Remove all non checkpoint files
        all_ckpts = {
            k: [x for x in v if ".pth" in x and "resume-state" not in x]
            for k, v in all_ckpts.items()
        }

        # Sort checkpoint order
        all_ckpts = {
            k: sorted(v, key=lambda x: int(x.split(".")[-2]))
            for k, v in all_ckpts.items()
        }

        max_len = max([len(x) for x in all_ckpts.values()])
        EVAL_COUNT = 11
        fracs = [i / (EVAL_COUNT - 1) for i in range(EVAL_COUNT)]
        for frac in fracs:
            for k, ckpts in all_ckpts.items():
                use_idx = math.ceil(len(ckpts) * frac)
                self.config[k] = os.path.join(
                    all_ckpts_dirs[k], ckpts[use_idx]
                )
            self._eval_checkpoint(checkpoint_path, writer, checkpoint_index)

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        raise NotImplementedError

    def save_checkpoint(self, file_name) -> None:
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        raise NotImplementedError


class BaseRLTrainer(BaseTrainer):
    r"""Base trainer class for RL trainers. Future RL-specific
    methods should be hosted here.
    """
    device: torch.device  # type: ignore
    config: Config
    video_option: List[str]
    num_updates_done: int
    num_steps_done: int
    _flush_secs: int
    _last_checkpoint_percent: float

    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config is not None, "needs config file to initialize trainer"
        self.config = config
        self._flush_secs = 30
        self.num_updates_done = 0
        self.num_steps_done = 0
        self._last_checkpoint_percent = -1.0

        if config.NUM_UPDATES != -1 and config.TOTAL_NUM_STEPS != -1:
            raise RuntimeError(
                "NUM_UPDATES and TOTAL_NUM_STEPS are both specified.  One must be -1.\n"
                " NUM_UPDATES: {} TOTAL_NUM_STEPS: {}".format(
                    config.NUM_UPDATES, config.TOTAL_NUM_STEPS
                )
            )

        if config.NUM_UPDATES == -1 and config.TOTAL_NUM_STEPS == -1:
            raise RuntimeError(
                "One of NUM_UPDATES and TOTAL_NUM_STEPS must be specified.\n"
                " NUM_UPDATES: {} TOTAL_NUM_STEPS: {}".format(
                    config.NUM_UPDATES, config.TOTAL_NUM_STEPS
                )
            )

        if config.NUM_CHECKPOINTS != -1 and config.CHECKPOINT_INTERVAL != -1:
            raise RuntimeError(
                "NUM_CHECKPOINTS and CHECKPOINT_INTERVAL are both specified."
                "  One must be -1.\n"
                " NUM_CHECKPOINTS: {} CHECKPOINT_INTERVAL: {}".format(
                    config.NUM_CHECKPOINTS, config.CHECKPOINT_INTERVAL
                )
            )

        if config.NUM_CHECKPOINTS == -1 and config.CHECKPOINT_INTERVAL == -1:
            raise RuntimeError(
                "One of NUM_CHECKPOINTS and CHECKPOINT_INTERVAL must be specified"
                " NUM_CHECKPOINTS: {} CHECKPOINT_INTERVAL: {}".format(
                    config.NUM_CHECKPOINTS, config.CHECKPOINT_INTERVAL
                )
            )

    def percent_done(self) -> float:
        if self.config.NUM_UPDATES != -1:
            return self.num_updates_done / self.config.NUM_UPDATES
        else:
            return self.num_steps_done / self.config.TOTAL_NUM_STEPS

    def is_done(self) -> bool:
        return self.percent_done() >= 1.0

    def should_checkpoint(self) -> bool:
        needs_checkpoint = False
        if self.config.NUM_CHECKPOINTS != -1:
            checkpoint_every = 1 / self.config.NUM_CHECKPOINTS
            if (
                self._last_checkpoint_percent + checkpoint_every
                < self.percent_done()
            ):
                needs_checkpoint = True
                self._last_checkpoint_percent = self.percent_done()
        else:
            needs_checkpoint = (
                self.num_updates_done % self.config.CHECKPOINT_INTERVAL
            ) == 0

        return needs_checkpoint

    def _should_save_resume_state(self) -> bool:
        return SAVE_STATE.is_set() or (
            (
                not self.config.RL.preemption.save_state_batch_only
                or is_slurm_batch_job()
            )
            and (
                (
                    int(self.num_updates_done + 1)
                    % self.config.RL.preemption.save_resume_state_interval
                )
                == 0
            )
        )

    @property
    def flush_secs(self):
        return self._flush_secs

    @flush_secs.setter
    def flush_secs(self, value: int):
        self._flush_secs = value

    def train(self) -> None:
        raise NotImplementedError

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint. Trainer algorithms should
        implement this.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        raise NotImplementedError

    def save_checkpoint(self, file_name) -> None:
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        raise NotImplementedError

    @staticmethod
    def _pause_envs(
        envs_to_pause: List[int],
        envs: Union[VectorEnv, RLEnv, Env],
        test_recurrent_hidden_states: Tensor,
        not_done_masks: Tensor,
        current_episode_reward: Tensor,
        prev_actions: Tensor,
        batch: Dict[str, Tensor],
        rgb_frames: Union[List[List[Any]], List[List[ndarray]]],
    ) -> Tuple[
        Union[VectorEnv, RLEnv, Env],
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Dict[str, Tensor],
        List[List[Any]],
    ]:
        # pausing self.envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            # indexing along the batch dimensions
            test_recurrent_hidden_states = test_recurrent_hidden_states[
                state_index
            ]
            not_done_masks = not_done_masks[state_index]
            current_episode_reward = current_episode_reward[state_index]
            prev_actions = prev_actions[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

            rgb_frames = [rgb_frames[i] for i in state_index]

        return (
            envs,
            test_recurrent_hidden_states,
            not_done_masks,
            current_episode_reward,
            prev_actions,
            batch,
            rgb_frames,
        )
