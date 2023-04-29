#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import os.path as osp
import random
import time

import numpy as np
import torch

import habitat_baselines.bc.bc_trainer
from habitat.config import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def execute_exp(config: Config, run_type: str) -> None:
    r"""This function runs the specified config with the specified runtype
    Args:
    config: Habitat.config
    runtype: str {train or eval}
    """
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    if config.FORCE_TORCH_SINGLE_THREADED and torch.cuda.is_available():
        torch.set_num_threads(1)

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config, run_type)

    if run_type == "train":
        return trainer.train()
    elif run_type == "eval":
        return trainer.eval()


PATHS_TO_JUNK = {
    "LOG_FILE": "junk/train.log",
    "CHECKPOINT_FOLDER": "junk/",
    "TENSORBOARD_DIR": "junk/",
    "VIDEO_DIR": "junk/",
}


def run_exp(exp_config: str, run_type: str, opts=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    if "BASE_DIR" in opts:
        base_dir_idx = opts.index("BASE_DIR")
        base_dir = opts[base_dir_idx + 1]
        for _ in range(2):
            opts.pop(base_dir_idx)

        assert "PREFIX" in opts
        opts[opts.index("PREFIX") + 1] += f"_{int(time.time())}"
        prefix = opts[opts.index("PREFIX") + 1]
        slurm_dir = osp.join(base_dir, "slurm_files")
        if osp.isdir(slurm_dir):
            prefix = osp.join("slurm_files", prefix)

        sub_paths = {
            "LOG_FILE": prefix + ".log",
            "CHECKPOINT_FOLDER": "checkpoints",
            "TENSORBOARD_DIR": "tb",
            "VIDEO_DIR": "videos",
        }

        for k, v in sub_paths.items():
            if k not in opts:
                opts.extend([k, osp.join(base_dir, v)])

    if "EVAL_CKPT_PATH_DIR" in opts:
        eval_ckpt_idx = opts.index("EVAL_CKPT_PATH_DIR")
        orig_eval_ckpt = opts[eval_ckpt_idx + 1]
        is_file = orig_eval_ckpt.endswith(".pth")
        if is_file:
            ckpt_dir = osp.dirname(orig_eval_ckpt)
            matches = glob.glob(ckpt_dir + "*")
            assert len(matches) == 1
            eval_ckpt = osp.join(matches[0], osp.basename(orig_eval_ckpt))
            assert osp.isfile(eval_ckpt)
        else:
            while orig_eval_ckpt.endswith("/"):
                orig_eval_ckpt = orig_eval_ckpt[:-1]
            matches = glob.glob(orig_eval_ckpt + "*")
            assert len(matches) == 1
            eval_ckpt = matches[0]
            assert osp.isdir(eval_ckpt)
        opts[eval_ckpt_idx + 1] = eval_ckpt
        print("EVAL_CKPT_PATH_DIR:", opts[eval_ckpt_idx + 1])

    if "JUNK" in opts:
        junk_idx = opts.index("JUNK")
        for _ in range(2):
            opts.pop(junk_idx)
        for k, v in PATHS_TO_JUNK.items():
            if k in opts:
                opts[opts.index(k) + 1] = v
            else:
                opts.extend([k, v])

    if "RL.POLICY.fuse_states" in opts:
        i = opts.index("RL.POLICY.fuse_states")
        opts[i + 1] = opts[i + 1].split(",")
        if len(opts[i + 1]) == 1 and opts[i + 1][0] == "":
            opts[i + 1] = []

    config = get_config(exp_config, opts)
    execute_exp(config, run_type)


if __name__ == "__main__":
    main()
