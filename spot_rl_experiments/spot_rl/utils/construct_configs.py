# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os.path as osp

import hydra
from omegaconf import DictConfig
import re

THIS_DIR = osp.dirname(osp.abspath(__file__))
SPOT_RL_DIR = osp.join(osp.dirname(THIS_DIR))
SPOT_RL_EXPERIMENTS_DIR = osp.join(osp.dirname(SPOT_RL_DIR))
CONFIGS_DIR = osp.join(SPOT_RL_EXPERIMENTS_DIR, "configs")
DEFAULT_CONFIG = osp.join(CONFIGS_DIR, "config.yaml")

def prepend_experiments(d):
    for key, value in d.items():
        if isinstance(value, DictConfig):
            prepend_experiments(value)
        elif isinstance(value, str) and (('.pth') in value or ('.torchscript') in value):
            full_path = osp.join(SPOT_RL_EXPERIMENTS_DIR, value)
            # if not osp.isfile(value):
                # raise KeyError(f"Neither {value} nor {full_path} exist!")
            d[key] = full_path
@hydra.main(config_path=CONFIGS_DIR, config_name="config")
def construct_config(cfg: DictConfig):
    prepend_experiments(cfg)
    print(cfg)
    return cfg

def construct_config_for_nav(file_path=None, opts=[]):
    """
    Constructs and updates the config for nav

    Args:
        file_path (str): Path to the config file
        opts (list): List of options to update the config

    Returns:
        config (Config): Updated config object
    """
    config = None
    if file_path is None:
        config = construct_config(opts=opts)
    else:
        config = construct_config(file_path=file_path, opts=opts)

    # Don't need gripper camera for Nav
    config.USE_MRCNN = False
    return config


def construct_config_for_gaze(
    file_path=None, opts=[], dont_pick_up=False, max_episode_steps=None
):
    """
    Constructs and updates the config for gaze

    Args:
        file_path (str): Path to the config file
        opts (list): List of options to update the config

    Returns:
        config (Config): Updated config object
    """
    config = None
    if file_path is None:
        config = construct_config(opts=opts)
    else:
        config = construct_config(file_path=file_path, opts=opts)

    # Don't need head cameras for Gaze
    config.USE_HEAD_CAMERA = False

    # Update the config based on the input argument
    if dont_pick_up != config.DONT_PICK_UP:
        print(
            f"WARNING: Overriding dont_pick_up in config from {config.DONT_PICK_UP} to {dont_pick_up}"
        )
        config.DONT_PICK_UP = dont_pick_up

    # Update max episode steps based on the input argument
    if max_episode_steps is not None:
        print(
            f"WARNING: Overriding max_espisode_steps in config from {config.MAX_EPISODE_STEPS} to {max_episode_steps}"
        )
        config.MAX_EPISODE_STEPS = max_episode_steps
    return config


def construct_config_for_place(file_path=None, opts=[]):
    config = None
    if file_path is None:
        config = construct_config(opts=opts)
    else:
        config = construct_config(file_path=file_path, opts=opts)

    # Don't need cameras for Place
    config.USE_HEAD_CAMERA = False
    config.USE_MRCNN = False

    return config


def construct_config_for_semantic_place(file_path=None, opts=[]):
    config = None
    if file_path is None:
        config = construct_config(opts=opts)
    else:
        config = construct_config(file_path=file_path, opts=opts)

    # Don't need cameras for Place
    config.USE_HEAD_CAMERA = False
    config.USE_MRCNN = True

    return config


def construct_config_for_open_close_drawer(file_path=None, opts=[]):
    """
    Constructs and updates the config for open close drawer

    Args:
        file_path (str): Path to the config file
        opts (list): List of options to update the config

    Returns:
        config (Config): Updated config object
    """
    config = None
    if file_path is None:
        config = construct_config(opts=opts)
    else:
        config = construct_config(file_path=file_path, opts=opts)

    # Don't need head cameras for Gaze
    config.USE_HEAD_CAMERA = False

    return config
