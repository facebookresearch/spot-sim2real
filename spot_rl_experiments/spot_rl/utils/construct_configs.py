# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os.path as osp

from yacs.config import CfgNode as CN

this_dir = osp.dirname(osp.abspath(__file__))
spot_rl_dir = osp.join(osp.dirname(this_dir))
spot_rl_experiments_dir = osp.join(osp.dirname(spot_rl_dir))
configs_dir = osp.join(spot_rl_experiments_dir, "configs")
DEFAULT_CONFIG = osp.join(configs_dir, "config.yaml")


def construct_config(file_path=DEFAULT_CONFIG, opts=None):
    if opts is None:
        opts = []
    config = CN()
    config.set_new_allowed(True)
    config.merge_from_file(file_path)
    config.merge_from_list(opts)

    new_weights = {}
    for k, v in config.WEIGHTS.items():
        if not osp.isfile(v):
            new_v = osp.join(spot_rl_experiments_dir, v)
            if not osp.isfile(new_v):
                raise KeyError(f"Neither {v} nor {new_v} exist!")
            new_weights[k] = new_v
    config.WEIGHTS.update(new_weights)

    return config


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
