# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os.path as osp

import hydra
from omegaconf import OmegaConf, DictConfig
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
    return d

def merge_dicts(config):
    config = OmegaConf.to_container(config)
    config = OmegaConf.create(config)
    skills = ['nav', 'pick', 'place', 'open_close']

    for key in config.keys():
        if key not in skills:
            config.nav = OmegaConf.merge(config.nav, {key: config[key]})
            config.pick = OmegaConf.merge(config.pick, {key: config[key]})
            config.place = OmegaConf.merge(config.place, {key: config[key]})
            config.open_close = OmegaConf.merge(config.open_close, {key: config[key]})

    # Convert back to a regular dictionary if needed
    return OmegaConf.to_container(config)

def construct_config() -> DictConfig:
    GlobalHydra.instance().clear()
    rel_pth = os.path.relpath(CONFIGS_DIR, CONSTRUCT_CONFIG_DIR)
    initialize(config_path=rel_pth)
    cfg = compose(config_name="config")
    return prepend_experiments(cfg)

# @hydra.main(config_path=CONFIGS_DIR, config_name="config")
def construct_config(cfg: DictConfig):
    prepend_experiments(cfg)
    print(cfg)
    return cfg