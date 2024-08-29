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

@hydra.main(config_path=CONFIGS_DIR, config_name="config")
def construct_config(cfg: DictConfig):
    prepend_experiments(cfg)
    print(cfg)
    return cfg