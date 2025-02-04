import math
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch
from einops import rearrange
from envs.pick_env import SpotPickEnv
from omegaconf import OmegaConf
from PIL import Image
from spot_rl.utils.utils import construct_config
from spot_wrapper.spot import Spot
from vla_policy import VLAPolicy


def main(spot):
    config = construct_config([])
    policy = VLAPolicy(config.VLA_CONFIGS, config.DEVICE)

    env = SpotPickEnv(config, spot)
    env.power_robot()

    observations = env.reset()
    done = False
    policy.reset()
    while not done:
        action = policy.act(observations)
        observations, _, done, _ = env.step(base_action=action)


if __name__ == "__main__":
    spot = Spot("VLA")
    with spot.get_lease(hijack=True):
        main(spot)
