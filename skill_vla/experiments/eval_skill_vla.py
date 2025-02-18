import math
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch
from envs.pick_env import SpotPickEnv
from PIL import Image
from spot_wrapper.spot import Spot
from utils.utils import construct_config
from vla_policy import VLAPolicy


def main(spot):
    config = construct_config(
        "/home/tushar/Desktop/spot-vla/spot-sim2real/skill_vla/configs/config.yaml"
    )
    policy = VLAPolicy(config.VLA_CONFIGS, config.DEVICE)

    env = SpotPickEnv(config, spot)
    env.power_robot()

    observations = env.reset()
    done = False
    policy.reset()
    while not done:
        print(f"=================== step # {env.num_steps} ===================")
        action = policy.act(observations)[0]
        action_dict = {
            "arm_action": action[:4],
            "base_action": action[4:6],
            "grasp_action": action[-1],
        }
        print("action_dict: ", action_dict)
        observations, _, done, _ = env.step(action_dict)
        if done:
            time.sleep(100)


if __name__ == "__main__":
    spot = Spot("VLA")
    with spot.get_lease(hijack=True):
        main(spot)
