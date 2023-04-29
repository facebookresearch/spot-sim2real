import argparse
import time

from spot_wrapper.spot import Spot
from spot_wrapper.utils import say

from spot_rl.envs.gaze_env import run_env
from spot_rl.utils.utils import construct_config

names = ["ball", "penguin", "rubiks_cube", "lion", "toy_car", "yellow_truck"]


def main(spot, bd=False):
    config = construct_config()
    config.DONT_PICK_UP = True
    config.OBJECT_LOCK_ON_NEEDED = 5
    # config.CTRL_HZ = 2.0
    config.TERMINATE_ON_GRASP = True
    config.FORGET_TARGET_OBJECT_STEPS = 1000000
    if bd:
        config.GRASP_EVERY_STEP = True
        config.MAX_JOINT_MOVEMENT = 0.0  # freeze arm
        config.MAX_EPISODE_STEPS = 20
    else:
        config.MAX_EPISODE_STEPS = 150
    orig_pos = None
    for _ in range(3):
        for name in names:
            say("Targeting " + name)
            time.sleep(2)
            orig_pos = run_env(spot, config, target_obj_id=name, orig_pos=orig_pos)
            say("Episode over")
            time.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bd", action="store_true")
    args = parser.parse_args()
    spot = Spot("RealGazeEnv")
    with spot.get_lease(hijack=True):
        main(spot, bd=args.bd)
