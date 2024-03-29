# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import time
from typing import Any, Dict

from spot_rl.envs.gaze_env import SpotGazeEnv
from spot_rl.real_policy import GazePolicy
from spot_rl.utils.utils import (
    construct_config,
    get_default_parser,
    get_waypoint_yaml,
    nav_target_from_waypoint,
    object_id_to_object_name,
)
from spot_wrapper.spot import Spot


def main(spot):
    parser = get_default_parser()
    args = parser.parse_args()
    config = construct_config(opts=args.opts)
    waypoint_yaml = get_waypoint_yaml()

    env = SpotGazeEnv(config, spot, mask_rcnn_weights=config.WEIGHTS.MRCNN)
    spot.power_robot()
    policy = GazePolicy(config.WEIGHTS.GAZE, device=config.DEVICE, config=config)
    for target_id in range(1, 9):
        goal_x, goal_y, goal_heading = nav_target_from_waypoint(
            "white_box", waypoints_yaml=waypoint_yaml
        )
        spot.set_base_position(
            x_pos=goal_x, y_pos=goal_y, yaw=goal_heading, end_time=100, blocking=True
        )
        time.sleep(4)
        policy.reset()
        observations = env.reset(target_obj_id=target_id)
        done = False
        env.say("Looking for", object_id_to_object_name(target_id))
        action_dict = {
            "base_action": None,
            "arm_action": None,
        }  # type: Dict[str, Any]
        while not done:
            action_dict["arm_action"] = policy.act(observations)
            observations, _, done, _ = env.step(action_dict=action_dict)


if __name__ == "__main__":
    spot = Spot("MultipleGazeEnv")
    with spot.get_lease(hijack=True):
        try:
            main(spot)
        finally:
            spot.power_off()
