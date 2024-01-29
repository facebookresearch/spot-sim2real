# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import sys
from typing import Dict, List

from spot_rl.skills.atomic_skills import Pick
from spot_rl.utils.construct_configs import construct_config_for_gaze
from spot_rl.utils.utils import get_default_parser
from spot_wrapper.spot import Spot


def parse_arguments(args=sys.argv[1:]):
    parser = get_default_parser()
    parser.add_argument(
        "-t", "--target-object", type=str, help="name of the target object"
    )
    parser.add_argument(
        "-dp",
        "--dont_pick_up",
        action="store_true",
        help="robot should attempt pick but not actually pick",
    )
    parser.add_argument(
        "-ms", "--max_episode_steps", type=int, help="max episode steps"
    )
    parser.add_argument(
        "-mg",
        "--mobile_gaze",
        action="store_true",
        help="whether to use mobile gaze or static gaze",
    )
    args = parser.parse_args(args=args)

    if args.max_episode_steps is not None:
        args.max_episode_steps = int(args.max_episode_steps)
    return args


if __name__ == "__main__":
    spot = Spot("RealGazeEnv")
    args = parse_arguments()
    config = construct_config_for_gaze(
        opts=args.opts,
        dont_pick_up=args.dont_pick_up,
        max_episode_steps=args.max_episode_steps,
    )

    target_objects_list = []
    if args.target_object is not None:
        print(args.target_object)
        target_objects_list = [
            target
            for target in args.target_object.replace(" ,", ",")
            .replace(", ", ",")
            .split(",")
            if target.strip() is not None
        ]

    print(f"Target_objects list - {target_objects_list}")
    with spot.get_lease(hijack=True):
        spot.power_robot()
        gaze_controller = Pick(
            spot=spot, config=config, use_mobile_pick=args.mobile_gaze
        )
        gaze_results = []  # type: List[Dict]
        try:
            for target_object in target_objects_list:
                goal_dict = {
                    "target_object": target_object,
                    "take_user_input": True,
                }
                gaze_controller.execute(goal_dict=goal_dict)
                gaze_results.append(gaze_controller.get_most_recent_result_log())
        except Exception as e:
            print(f"Error encountered while picking - {e}")
            raise e
        finally:
            spot.shutdown(should_dock=False)

    print(f"Gaze results - {gaze_results}")
