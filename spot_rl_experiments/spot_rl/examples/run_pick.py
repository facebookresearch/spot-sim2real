# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import sys

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
        gaze_controller = Pick(spot=spot, config=config)
        try:
            gaze_result = gaze_controller.execute_pick(
                target_objects_list, take_user_input=True
            )
            print(gaze_result)
        finally:
            spot.shutdown(should_dock=False)
