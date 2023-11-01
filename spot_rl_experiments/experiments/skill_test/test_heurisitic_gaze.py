import sys

import numpy as np
from spot_rl.envs.nav_env import construct_config_for_nav
from spot_rl.envs.skill_manager import SpotSkillManager
from spot_rl.utils.heuristic_nav import heurisitic_object_search_and_navigation
from spot_rl.utils.utils import get_default_parser, map_user_input_to_boolean


def parse_arguments(args=sys.argv[1:]):
    parser = get_default_parser()
    parser.add_argument(
        "-pct",
        "--pick_target",
        help="input:string -> pick target poplated from Aria",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-ot",
        "--object_target",
        help="input:string -> target object to pick",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-pt",
        "--place_target",
        help="input:string -> place target",
        required=True,
        type=str,
    )

    args = parser.parse_args(args=args)

    return args


def get_in_position(spotskillmanager: SpotSkillManager):
    spotskillmanager.nav("prep")
    spotskillmanager.spot.sit()


if __name__ == "__main__":
    args = parse_arguments()
    pick_target, object_target, place_target = (
        args.pick_target,
        args.object_target,
        args.place_target,
    )
    pick_targets = {
        "kitchen": (
            3.8482142244527835,
            -3.4519528625906206,
            np.deg2rad(-89.14307672622927),
        ),
        "table": (5.56978867, 4.35661922, -0.001447959234375512),
    }
    assert pick_target in pick_targets
    x, y, theta = pick_targets[pick_target]
    nav_config = construct_config_for_nav()
    spotskillmanager = SpotSkillManager(nav_config)
    print(f"Original Nav Goal {x, y, np.degrees(theta)}")
    at_pick_position = spotskillmanager.heuristic_mobile_gaze(
        x, y, theta, object_target=object_target, pull_back=False
    )
    print(f"Spot was able to reach the goal ? {at_pick_position}")
    if at_pick_position:
        spotskillmanager.pick(object_target)
        spotskillmanager.nav(place_target)
        spotskillmanager.place(place_target)

    should_dock = map_user_input_to_boolean("Do you want to dock & exit ?")
    if should_dock:
        spotskillmanager.nav_controller.nav_env.disable_nav_by_hand()
        spotskillmanager.dock()
