# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import sys
from typing import Dict, List

from spot_rl.skills.atomic_skills import Place
from spot_rl.utils.construct_configs import construct_config_for_place
from spot_rl.utils.utils import (
    get_default_parser,
    get_waypoint_yaml,
    place_target_from_waypoint,
)
from spot_wrapper.spot import Spot


def parse_arguments(args=sys.argv[1:]):
    parser = get_default_parser()
    parser.add_argument(
        "-p",
        "--place_target",
        help="input:float,float,float -> place target x,y,z in meters from the global frame (or robot's base frame if -l is specified)",
    )
    parser.add_argument(
        "-w",
        "--waypoints",
        type=str,
        help="input:string -> place target waypoints (comma separated place_target names) where robot needs to place the object",
    )
    parser.add_argument(
        "-l",
        "--target_is_local",
        action="store_true",
        help="whether the place target specified is in the local frame of the robot",
    )
    parser.add_argument(
        "-up",
        "--use_policies",
        action="store_true",
        help="Whether to use policies or use BD API for place",
    )
    args = parser.parse_args(args=args)

    return args


if __name__ == "__main__":
    args = parse_arguments()
    config = construct_config_for_place(opts=args.opts)
    waypoints_yaml_dict = get_waypoint_yaml()

    # Get place_target_list (list) to go and pick from
    place_target_list = None
    if args.waypoints is not None:
        waypoints = [
            waypoint
            for waypoint in args.waypoints.replace(" ,", ",")
            .replace(", ", ",")
            .split(",")
            if waypoint.strip() is not None
        ]
        place_target_list = [
            place_target_from_waypoint(waypoint, waypoints_yaml_dict)
            for waypoint in waypoints
        ]
    else:
        assert args.place_target is not None
        place_target_list = [[float(i) for i in args.place_target.split(",")]]

    spot = Spot("RealPlaceEnv")
    with spot.get_lease(hijack=True):
        spot.power_robot()
        place_controller = Place(spot, config, use_policies=args.use_policies)
        place_results = []  # type: List[Dict]
        try:
            for place_target in place_target_list:
                print(f"Place target - {place_target}")
                goal_dict = {
                    "place_target": place_target,
                    "is_local": args.target_is_local,
                }
                if args.use_policies:
                    place_controller.execute(goal_dict=goal_dict)
                place_results.append(place_controller.get_most_recent_result_log())
        except Exception as e:
            print(f"Error encountered while placing - {e}")
            raise e
        finally:
            spot.shutdown(should_dock=False)

    print(f"Place results - {place_results}")
