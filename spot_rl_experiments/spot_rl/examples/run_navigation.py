# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import sys
import time

from spot_rl.skills.atomic_skills import Navigation
from spot_rl.utils.construct_configs import construct_config_for_nav
from spot_rl.utils.json_helpers import save_json_file
from spot_rl.utils.utils import (
    get_default_parser,
    get_waypoint_yaml,
    nav_target_from_waypoint,
)
from spot_wrapper.spot import Spot


def parse_arguments(args=sys.argv[1:]):
    parser = get_default_parser()
    parser.add_argument(
        "-g", "--goal", help="input:string -> goal x,y,theta in meters and radians"
    )
    parser.add_argument(
        "-w",
        "--waypoints",
        help="input:string -> nav target waypoints (comma seperated) where robot needs to navigate to",
    )
    parser.add_argument(
        "-d",
        "--dock",
        action="store_true",
        help="make the robot dock after finishing navigation to all waypoints",
    )
    parser.add_argument(
        "-rt",
        "--record_trajectories",
        action="store_true",
        help="record robot's trajectories while navigating to all waypoints",
    )
    parser.add_argument(
        "-stp",
        "--save_trajectories_path",
        help="input:string -> path to save robot's trajectory",
    )
    args = parser.parse_args(args=args)

    return args


if __name__ == "__main__":
    args = parse_arguments()
    config = construct_config_for_nav(opts=args.opts)
    waypoints_yaml_dict = get_waypoint_yaml()

    # Get nav_targets_list (list) to go to
    nav_targets_list = None
    if args.waypoints is not None:
        waypoints = [
            waypoint
            for waypoint in args.waypoints.replace(" ,", ",")
            .replace(", ", ",")
            .split(",")
            if waypoint.strip() is not None
        ]
        nav_targets_list = [
            nav_target_from_waypoint(waypoint, waypoints_yaml_dict)
            for waypoint in waypoints
        ]
    else:
        assert args.goal is not None
        goal_x, goal_y, goal_heading = [float(i) for i in args.goal.split(",")]
        nav_targets_list = [(goal_x, goal_y, goal_heading)]

    # Default value for `args.save_trajectories_path` is None. Raise error to ask for correct location
    if (args.save_trajectories_path is not None) and (
        not os.path.isdir(args.save_trajectories_path)
    ):
        raise Exception(
            f"The path for saving trajectories at {args.save_trajectories_path} either not specified or incorrect. Please provide a correct path"
        )

    record_trajectories = (args.record_trajectories) or (
        args.save_trajectories_path is not None
    )

    spot = Spot("RealNavEnv")
    with spot.get_lease(hijack=True):
        spot.power_robot()
        nav_controller = Navigation(
            spot=spot, config=config, record_robot_trajectories=record_trajectories
        )
        try:
            robot_trajectories = nav_controller.execute_nav(
                nav_targets_list=nav_targets_list
            )
        finally:
            spot.shutdown(should_dock=args.dock)

        if args.save_trajectories_path is not None:
            # Ensure the folder name ends with a trailing slash
            storage_dir = os.path.join(args.save_trajectories_path, "")

            # save dictionary to traj.json file
            file_name = "nav_" + (
                time.strftime("%b-%d-%Y_%H%M", time.localtime()) + ".json"
            )
            file_path = storage_dir + file_name
            save_json_file(file_path=file_path, data=robot_trajectories)
