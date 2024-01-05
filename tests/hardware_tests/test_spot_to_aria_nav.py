import sys

import numpy as np
from spot_rl.envs.nav_env import construct_config_for_nav
from spot_rl.envs.skill_manager import SpotSkillManager
from spot_rl.utils.utils import get_default_parser, map_user_input_to_boolean


def parse_arguments(args=sys.argv[1:]):
    parser = get_default_parser()
    # parser.add_argument(
    #     "-g",
    #     "--goal",
    #     help="input:string -> goal x,y,theta in meters and radians obtained from aria",
    #     required=False,
    #     type=str,
    # )
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

    # assert args.goal is not None, "Please provide x,y,theta in meters & radians that you get from aria"
    # goal_str = str(args.goal).strip().split(",")
    # assert len(goal_str) == 3, f"Goal str len was supposed to be 3 but found {len(goal_str)}"
    # x, y, theta = [float(s) for s in goal_str]
    # ["toy_lion", "bottle", "ball", "can", "cereal_box", "cereal_box", "toy_penguin", "tissue_roll"]

    # x, y, theta = (
    #     3.97,
    #     -3.65,
    #     -1.1429722791562886,
    # )  # , np.deg2rad(-96.2389683367895)
    # print(f"Original Nav Goal {x, y, np.degrees(theta)}")
    # x, y = push_forward_point_along_theta_by_offset(x, y, theta, 0.3)
    # print(f"Nav goal after pushing forward by 0.3m {x,y}")
    # Today's kitchen waypoints 3.99693434, -4.10032391, -1.966045713659559
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
    # get_in_position(spotskillmanager)
    # spotskillmanager.nav("dining_table")
    # spotskillmanager.place("dining_table")
    print(f"Original Nav Goal {x, y, np.degrees(theta)}")
    at_pick_position = spotskillmanager.nav_mobile_hueristic(
        x, y, theta, object_target=object_target, pull_back=True
    )
    print(f"Spot was able to reach the goal ? {at_pick_position}")
    if at_pick_position:
        spotskillmanager.pick(object_target)
        # backup_steps = spotskillmanager.nav_controller.nav_env.max_episode_steps
        # spotskillmanager.nav_controller.nav_env.max_episode_steps = 50
        spotskillmanager.nav(place_target)
        # spotskillmanager.nav_controller.nav_env.max_episode_steps = backup_steps
        spotskillmanager.place(place_target)

    should_dock = map_user_input_to_boolean("Do you want to dock & exit ?")
    if should_dock:
        spotskillmanager.nav_controller.nav_env.disable_nav_goal_change()
        # spotskillmanager.nav("prep")
        # spotskillmanager.spot.set_arm_joint_positions(spotskillmanager.nav_config.INITIAL_ARM_JOINT_ANGLES)
        # spotskillmanager.spot.sit()
        spotskillmanager.dock()
