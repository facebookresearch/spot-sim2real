import argparse
import os.path as osp

import numpy as np
import ruamel.yaml
from spot_wrapper.spot import Spot

from spot_rl.utils.generate_place_goal import get_global_place_target

spot_rl_dir = osp.abspath(__file__)
for _ in range(3):
    spot_rl_dir = osp.dirname(spot_rl_dir)
WAYPOINT_YAML = osp.join(spot_rl_dir, "configs/waypoints.yaml")


def main(spot: Spot):
    parser = argparse.ArgumentParser()
    parser.add_argument("waypoint_name")
    parser.add_argument("-c", "--clutter", action="store_true")
    parser.add_argument("-p", "--place-target", action="store_true")
    parser.add_argument("-n", "--nav-only", action="store_true")
    args = parser.parse_args()

    arg_bools = [args.clutter, args.place_target, args.nav_only]
    assert (
        len([i for i in arg_bools if i]) == 1
    ), "Must pass in either -c, -p, or -n as an arg, and not more than one."

    # Get current nav pose
    x, y, yaw = spot.get_xy_yaw()
    yaw_deg = np.rad2deg(yaw)

    # Get current gripper position if place target specified
    if args.place_target:
        place_target = get_global_place_target(spot)
    else:
        place_target = None

    yaml = ruamel.yaml.YAML()  # defaults to round-trip if no parameters given
    with open(WAYPOINT_YAML) as f:
        yaml_dict = yaml.load(f.read())

    # Erase existing waypoint data if present
    if args.waypoint_name in yaml_dict:
        print(f"Following existing info for {args.waypoint_name} will be overwritten:")
        print(f"\twaypoint:", yaml_dict[args.waypoint_name])
        if args.waypoint_name in yaml_dict["place_targets"]:
            print(
                f"\tplace_targets[{args.waypoint_name}]:",
                yaml_dict["place_targets"][args.waypoint_name],
            )
        if args.waypoint_name in yaml_dict["clutter"]:
            print(f"\tUn-marking {args.waypoint_name} from clutter list")

    # Add waypoint
    yaml_dict[args.waypoint_name] = [float(x), float(y), float(yaw_deg)]

    yaml_dict["clutter"] = [i for i in yaml_dict["clutter"] if i != args.waypoint_name]

    # Add waypoint as clutter or as a place_target
    if args.clutter:
        if "clutter" not in yaml_dict:
            yaml_dict["clutter"] = []
        if args.waypoint_name not in yaml_dict["clutter"]:
            yaml_dict["clutter"].append(args.waypoint_name)
    elif args.place_target:
        if "place_targets" not in yaml_dict:
            yaml_dict["place_targets"] = {}
        yaml_dict["place_targets"][args.waypoint_name] = [*place_target]

    with open(WAYPOINT_YAML, "w") as f:
        yaml.dump(yaml_dict, f)

    print(f"Successfully saved waypoint {args.waypoint_name}:")
    print(f"\twaypoint:", yaml_dict[args.waypoint_name])
    if args.clutter:
        print("\tMarked as clutter.")
    elif args.place_target:
        print(
            f"\tplace_targets[{args.waypoint_name}]:",
            yaml_dict["place_targets"][args.waypoint_name],
        )


if __name__ == "__main__":
    spot = Spot("WaypointRecorder")
    main(spot)
