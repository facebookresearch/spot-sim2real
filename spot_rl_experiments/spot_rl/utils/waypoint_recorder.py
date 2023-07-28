# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os.path as osp
import sys
from typing import Dict

import numpy as np
import ruamel.yaml
from spot_rl.utils.generate_place_goal import get_global_place_target
from spot_rl.utils.utils import get_default_parser
from spot_wrapper.spot import Spot

spot_rl_dir = osp.abspath(__file__)
for _ in range(3):
    spot_rl_dir = osp.dirname(spot_rl_dir)
WAYPOINT_YAML = osp.join(spot_rl_dir, "configs/waypoints.yaml")


def parse_arguments(args):
    parser = get_default_parser()
    parser.add_argument("-c", "--clutter", help="input:string -> clutter target name")
    parser.add_argument(
        "-p", "--place-target", help="input:string -> place target name"
    )
    parser.add_argument("-n", "--nav-only", help="input:string -> nav target name")
    parser.add_argument(
        "-x",
        "--create-file",
        action="store_true",
        help="input: not needed -> create a new waypoints.yaml file",
    )
    args = parser.parse_args(args=args)

    return args


class YamlHandler:
    """
    Class to handle reading and writing to yaml files

    How to use:
    1. Create a simple yaml file with the following format:

        place_targets:
            test_receptacle:
                - 3.0
                - 0.0
                - 0.8
        clutter:
            - test_receptacle
        clutter_amounts:
            test_receptacle: 1
        object_targets:
            0: [penguin, test_receptacle]
        nav_targets:
            dock:
                - 1.5
                - 0.0
                - 0.0
            test_receptacle:
                - 2.5
                - 0.0
                - 0.0

    2. Create an instance of this class
    3. Read the yaml file using the read_yaml method as a dict
    4. Modify the yaml_dict outside of this class object as needed
    5. Write the yaml_dict yaml_file using the write_yaml method with the created instance

    Example:
    yaml_handler = YamlHandler()
    yaml_dict = yaml_handler.read_yaml(waypoint_file=waypoint_file)
    yaml_dict["nav_targets"]["test_receptacle"] = [2.5, 0.0, 0.0]       # Modify the yaml_dict
    yaml_handler.write_yaml(waypoint_file=waypoint_file, yaml_dict=yaml_dict)
    """

    def __init__(self):
        pass

    def construct_yaml_dict(self):
        """
        Constructs and returns a simple yaml dict with "test_receptacle" as nav, place, and clutter target and dock as nav target
        """
        init_yaml_dict = """
place_targets: # i.e., where an object needs to be placed (x,y,z)
    test_receptacle:
        - 3.0
        - 0.0
        - 0.8
clutter: # i.e., where an object is currently placed
# <receptacle where clutter exists>
    - test_receptacle
clutter_amounts: # i.e., how much clutter exists in each receptacle
# <receptacle where clutter exists>: <number of objects in that receptacle>
    test_receptacle: 1
object_targets: # i.e., where an object belongs / needs to be placed
  # <Class_id>: [<object's name>, <which place_target it belongs to>]
    0: [penguin, test_receptacle]
nav_targets: # i.e., where the robot needs to navigate to (x,y,yaw)
    dock:
        - 1.5
        - 0.0
        - 0.0
    test_receptacle:
        - 2.5
        - 0.0
        - 0.0"""

        return init_yaml_dict

    @staticmethod
    def read_yaml(self, waypoint_file: str):
        """
        Read a yaml file and returns a dict

        Args:
            waypoint_file (str): path to yaml file

        Returns:
            yaml_dict (dict): Contens of the yaml file as a dict if it exists, else an contructs a new simple yaml_dict
        """
        yaml_dict = {}  # type: Dict
        yaml = ruamel.yaml.YAML()  # defaults to round-trip if no parameters given

        # Read yaml file if it exists
        if osp.exists(waypoint_file):
            with open(waypoint_file, "r") as f:
                print(
                    f"Reading waypoints from already existing waypoints.yaml file at {waypoint_file}"
                )
                yaml_dict = yaml.load(f.read())
        else:
            print(
                f"Creating a new waypoints dict as waypoints.yaml does not exist on path {waypoint_file}"
            )
            yaml_dict = yaml.load(self.construct_yaml_dict())

        return yaml_dict

    @staticmethod
    def write_yaml(self, waypoint_file: str, yaml_dict):
        """
        Write the yaml_dict into the yaml_file.
        If the yaml_file does not exist, it will be created.

        Args:
            waypoint_file (str): path to yaml file
            yaml_dict (dict): dict to be written to yaml file
        """
        with open(waypoint_file, "w+") as f:
            ruamel.yaml.dump(yaml_dict, f, Dumper=ruamel.yaml.RoundTripDumper)


class WaypointRecorder:
    """
    Class to record waypoints and clutter targets for the Spot robot

    How to use:
    1. Create an instance of this class
    2. Call the record_nav_target method with the nav_target_name as an argument (str)
    3. Call the record_clutter_target method with the clutter_target_name as an argument (str)
    4. Call the record_place_target method with the place_target_name as an argument (str)
    5. Call the save_yaml method to save the waypoints to the yaml file


    Args:
        spot (Spot): Spot robot object
        waypoint_file_path (str): path to yaml file to save waypoints to


    Example:
    waypoint_recorder = WaypointRecorder(spot=Spot)
    waypoint_recorder.record_nav_target("test_nav_target")
    waypoint_recorder.record_clutter_target("test_clutter_target")
    waypoint_recorder.record_place_target("test_place_target")
    waypoint_recorder.save_yaml()
    """

    def __init__(self, spot: Spot, waypoint_file_path: str = WAYPOINT_YAML):
        self.spot = spot

        # Local copy of waypoints.yaml which keeps getting updated as new waypoints are added
        self.waypoint_file = waypoint_file_path
        self.yaml_handler = YamlHandler()
        self.yaml_dict = {}  # type: Dict

    def init_yaml(self):
        """
        Initialize member variable `self.yaml_dict` with the contents of the yaml file as a dict if it is not initialized.
        """
        if self.yaml_dict == {}:
            self.yaml_dict = self.yaml_handler.read_yaml(
                waypoint_file=self.waypoint_file
            )

    def save_yaml(self):
        """
        Save the waypoints (self.yaml_dict) to the yaml file if it is not empty.
        It will overwrite the existing yaml file if it exists and will create a new one if it does not exist.
        """
        if self.yaml_dict == {}:
            print("No waypoints to save. Exiting...")
            return

        self.yaml_handler.write_yaml(self.waypoint_file, self.yaml_dict)
        print(
            f"Successfully saved(/overwrote) all waypoints to file at {self.waypoint_file}:\n"
        )

    def unmark_clutter(self, clutter_target_name: str):
        """
        INTERNAL METHOD:
        Unmark a waypoint as clutter if it is already marked.

        It is used internally by the `record_nav_target` method to unmark a waypoint as clutter if it is already marked.
        This is done to avoid cluttering the yaml file with duplicate clutter targets and also to update the waypoints.yaml
        file if a previously marked "clutter" is not marked as a "nav_target" anymore.

        Args:
            clutter_target_name (str): name of the waypoint to be unmarked as clutter
        """
        # Add clutter list if not present
        if "clutter" not in self.yaml_dict:
            self.yaml_dict["clutter"] = []
        # Remove waypoint from clutter list if it exists
        elif clutter_target_name in self.yaml_dict.get("clutter"):
            print(f"Unmarking {clutter_target_name} from clutter list")
            self.yaml_dict.get("clutter").remove(clutter_target_name)

    def mark_clutter(self, clutter_target_name: str):
        """
        INTERNAL METHOD:
        Mark a waypoint as clutter if it is not already marked.

        It is used internally by the `record_clutter_target` method to mark a waypoint as clutter if it is not already marked.

        Args:
            clutter_target_name (str): name of the waypoint to be marked as clutter
        """
        # Add clutter list if not present
        if "clutter" not in self.yaml_dict:
            self.yaml_dict["clutter"] = []

        # Add waypoint as clutter if it does not exist
        if clutter_target_name not in self.yaml_dict.get("clutter"):
            print(f"Marking {clutter_target_name} in clutter list")
            self.yaml_dict.get("clutter").append(clutter_target_name)

    def record_nav_target(self, nav_target_name: str):
        """
        Record a waypoint as a nav target

        It will also unmark the waypoint as clutter if it is already marked as clutter.
        If "nav_targets" does not exist, it will create a new "nav_targets" list initialized with the default "dock" waypoint
        and add the new waypoint to it.
        If the waypoint already exists, it will overwrite the existing waypoint data.

        Args:
            nav_target_name (str): name of the waypoint to be recorded as a nav target
        """
        # Initialize yaml_dict
        self.init_yaml()

        # Get current nav pose
        x, y, yaw = self.spot.get_xy_yaw()
        yaw_deg = np.rad2deg(yaw)
        nav_target = [float(x), float(y), float(yaw_deg)]

        # Unmark waypoint as clutter if it is already marked
        self.unmark_clutter(clutter_target_name=nav_target_name)

        # Add nav_targets list if not present
        if "nav_targets" not in self.yaml_dict:
            self.yaml_dict["nav_targets"] = {
                "dock": "[1.5, 0.0, 0.0]",
            }

        # Erase existing waypoint data if present
        if nav_target_name in self.yaml_dict.get("nav_targets"):
            print(
                f"Nav target for {nav_target_name} already exists as follows inside waypoints.yaml and will be overwritten."
            )
            print(
                f"old waypoint : {self.yaml_dict.get('nav_targets').get(nav_target_name)}"
            )
            input("Press Enter if you want to continue...")

        # Add new waypoint data
        self.yaml_dict.get("nav_targets").update({nav_target_name: nav_target})

    def record_clutter_target(self, clutter_target_name: str):
        """
        Record a waypoint as a clutter target

        It will initialize the member variable `self.yaml_dict` with appropriate content.
        It will mark the waypoint as nav target, thereby also clearing it from the clutter list if it is already marked as clutter
        It will mark the waypoint as clutter if not done already.
        It will add the waypoint to the "clutter_amounts" list if it does not exist, and will update its value to 1.

        Args:
            clutter_target_name (str): name of the waypoint to be recorded as a clutter target
        """
        # Initialize yaml_dict
        self.init_yaml()

        self.record_nav_target(clutter_target_name)

        # Mark waypoint as clutter
        self.mark_clutter(clutter_target_name=clutter_target_name)

        # Add clutter_amounts list if not present
        if "clutter_amounts" not in self.yaml_dict:
            self.yaml_dict["clutter_amounts"] = {}

        # Add waypoint as clutter_amounts if it does not exist
        if clutter_target_name not in self.yaml_dict.get("clutter_amounts"):
            self.yaml_dict["clutter_amounts"].update({clutter_target_name: 1})
            print(
                f"Added {clutter_target_name} in 'clutter_amounts' => ({clutter_target_name}:{self.yaml_dict.get('clutter_amounts').get(clutter_target_name)})"
            )
        else:
            print(
                f"{clutter_target_name} already exists in 'clutter_amounts' => ({clutter_target_name}:{self.yaml_dict.get('clutter_amounts').get(clutter_target_name)})"
            )

    def record_place_target(self, place_target_name: str):
        """
        Record a waypoint as a place target

        It will initialize the member variable `self.yaml_dict` with appropriate content
        It will mark the waypoint as nav target, thereby also clearing it from the clutter list if it is already marked as clutter
        It will add the waypoint to "place_targets" list if it does not exist, and will update its value to the current gripper position.

        Args:
            place_target_name (str): name of the waypoint to be recorded as a place target
        """
        # Initialize yaml_dict
        self.init_yaml()

        self.record_nav_target(place_target_name)

        # Get place target as current gripper position
        place_target = get_global_place_target(self.spot)

        # Add place_targets list if not present
        if "place_targets" not in self.yaml_dict:
            self.yaml_dict["place_targets"] = {}

        # Erase existing waypoint data if present
        if place_target_name in self.yaml_dict.get("place_targets"):
            print(
                f"Place target for {place_target_name} already exists as follows inside waypoints.yaml and will be overwritten."
            )
            print(
                f"old waypoint : {self.yaml_dict.get('place_targets').get(place_target_name)}"
            )

        # Add new place target data
        self.yaml_dict.get("place_targets").update({place_target_name: [*place_target]})


def main(spot: Spot):
    args = parse_arguments(args=sys.argv[1:])
    arg_bools = [args.clutter, args.place_target, args.nav_only, args.create_file]
    assert (
        len([i for i in arg_bools if i]) == 1
    ), "Must pass in either -c, -p, -n, or -x as an arg, and not more than one."

    # Create WaypointRecorder object with default waypoint file
    waypoint_recorder = WaypointRecorder(spot=spot)

    if args.create_file:
        waypoint_recorder.init_yaml()
    elif args.nav_only:
        waypoint_recorder.record_nav_target(args.nav_only)
    elif args.clutter:
        waypoint_recorder.record_clutter_target(args.clutter)
    elif args.place_target:
        waypoint_recorder.record_place_target(args.place_target)
    else:
        raise NotImplementedError

    waypoint_recorder.save_yaml()


if __name__ == "__main__":
    spot = Spot("WaypointRecorder")
    main(spot)
