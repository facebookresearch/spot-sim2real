# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os.path as osp
from collections import OrderedDict

import numpy as np
import yaml
from yacs.config import CfgNode as CN

this_dir = osp.dirname(osp.abspath(__file__))
spot_rl_dir = osp.join(osp.dirname(this_dir))
spot_rl_experiments_dir = osp.join(osp.dirname(spot_rl_dir))
configs_dir = osp.join(spot_rl_experiments_dir, "configs")
DEFAULT_CONFIG = osp.join(configs_dir, "config.yaml")
WAYPOINTS_YAML = osp.join(configs_dir, "waypoints.yaml")

ROS_TOPICS = osp.join(configs_dir, "ros_topic_names.yaml")
ros_topics = CN()
ros_topics.set_new_allowed(True)
ros_topics.merge_from_file(ROS_TOPICS)


def get_waypoint_yaml(waypoint_file=WAYPOINTS_YAML):
    with open(waypoint_file) as f:
        return yaml.safe_load(f)


def get_default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--opts", nargs="*", default=[])
    return parser


def construct_config(file_path=DEFAULT_CONFIG, opts=None):
    if opts is None:
        opts = []
    config = CN()
    config.set_new_allowed(True)
    config.merge_from_file(file_path)
    config.merge_from_list(opts)

    new_weights = {}
    for k, v in config.WEIGHTS.items():
        if not osp.isfile(v):
            new_v = osp.join(spot_rl_experiments_dir, v)
            if not osp.isfile(new_v):
                raise KeyError(f"Neither {v} nor {new_v} exist!")
            new_weights[k] = new_v
    config.WEIGHTS.update(new_weights)

    return config


def nav_target_from_waypoint(waypoint, waypoints_yaml):
    waypoints_yaml_nav_target_dict = waypoints_yaml.get("nav_targets")
    if waypoints_yaml_nav_target_dict is None:
        raise Exception(
            "No `nav_targets` found in waypoints.yaml. Please construct waypoints.yaml correctly as per the README.md"
        )

    nav_target = waypoints_yaml_nav_target_dict.get(waypoint)
    if nav_target is None:
        raise Exception(
            f"Requested waypoint - {waypoint} does not exist inside `nav_targets` in file waypoints.yaml. Please construct waypoints.yaml correctly as per the README.md"
        )

    goal_x, goal_y, goal_heading = nav_target
    return goal_x, goal_y, np.deg2rad(goal_heading)


def place_target_from_waypoint(waypoint, waypoints_yaml):
    waypoints_yaml_place_target_dict = waypoints_yaml.get("place_targets")
    if waypoints_yaml_place_target_dict is None:
        raise Exception(
            "No `place_targets` found in waypoints.yaml. Please construct waypoints.yaml correctly as per the README.md"
        )

    place_target = waypoints_yaml_place_target_dict.get(waypoint)
    if place_target is None:
        raise Exception(
            f"Requested waypoint - {waypoint} does not exist inside `place_targets` in file waypoints.yaml. Please construct waypoints.yaml correctly as per the README.md"
        )
    return np.array(place_target)


def closest_clutter(x, y, clutter_blacklist=None):
    waypoints_yaml = get_waypoint_yaml(WAYPOINTS_YAML)
    if clutter_blacklist is None:
        clutter_blacklist = []
    clutter_locations = [
        (np.array(nav_target_from_waypoint(w, waypoints_yaml)[:2]), w)
        for w in waypoints_yaml["clutter"]
        if w not in clutter_blacklist
    ]
    xy = np.array([x, y])
    dist_to_clutter = lambda i: np.linalg.norm(i[0] - xy)  # noqa
    _, waypoint_name = sorted(clutter_locations, key=dist_to_clutter)[0]
    return waypoint_name, nav_target_from_waypoint(waypoint_name, waypoints_yaml)


def object_id_to_nav_waypoint(object_id):
    waypoints_yaml = get_waypoint_yaml(WAYPOINTS_YAML)
    if isinstance(object_id, str):
        for k, v in waypoints_yaml["object_targets"].items():
            if v[0] == object_id:
                object_id = int(k)
                break
        if isinstance(object_id, str):
            KeyError(f"{object_id} not a valid class name!")
    place_nav_target_name = waypoints_yaml["object_targets"][object_id][1]
    return place_nav_target_name, nav_target_from_waypoint(
        place_nav_target_name, waypoints_yaml
    )


def object_id_to_object_name(object_id):
    waypoints_yaml = get_waypoint_yaml(WAYPOINTS_YAML)
    return waypoints_yaml["object_targets"][object_id][0]


def get_clutter_amounts():
    waypoints_yaml = get_waypoint_yaml(WAYPOINTS_YAML)
    return waypoints_yaml["clutter_amounts"]


def arr2str(arr):
    if arr is not None:
        return f"[{', '.join([f'{i:.2f}' for i in arr])}]"
    return


def map_user_input_to_boolean(prompt):
    """
    Maps user input to boolean

    Args:
        prompt (str): Prompt to display to user

    Returns:
        bool: True if user input is y or yes, False if user input is n or no
    """
    while True:
        user_input = input(prompt + "(y/n): ").strip()
        if user_input.lower() in ["y", "yes"]:
            return True
        elif user_input.lower() in ["n", "no"]:
            return False
        else:
            print("Please enter a valid input - y, yes, n or no")


def conditional_print(message: str, verbose: bool = False):
    """
    Print the message if the verbose flag is set to True

    Args:
        message (str): Message to be printed if verbose is True (can also be f-string)
        verbose (bool): Flag to determine whether to print the message or not
    """
    if verbose:
        print(message)


class FixSizeOrderedDict(OrderedDict):
    def __init__(self, *args, maxlen=0, **kwargs):
        self._maxlen = maxlen
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        if self._maxlen > 0:
            if len(self) > self._maxlen:
                self.popitem(False)
