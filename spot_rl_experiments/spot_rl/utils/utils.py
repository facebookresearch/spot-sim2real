# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import json
import os
import os.path as osp
from collections import OrderedDict

import numpy as np
import rospy
import yaml
from spot_rl.utils.construct_configs import construct_config_for_semantic_place
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

ROS_FRAMES = osp.join(configs_dir, "ros_frame_names.yaml")
ros_frames = CN()
ros_frames.set_new_allowed(True)
ros_frames.merge_from_file(ROS_FRAMES)


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


def get_skill_name_and_input_from_ros():
    """
    Get the ros parameters to get the current skill name and its input
    """
    skill_name_input = rospy.get_param("/skill_name_input", "None,None")
    skill_name_input = skill_name_input.split(",")
    if len(skill_name_input) == 2:
        # We get the correct format to execute the skill
        skill_name = skill_name_input[0]
        skill_input = skill_name_input[1]
    else:
        # We do not get the skill name and input
        skill_name = "None"
        skill_input = "None"

    return skill_name, skill_input


def arr2str(arr):
    if arr is not None:
        return f"[{', '.join([f'{i:.2f}' for i in arr])}]"
    return


def load_cg(default_config):
    script_dir = os.path.dirname(__file__)
    json_file_path = os.path.join(script_dir, default_config.CONCEPT_GRAPH_FILE)

    world_graph = None
    if osp.isfile(json_file_path):
        with open(json_file_path) as f:
            world_graph = json.load(f)
    else:
        print("Concept Graph File does not exist")
    return world_graph


def calculate_height(object_tag):
    default_config = construct_config_for_semantic_place()
    default_height = default_config.HEIGHT_THRESHOLD

    world_graph = load_cg(default_config)
    if not world_graph:
        print(f"Using default height: {default_height}")
        return default_height
    try:
        object_id_str, object_tag = object_tag.split("_", 1)
        object_id = int(object_id_str)
    except Exception as e:
        print(f"Invalid object_tag format: {object_tag} due to {e}")
        return default_height

    for rel in world_graph:
        for key, value in rel.items():
            if isinstance(value, dict):
                if (
                    value.get("id") == object_id
                    and value.get("object_tag") == object_tag
                ):
                    object_node = value
                    # Extract the height
                    if "bbox_center" in object_node and "bbox_extent" in object_node:
                        bbox_center = object_node["bbox_center"]
                        bbox_extent = object_node["bbox_extent"]
                        # Calculate the height
                        height = bbox_center[2] + bbox_extent[2]
                        return height
                    else:
                        print(
                            f"Object with ID '{object_id}' and tag '{object_tag}' missing bbox properties! Returning Default Height"
                        )
                        return default_height
    # If the object tag is empty, we return the threhold height
    print(
        f"Object with ID '{object_id}' and tag '{object_tag}' not found in world_graph"
    )
    return default_height


def is_on_top(
    point, ee_pose, min_coord, max_coord, tolerance=1e-6, height_tolerance=0.1
):
    """
    Check if a point is on the top of the bounding box.

    :param point: (x, y, z) coordinate to check
    :param center: (x, y, z) center coordinate of the bounding box
    :param extents: (dx, dy, dz) extents of the bounding box
    :param tolerance: floating-point tolerance for comparisons
    :return: Boolean indicating if the point is on top of the box
    """
    place_region_min = point - np.array([0.2] * 3)
    place_region_max = point + np.array([0.2] * 3)

    place_region_x_bounds_min = min(place_region_min[0], place_region_max[0])
    bbox_x_bounds_min = min(min_coord[0], max_coord[0])
    x_bounds_min = max(place_region_x_bounds_min, bbox_x_bounds_min)
    # For Y
    place_region_y_bounds_min = min(place_region_min[1], place_region_max[1])
    bbox_y_bounds_min = min(min_coord[1], max_coord[1])
    y_bounds_min = max(place_region_y_bounds_min, bbox_y_bounds_min)

    x_in_bounds = x_bounds_min <= point[0] <= max(min_coord[0], max_coord[0])
    y_in_bounds = y_bounds_min <= point[1] <= max(min_coord[1], max_coord[1])

    ee_x_in_bounds = x_bounds_min <= ee_pose[0] <= max(min_coord[0], max_coord[0])
    ee_y_in_bounds = (
        min(min_coord[1], max_coord[1]) <= ee_pose[1] <= max(min_coord[1], max_coord[1])
    )

    print("x_in_bounds: ", x_in_bounds, x_bounds_min, max(min_coord[0], max_coord[0]))
    print(
        "y_in_bounds: ",
        y_in_bounds,
        min(min_coord[1], max_coord[1]),
        max(min_coord[1], max_coord[1]),
    )
    print(
        "EE_IN_X_BOUNDS :",
        ee_x_in_bounds,
        min(min_coord[0], max_coord[0]),
        max(min_coord[0], max_coord[0]),
    )
    print(
        "EE_IN_Y_BOUNDS :",
        ee_y_in_bounds,
        min(min_coord[1], max_coord[1]),
        max(min_coord[1], max_coord[1]),
    )

    return (x_in_bounds and y_in_bounds), (ee_x_in_bounds and ee_y_in_bounds)


def bbox_coords(object_tag, fn_to_convert_from_home_to_base):
    default_config = construct_config_for_semantic_place()
    script_dir = os.path.dirname(__file__)
    json_file_path = os.path.join(script_dir, default_config.CONCEPT_GRAPH_FILE)
    default_height = default_config.HEIGHT_THRESHOLD

    if osp.isfile(json_file_path):
        with open(json_file_path) as f:
            world_graph = json.load(f)
    else:
        print(
            f"Concept Graph File does not exist. Using default height: {default_height}"
        )
        return default_height
    try:
        object_id_str, object_tag = object_tag.split("_", 1)
        object_id = int(object_id_str)
    except Exception as e:
        print(f"Invalid object_tag format: {object_tag} due to {e}")
        return default_height

    for rel in world_graph:
        for key, value in rel.items():
            if isinstance(value, dict):
                if (
                    value.get("id") == object_id
                    and value.get("object_tag") == object_tag
                ):
                    object_node = value
                    if "bbox_center" in object_node and "bbox_extent" in object_node:
                        bbox_center = object_node["bbox_center"]
                        bbox_extent = object_node["bbox_extent"]
                        # Calculate the coords
                        x_center, y_center, z_center = bbox_center
                        length, width, depth = bbox_extent
                        half_length = length / 2
                        half_width = width / 2
                        half_depth = depth / 2

                        # Calculate xmin, xmax, ymin, ymax
                        xmin = x_center - half_length
                        xmax = x_center + half_length
                        ymax = y_center - half_width
                        ymin = y_center + half_width
                        zmin = z_center - half_depth
                        zmax = z_center + half_depth
                        maxc = np.array([xmax, ymax, zmax])
                        minc = np.array([xmin, ymin, zmin])

                        return maxc, minc
                    else:
                        print(
                            f"Object with ID '{object_id}' and tag '{object_tag}' missing bbox properties!"
                        )
                        return None
    # If the object tag is empty, we return the threhold height
    print(
        f"Object with ID '{object_id}' and tag '{object_tag}' not found in world_graph"
    )
    return None


class FixSizeOrderedDict(OrderedDict):
    def __init__(self, *args, maxlen=0, **kwargs):
        self._maxlen = maxlen
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        if self._maxlen > 0:
            if len(self) > self._maxlen:
                self.popitem(False)
