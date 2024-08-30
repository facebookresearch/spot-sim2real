# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# black: ignore-errors
import json
import os


def calculate_height(object_tag):
    # Iterate through each dictionary in the world_graph
    os.chdir(
        "/home/tusharsangam/Desktop/spot-sim2real/spot_rl_experiments/spot_rl/envs"
    )
    with open("cfslam_object_relations.json") as f:
        world_graph = json.load(f)
    print("THE OBJECT I GOT IS", object_tag)
    for rel in world_graph:
        # Iterate through all keys in the dictionary
        for key, value in rel.items():
            if isinstance(value, dict) and value.get("object_tag") == object_tag:
                object_node = value
                # Extract the height
                if "bbox_center" in object_node and "bbox_extent" in object_node:
                    bbox_center = object_node["bbox_center"]
                    bbox_extent = object_node["bbox_extent"]
                    # Calculate the height
                    height = bbox_center[2] + bbox_extent[2]
                    return height
                else:
                    raise ValueError(f"Object with tag '{object_tag}' is missing")
    raise ValueError(f"Object with tag '{object_tag}' not found in world_graph")
