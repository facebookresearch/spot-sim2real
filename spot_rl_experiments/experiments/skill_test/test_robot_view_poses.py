import os
import os.path as osp

import numpy as np
from perception_and_utils.utils.generic_utils import map_user_input_to_boolean
from spot_rl.utils.retrieve_robot_poses_from_cg import get_view_poses
from spot_rl.utils.waypoint_estimation_based_on_robot_poses_from_cg import (
    get_navigation_points,
)

bboxs_info_old = {
    "coffee_table": {
        "bbox_center": [
            2.6,
            2.1,
            0.1,
        ],
        "bbox_extent": [0.8, 0.6, 0.6],
        "object_tag": ["wooden coffee table"],
    },
    "dining_table": {
        "bbox_center": [3.4, -1.1, 0.4],
        "bbox_extent": [0.4, 0.2, 0.2],
        "object_tag": ["dining table"],
    },
    "cabinet": {
        "bbox_center": [1.5, 6.1, -0.2],
        "bbox_extent": [0.7, 0.5, 0.3],
        "object_tag": ["cabinet"],
    },
    "box_of_cereal": {
        "bbox_center": [3.6, -4.2, 0.7],
        "bbox_extent": [1.0, 0.4, 0.2],
        "object_tag": [
            "kitchen counter"
        ],  # ["box of cereal", "cereal box", "corn flakes box", "cheerios box"]
    },
    "wooden_dresser": {
        "bbox_center": [5.4, 2.5, -0.2],
        "bbox_extent": [1.2, 0.9, 0.4],
        "object_tag": ["wooden dresser"],
    },
    "office_chair": {
        "bbox_center": [0.5, 8.8, -0.4],
        "bbox_extent": [2.5, 2.0, 0.3],
        "object_tag": ["chair"],
    },
    "couch": {
        "bbox_center": [2.1, 1.5, 0.2],
        "bbox_extent": [1.8, 0.7, 0.4],
        "object_tag": ["white couch"],
    },
    "sink": {
        "bbox_center": [4.3, -2.0, 0.5],
        "bbox_extent": [1.0, 0.8, 0.4],
        "object_tag": ["sink"],
    },
    "shelf": {
        "bbox_center": [-1.0, 6.5, 0.6],
        "bbox_extent": [1.1, 0.9, 0.5],
        "object_tag": ["bookshelf"],
    },
    "white_vase": {
        "bbox_center": [
            0.1,
            -2.9,
            0.6,
        ],
        "bbox_extent": [0.7, 0.4, 0.3],
        "object_tag": ["white vase"],
    },
    "teddy_bear": {
        "bbox_center": [8.2, 6.0, 0.1],
        "bbox_extent": [1.3, 1.0, 0.8],
        "object_tag": ["teddy bear"],
    },
}

bboxs_info = {
    "sink": {
        "bbox_center": [4.3, -2.0, 0.6],
        "bbox_extent": [0.9, 0.7, 0.3],
        "object_tag": ["sink"],
        "object_id": 42,
    },
    "teddy bear": {
        "id": 5,
        "bbox_extent": [0.9, 0.7, 0.6],
        "bbox_center": [9.0, 6.3, 0.1],
        "object_tag": "dresser drawer",
    },
    "chair_besides_cabinet_in_living_room": {
        "id": 8,
        "bbox_extent": [1.3, 1.3, 0.9],
        "bbox_center": [5.9, 4.7, 0.0],
        "object_tag": "chair",
    },
    "cabinet_in_living_room": {
        "id": 9,
        "bbox_extent": [2.0, 1.0, 0.7],
        "bbox_center": [6.0, 3.4, 0.1],
        "object_tag": "wooden cabinet or drawer",
    },
    "office_shelf": {
        "id": 20,
        "bbox_extent": [2.0, 0.7, 0.6],
        "bbox_center": [-0.7, 7.5, 0.2],
        "object_tag": "wooden desk",
    },
    "cabinet": {
        "id": 27,
        "bbox_extent": [1.0, 0.8, 0.7],
        "bbox_center": [2.1, 5.3, 0.1],
        "object_tag": "wooden cabinet",
    },
    "kitchen_island": {
        "id": 29,
        "bbox_extent": [0.5, 0.3, 0.2],
        "bbox_center": [3.4, -0.7, 0.5],
        "object_tag": "chair",
    },
}

VISUALIZE = True
PATH_PLANNING_VISUALIZATION_FOLDER = "path_planning_vis_for_cg"
os.makedirs(PATH_PLANNING_VISUALIZATION_FOLDER, exist_ok=True)


for receptacle_name in bboxs_info:
    if receptacle_name == "coffee_table":  # receptacle_name == "teddy_bear":
        print(f"Current Receptacle {receptacle_name}")
        bbox_info = bboxs_info[receptacle_name]
        if isinstance(bbox_info["object_tag"], str):
            bbox_info["object_tag"] = [bbox_info["object_tag"]]
        # Get the view poses
        view_poses = get_view_poses(
            bbox_info["bbox_center"],
            bbox_info["bbox_extent"],
            bbox_info["object_tag"],
            VISUALIZE,
        )
        # Get the robot x, y, yaw

        # Get the navigation points
        nav_pts = get_navigation_points(
            view_poses,
            np.array(bbox_info["bbox_center"]),
            np.array(bbox_info["bbox_extent"]),
            [0, 0],
            VISUALIZE,
            osp.join(PATH_PLANNING_VISUALIZATION_FOLDER, f"{receptacle_name}.png"),
        )
        print(
            f"Final Nav point for {receptacle_name}:  {*nav_pts[-1][:2], np.rad2deg(nav_pts[-1][-1])}"
        )
        continu = map_user_input_to_boolean("Continue to next receptacle ?")
        if not continu:
            break
