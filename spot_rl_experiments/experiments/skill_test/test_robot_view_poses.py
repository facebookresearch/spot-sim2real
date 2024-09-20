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

bboxs_info_iphone_fixed_classes_old_jmmy = {
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

fallback_CG = {
    "coffee_table": {
        "id": 68,
        "bbox_extent": [0.9, 0.5, 0.3],
        "bbox_center": [2.3, 2.3, -0.1],
        "object_tag": "coffee table",
    },
    "sink": {
        "id": 119,
        "bbox_extent": [0.9, 0.7, 0.3],
        "bbox_center": [4.3, -2.0, 0.6],
        "object_tag": "sink",
    },
    "dining_table": {
        "id": 95,
        "bbox_extent": [1.7, 0.7, 0.1],
        "bbox_center": [0.1, -2.0, 0.5],
        "object_tag": "dining table",
    },
    "teddy_bear": {
        "id": 10,
        "bbox_extent": [1.4, 0.9, 0.2],
        "bbox_center": [8.8, 6.1, 0.0],
        "object_tag": "cabinet",
        "category_tag": "furniture",
    },
    "cabinet": {
        "id": 35,
        "bbox_extent": [0.9, 0.5, 0.4],
        "bbox_center": [1.4, 5.8, 0.0],
        "object_tag": "cabinet",
    },
    "box_of_cereal": {
        "id": 135,
        "bbox_extent": [0.7, 0.6, 0.1],
        "bbox_center": [4.9, -4.5, 0.8],
        "object_tag": "gas stove",
    },
    "wooden_dresser": {
        "id": 23,
        "bbox_extent": [2.0, 1.1, 0.7],
        "bbox_center": [6.0, 3.4, 0.1],
        "object_tag": "furniture",
    },
    "office_chair": {
        "id": 38,
        "bbox_extent": [0.7, 0.5, 0.1],
        "bbox_center": [0.3, 7.3, 0.3],
        "object_tag": "chair",
    },
    "white_couch": {
        "id": 33,
        "bbox_extent": [0.4, 0.4, 0.1],
        "bbox_center": [2.8, 2.3, 0.1],
        "object_tag": "white couch",
    },
    "kitchen_island": {
        "id": 154,
        "bbox_extent": [0.8, 0.5, 0.3],
        "bbox_center": [3.2, -1.1, 0.5],
        "object_tag": "chair",
    },
    "shelf": {
        "id": 51,
        "bbox_extent": [1.5, 0.8, 0.6],
        "bbox_center": [-0.9, 8.3, 1.1],
        "object_tag": "shelf",
    },
}

bboxs_info = {
    # kitchen area
    "box_of_cereal": {
        "id": 39,
        "bbox_extent": [0.9, 0.8, 0.5],
        "bbox_center": [4.6, -4.6, 1.7],
        "object_tag": "bottle",
    },
    "gas_stove": {
        "id": 40,
        "bbox_extent": [2.8, 0.9, 0.1],
        "bbox_center": [5.1, -4.5, 0.7],
        "object_tag": "stove",
    },
    "fridge": {
        "id": 31,
        "bbox_extent": [1.9, 1.2, 0.3],
        "bbox_center": [6.9, -3.9, 0.6],
        "object_tag": "large refrigerator",
    },
    "sink": {
        "id": 48,
        "bbox_extent": [1.1, 0.7, 0.3],
        "bbox_center": [4.4, -2.0, 0.5],
        "object_tag": "sink",
    },
    "kitchen_island": {  # redo
        "id": 53,
        "bbox_extent": [0.7, 0.5, 0.5],
        "bbox_center": [3.5, -1.2, 0.3],
        "object_tag": "chair",
    },
    "dining_table": {
        "id": 61,
        "bbox_extent": [2.1, 1.3, 0.8],
        "bbox_center": [0.4, -2.3, 0.2],
        "object_tag": "wooden dining table",
    },
    # living room
    "couch_overlapping_with_coffee": {
        "id": 74,
        "bbox_extent": [1.2, 0.7, 0.4],
        "bbox_center": [1.5, 1.9, 0.2],
        "object_tag": "colorful pillow",
    },
    "cocuh_left_edge": {
        "id": 75,
        "bbox_extent": [3.9, 0.9, 0.5],
        "bbox_center": [0.9, 2.2, 0.1],
        "object_tag": "couch",
    },
    "white_couch_L": {
        "id": 72,
        "bbox_extent": [1.6, 1.5, 0.6],
        "bbox_center": [-0.4, 2.9, -0.0],
        "object_tag": "white couch",
    },
    "coffee_table": {
        "id": 71,
        "bbox_extent": [0.9, 0.9, 0.6],
        "bbox_center": [1.1, 3.3, -0.1],
        "object_tag": "coffee table",
    },
    "wooden_dresser": {  # id 2
        "id": 27,
        "bbox_extent": [1.9, 1.0, 0.6],
        "bbox_center": [6.0, 3.6, 0.1],
        "object_tag": "wooden cabinet or dresser",
    },
    "living_room_desktop_table": {
        "id": 83,
        "bbox_extent": [1.6, 0.8, 0.6],
        "bbox_center": [1.7, 5.2, 0.1],
        "object_tag": "wooden furniture",
    },
    "chair_beside_wooden_dresser": {
        "id": 0,
        "bbox_extent": [1.2, 1.0, 0.5],
        "bbox_center": [5.9, 4.8, 0.1],
        "object_tag": "chair",
    },
    "chair_besides_desktop": {
        "id": 84,
        "bbox_extent": [1.1, 0.9, 0.6],
        "bbox_center": [3.1, 5.2, 0.1],
        "object_tag": "white chair",
    },
    # office
    "left_cabinet_in_office": {
        "id": 13,
        "bbox_extent": [1.7, 0.9, 0.7],
        "bbox_center": [0.8, 5.9, 0.0],
        "object_tag": "wooden cabinet or shelf",
    },
    "office_right_shelf": {
        "id": 22,
        "bbox_extent": [2.3, 1.4, 0.7],
        "bbox_center": [-1.1, 8.1, 0.8],
        "object_tag": "wooden shelf/table",
    },
    "office_left_shelf": {
        "id": 15,
        "bbox_extent": [1.0, 0.5, 0.3],
        "bbox_center": [-0.8, 6.4, 0.1],
        "object_tag": "shelf",
    },
    "office_desk": {
        "id": 21,
        "bbox_extent": [1.8, 0.8, 0.3],
        "bbox_center": [-0.9, 7.1, 0.3],
        "object_tag": "office desk",
    },
    # bedroom
    "bedroom_dresser": {
        "id": 3,
        "bbox_extent": [1.9, 0.9, 0.7],
        "bbox_center": [8.5, 6.5, 0.0],
        "object_tag": "dresser",
    },
    "lamp_left_bed": {
        "id": 5,
        "bbox_extent": [0.8, 0.6, 0.4],
        "bbox_center": [10.6, 2.7, 0.2],
        "object_tag": "wooden dresser/nightstand",
    },
    "lamp_right_bed": {
        "id": 10,
        "bbox_extent": [0.9, 0.7, 0.5],
        "bbox_center": [8.3, 2.6, 0.2],
        "object_tag": "wooden nightstand/dresser",
    },
}


bboxs_info = {
    "23_sink": {
        "id": 23,
        "bbox_extent": [0.6, 0.6, 0.2],
        "bbox_center": [-0.7, 2.8, 0.5],
        "object_tag": "sink",
    },
    "3_wooden_dining_table": {
        "id": 3,
        "bbox_extent": [0.3, 0.1, 0.1],
        "bbox_center": [0.7, 1.3, -0.3],
        "object_tag": "wooden dining table",
    },
    "36_tall_black_bookshelf": {
        "id": 36,
        "bbox_extent": [0.9, 0.3, 0.1],
        "bbox_center": [5.4, 2.7, 0.4],
        "object_tag": "tall black bookshelf",
    },
    "14_white_cabinet": {
        "id": 14,
        "bbox_extent": [1.2, 1.1, 0.7],
        "bbox_center": [3.4, 2.7, 0.1],
        "object_tag": "white cabinet",
    },
    "12_wooden_round_table": {
        "id": 12,
        "bbox_extent": [0.9, 0.8, 0.2],
        "bbox_center": [3.1, -1.3, 0.0],
        "object_tag": "wooden round table",
    },
    "31_desk": {
        "id": 31,
        "bbox_extent": [1.4, 0.9, 0.1],
        "bbox_center": [6.1, 0.2, 0.3],
        "object_tag": "desk",
    },
    "28_office_chair": {
        "id": 28,
        "bbox_extent": [0.3, 0.1, 0.1],
        "bbox_center": [5.6, 0.6, 0.3],
        "object_tag": "office chair",
    },
    "32_office_chair": {
        "id": 32,
        "bbox_extent": [1.2, 1.0, 0.7],
        "bbox_center": [3.7, 0.3, 0.0],
        "object_tag": "cabinet",
    },
    "13_white dresser or cabinet": {
        "id": 13,
        "bbox_extent": [1.1, 1.1, 0.7],
        "bbox_center": [3.1, 0.5, 0.0],
        "object_tag": "white dresser or cabinet",
    },
}


# bboxs_info = bboxs_info_iphone_fixed_classes_new_jmmy
VISUALIZE = True
PATH_PLANNING_VISUALIZATION_FOLDER = "receptacle_before_demo"
os.makedirs(PATH_PLANNING_VISUALIZATION_FOLDER, exist_ok=True)


for receptacle_name in bboxs_info:
    if (
        receptacle_name != "None"
    ):  # "lamp_right_bed":  # receptacle_name == "teddy_bear":
        print(f"Current Receptacle {receptacle_name}")
        bbox_info = bboxs_info[receptacle_name]
        if isinstance(bbox_info["object_tag"], str):
            bbox_info["object_tag"] = [bbox_info["object_tag"]]
        # Get the view poses
        view_poses = get_view_poses(
            bbox_info["bbox_center"],
            bbox_info["bbox_extent"],
            bbox_info["object_tag"],
            bbox_info.get("id", None),
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
        continu = True  # map_user_input_to_boolean("Continue to next receptacle ?")
        if not continu:
            break
