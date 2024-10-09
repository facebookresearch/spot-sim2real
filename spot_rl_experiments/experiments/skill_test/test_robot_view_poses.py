import os
import os.path as osp

import numpy as np
from perception_and_utils.utils.generic_utils import map_user_input_to_boolean
from spot_rl.utils.retrieve_robot_poses_from_cg import get_view_poses
from spot_rl.utils.waypoint_estimation_based_on_robot_poses_from_cg import (
    get_navigation_points,
)

# NYC lab
bboxs_info_nyc = {
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
    "32_cabinet": {
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

# Fremont apartment
bboxs_info_fre = {
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
    "green_pumpkin_toy_on_night_stand": {
        "id": 12,
        "bbox_extent": [0.2, 0.2, 0.1],
        "bbox_center": [10.4, 2.5, 0.4],
        "object_tag": "green pumpkin toy",
    },
    "lamp_right_bed": {
        "id": 10,
        "bbox_extent": [0.9, 0.7, 0.5],
        "bbox_center": [8.3, 2.6, 0.2],
        "object_tag": "wooden nightstand/dresser",
    },
}

bbox_info_static_graph = {
    # kitchen
    "pumpkin_on_kitchen_island": {
        "id": 178,
        "bbox_extent": [0.3, 0.2, 0.1],
        "bbox_center": [3.2, -1.2, 0.7],
        "object_tag": "green apple",
    },
    "white_vase_on_dining_table_in_kitchen": {
        "id": 215,
        "bbox_extent": [0.3, 0.2, 0.2],
        "bbox_center": [0.4, -2.4, 0.6],
        "object_tag": "white vase",
    },
    "dining_table_in_kitchen": {
        "id": 203,
        "bbox_extent": [0.5, 0.3, 0.1],
        "bbox_center": [0.6, -2.9, -0.0],
        "object_tag": "wooden dining table",
    },
    "sink_in_kitchen": {
        "id": 171,
        "bbox_extent": [0.9, 0.6, 0.3],
        "bbox_center": [4.3, -2.0, 0.5],
        "object_tag": "sink",
    },
    "bottle_on_kitchen_counter": {
        # not working well,
        # can choose something else in vicinity
        "id": 153,
        "bbox_extent": [0.1, 0.0, 0.0],
        "bbox_center": [3.3, -4.8, 0.8],
        "object_tag": "bottle",
    },
    "receptacle_for_botle": {
        "id": 158,
        "bbox_extent": [2.3, 0.9, 0.1],
        "bbox_center": [4.3, -4.4, 0.6],
        "object_tag": "stove top oven",
    },
    # living room
    "coffee table_in_living_room": {
        "id": 116,
        "bbox_extent": [0.9, 0.8, 0.6],
        "bbox_center": [1.2, 3.3, 0.1],
        "object_tag": "table with decorations",
    },
    "avocado_on_living_room_console": {
        "id": 85,
        "bbox_extent": [0.3, 0.1, 0.1],
        "bbox_center": [6.3, 3.2, 0.6],
        "object_tag": "avocado",
    },
    "avocado_receptacle": {
        "id": 0,
        "bbox_extent": [1.2, 0.9, 0.6],
        "bbox_center": [5.9, 4.9, 0.2],
        "object_tag": "furniture",
    },
    "birthday_cake_on_coffee_table": {
        "id": 100,
        "bbox_extent": [0.9, 0.8, 0.7],
        "bbox_center": [1.2, 3.3, 0.0],
        "object_tag": "birthday cake on coffee table",
    },
    "white_couch_in_living_room": {
        "id": 126,
        "bbox_extent": [2.9, 1.1, 0.6],
        "bbox_center": [1.5, 2.2, 0.1],
        "object_tag": "white couch",
    },
    "pengun_plush_toy_on_white_chair": {
        "id": 106,
        "bbox_extent": [1.0, 0.8, 0.5],
        "bbox_center": [3.4, 5.1, 0.2],
        "object_tag": "stuffed penguin toy",
    },
    "white_chair_beside_living_room_console": {
        "id": 82,
        "bbox_extent": [0.2, 0.1, 0.0],
        "bbox_center": [5.8, 5.3, -0.3],
        "object_tag": "white couch",
    },
    # Office
    "coffee_cup_on_office_desk": {
        "id": 54,
        "bbox_extent": [0.3, 0.2, 0.2],
        "bbox_center": [-0.8, 7.0, 0.5],
        "object_tag": "coffee cup",
    },
    "computer_on_left_office_cabinet": {
        "id": 44,
        "bbox_extent": [0.8, 0.5, 0.2],
        "bbox_center": [1.1, 5.7, 0.9],
        "object_tag": "computer monitor",
    },
    "can_of_food_on_chair_in_office": {
        "id": 75,
        "bbox_extent": [0.8, 0.8, 0.4],
        "bbox_center": [1.2, 8.8, 0.1],
        "object_tag": "can of food on a chair",
    },
    "cup_on_chair_in_office": {
        "id": 80,
        "bbox_extent": [0.2, 0.1, 0.1],
        "bbox_center": [2.0, 8.7, 0.1],
        "object_tag": "cup",
    },
    "brown_leather_chair_for_cup": {
        "id": 78,
        "bbox_extent": [0.9, 0.8, 0.4],
        "bbox_center": [2.0, 8.8, 0.1],
        "object_tag": "brown leather chair",
    },
    # bedroom
    "ball_on_dresser_in_bedroom": {
        "id": 5,
        "bbox_extent": [0.3, 0.2, 0.1],
        "bbox_center": [8.7, 6.4, 0.6],
        "object_tag": "ball",
    },
    "receptacle_for_ball": {
        "id": 6,
        "bbox_extent": [1.2, 1.1, 0.7],
        "bbox_center": [8.9, 6.3, 0.3],
        "object_tag": "furniture",
    },
    "garlic_toy_on_left_lamp_stand_in_bedroom": {
        "id": 15,
        "bbox_extent": [0.3, 0.1, 0.1],
        "bbox_center": [10.6, 2.4, 0.4],
        "object_tag": "garlic toy",
    },
    "tomato_toy_on_right_lamp_stand_in_bedroom": {
        "id": 29,
        "bbox_extent": [0.1, 0.1, 0.0],
        "bbox_center": [8.3, 2.4, 0.4],
        "object_tag": "tomato toy",
    },
    "test_this": {
        "id": 10,
        "bbox_extent": [0.7, 0.5, 0.3],
        "bbox_center": [10.5, 2.6, 0.1],
        "object_tag": "wooden dresser",
    },
}
bbox_info_static_graph_v2 = {
    "kicthen_counter": {
        # pineapple plush toy should be on top  of kitchen counter
        "id": 67,
        "object_tag": "blue cabinet",
        "bbox_extent": [1.1, 1.1, 0.1],
        "bbox_center": [3.6, -4.1, 0.2],
    },
    "pineapple_plush_toy": {
        # not present in final json but present in cache
        # on top of kitchen counter
        "id": 71,
        "object_tag": "pineapple plush toy",
        "bbox_extent": [0.2, 0.1, 0.1],
        "bbox_center": [3.9, -4.2, 0.7],
    },
    "stove_top": {
        # stove top not present in cache as well not in json but we use it in instructions
        # this is good substitue for that
        "id": 53,
        "object_tag": "furniture",
        "bbox_extent": [1.1, 1.1, 0.2],
        "bbox_center": [5.8, -4.4, 1.7],
    },
    "sink": {
        "id": 83,
        "object_tag": "white sink",
        "bbox_extent": [0.9, 0.6, 0.3],
        "bbox_center": [4.3, -2.0, 0.5],
    },
    "can_of_food_on_kitchen_island": {
        "id": 92,
        "object_tag": "can of food",
        "bbox_extent": [0.1, 0.1, 0.1],
        "bbox_center": [3.1, -1.0, 0.7],
    },
    "kitchen_island": {
        "id": 90,
        "object_tag": "chair",
        "bbox_extent": [0.4, 0.3, 0.2],
        "bbox_center": [3.6, -0.9, 0.4],
    },
    "dining_table": {
        "id": "100 but not present in json",
        "object_tag": "wooden dining table",
        "bbox_extent": [0.6, 0.3, 0.3],
        "bbox_center": [0.5, -1.6, 0.0],
    },
    "donut_plush_toy_on_dining_table": {  # not detected just use substitute
        "id": 77,
        "object_tag": "chair",
        "bbox_extent": [0.6, 0.5, 0.2],
        "bbox_center": [0.8, -1.9, 0.3],
    },
    "sofa": {
        "id": 117,
        "object_tag": "pillow",
        "bbox_extent": [0.6, 0.6, 0.3],
        "bbox_center": [2.2, 1.8, 0.3],
    },
    "catterpillar_plush_toy_on_sofa": {
        "id": 116,
        "object_tag": "stuffed caterpillar",
        "bbox_extent": [0.4, 0.3, 0.1],
        "bbox_center": [2.2, 2.0, 0.1],
    },
    "coffee_table": {
        "id": 105,
        "object_tag": "coffee table",
        "bbox_extent": [0.9, 0.9, 0.6],
        "bbox_center": [1.1, 3.3, -0.1],
    },
    "bottle_on_coffee_table": {
        "id": 111,
        "object_tag": "bottle",
        "bbox_extent": [0.2, 0.2, 0.1],
        "bbox_center": [1.0, 3.3, 0.2],
    },
    "bulldozer_toy_car": {
        "id": 123,
        "object_tag": "toy truck",
        "bbox_extent": [1.1, 0.9, 0.4],
        "bbox_center": [3.0, 5.1, 0.2],
    },
    "chair_for_toy_car": {
        "id": 103,
        "object_tag": "chair",
        "bbox_extent": [0.7, 0.4, 0.2],
        "bbox_center": [2.9, 4.8, 0.0],
    },
    "frog_plush_toy": {
        "id": 10,
        "object_tag": "frog plush toy",
        "bbox_extent": [0.3, 0.2, 0.1],
        "bbox_center": [5.8, 4.8, 0.1],
    },
    "chair_for_frog": {
        "id": 1,
        "object_tag": "stuffed frog",
        "bbox_extent": [1.1, 1.1, 0.6],
        "bbox_center": [5.9, 4.7, 0.1],
        "category_tag": "toy",
        "orginal_class_name": "chair",
    },
    "living_room_console": {
        "id": 2,
        "object_tag": "wooden table",
        "bbox_extent": [0.6, 0.4, 0.2],
        "bbox_center": [6.0, 4.3, 0.4],
        "category_tag": "furniture",
        "orginal_class_name": "cabinet",
    },
    "avocado_plush_toy_on_living_room_console": {
        "id": 7,
        "object_tag": "lamp",
        "bbox_extent": [0.4, 0.3, 0.2],
        "bbox_center": [6.0, 4.1, 0.7],
        "category_tag": "object",
        "orginal_class_name": "lamp",
    },
    "office_cabinet": {
        "id": 35,
        "object_tag": "wooden furniture",
        "bbox_extent": [0.9, 0.8, 0.3],
        "bbox_center": [0.8, 6.0, 0.2],
        "category_tag": "furniture",
        "orginal_class_name": "cabinet",
    },
    "tajin_bottle_on_office_cabinet": {
        "id": 34,
        "object_tag": "bottle",
        "bbox_extent": [0.2, 0.1, 0.1],
        "bbox_center": [0.9, 5.9, 0.4],
        "category_tag": "object",
        "orginal_class_name": "bottle",
    },
    "brown_leather_chair_in_office": {
        "id": 41,
        "object_tag": "can of soda",
        "bbox_extent": [1.0, 0.8, 0.5],
        "bbox_center": [2.3, 9.0, 0.0],
        "category_tag": "object",
        "orginal_class_name": "chair",
    },
    "can_of_soda_on_leather_chair": {
        "id": 42,
        "object_tag": "bottle",
        "bbox_extent": [0.1, 0.1, 0.1],
        "bbox_center": [2.2, 8.5, 0.0],
        "category_tag": "object",
        "orginal_class_name": "can of soda",
    },
    "brown_leather_chair_for_cup": {
        "id": 39,
        "object_tag": "cup",
        "bbox_extent": [0.9, 0.7, 0.5],
        "bbox_center": [1.4, 8.9, 0.0],
        "category_tag": "object",
        "orginal_class_name": "chair",
    },
    "cup_on_brown_leather_chair": {
        "id": 40,
        "object_tag": "speaker",
        "bbox_extent": [0.2, 0.1, 0.1],
        "bbox_center": [1.4, 8.6, 0.1],
        "category_tag": "object",
        "orginal_class_name": "cup",
    },
    "bedroom_dresser": {
        "id": 11,  # it approaches from left so use below waypoint if not sure
        "object_tag": "cabinet",
        "bbox_extent": [1.6, 0.9, 0.6],
        "bbox_center": [8.6, 6.2, 0.1],
        "category_tag": "furniture",
        "orginal_class_name": "cabinet",
    },
    "ball_on_bedroom_dresser": {
        "id": 13,
        "object_tag": "blue and red ball",
        "bbox_extent": [0.2, 0.2, 0.1],
        "bbox_center": [8.5, 6.3, 0.5],
        "category_tag": "object",
        "orginal_class_name": "ball plush toy",
    },
    "nightstand_left_bed": {
        "id": "17 but not present in json",
        "object_tag": "dresser/nightstand",
        "bbox_extent": [0.8, 0.6, 0.3],
        "bbox_center": [10.4, 2.4, 0.0],
        "category_tag": "furniture",
        "orginal_class_name": "night stand",
    },
    "nightstand_right_bed": {
        "id": 24,
        "object_tag": "white dresser/nightstand",
        "bbox_extent": [0.8, 0.6, 0.5],
        "bbox_center": [8.1, 2.4, 0.0],
        "category_tag": "furniture",
        "orginal_class_name": "night stand",
    },
    "bed": {
        "id": 20,
        "object_tag": "pillow",
        "bbox_extent": [0.9, 0.8, 0.2],
        "bbox_center": [9.6, 2.3, 0.3],
        "category_tag": "object",
        "orginal_class_name": "pillow",
    },
}
VISUALIZE = True
PATH_PLANNING_VISUALIZATION_FOLDER = "path_simulations_for_static_cg_v2"
os.makedirs(PATH_PLANNING_VISUALIZATION_FOLDER, exist_ok=True)

if __name__ == "__main__":
    in_fre_lab = map_user_input_to_boolean("for FRE? Y/N ")
    run_real_hardware_nav = map_user_input_to_boolean("run real hardware nav? Y/N ")
    if run_real_hardware_nav:
        from spot_rl.envs.skill_manager import SpotSkillManager

    if in_fre_lab:
        bboxs_info = bbox_info_static_graph_v2
    else:
        bboxs_info = bboxs_info_nyc

    if run_real_hardware_nav:
        spotskillmanager = SpotSkillManager(
            use_mobile_pick=True, use_semantic_place=True
        )
    else:
        spotskillmanager = None

    for receptacle_name in bboxs_info:
        print(f"Current Receptacle {receptacle_name}")
        bbox_info = bboxs_info[receptacle_name]
        if isinstance(bbox_info["object_tag"], str):
            bbox_info["object_tag"] = [bbox_info["object_tag"]]
        # Get the view poses
        view_poses, _ = get_view_poses(
            np.array(bbox_info["bbox_center"]),
            np.array(bbox_info["bbox_extent"]),
            bbox_info["object_tag"],
            bbox_info.get("id", None),
            VISUALIZE,
        )
        # Get the robot x, y, yaw

        # Get the navigation points
        if spotskillmanager is not None and run_real_hardware_nav:
            curr_x, curr_y, _ = spotskillmanager.spot.get_xy_yaw()
        else:
            curr_x, curr_y = 1, 0

        nav_pts = get_navigation_points(
            view_poses,
            np.array(bbox_info["bbox_center"]),
            np.array(bbox_info["bbox_extent"]),
            [curr_x, curr_y],
            VISUALIZE,
            osp.join(PATH_PLANNING_VISUALIZATION_FOLDER, f"{receptacle_name}.png"),
        )
        print(
            f"Final Nav point for {receptacle_name}:  {*nav_pts[-1][:2], np.rad2deg(nav_pts[-1][-1])}"
        )
        if spotskillmanager is not None and run_real_hardware_nav:
            if len(nav_pts) > 0:
                final_pt_i = len(nav_pts) - 1
                num_steps = 0
                for pt_i, pt in enumerate(nav_pts):
                    x, y, yaw = pt
                    if pt_i == final_pt_i:
                        # Do normal point nav with yaw for the final location
                        succeded, msg = spotskillmanager.nav(x, y, yaw, False)
                    else:
                        # Do dynamic point yaw here for the intermediate points
                        succeded, msg = spotskillmanager.nav(x, y)
                    spotskillmanager.nav_controller.skill_result_log
