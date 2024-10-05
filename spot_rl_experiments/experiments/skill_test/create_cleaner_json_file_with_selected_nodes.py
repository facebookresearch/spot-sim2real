import json
import os.path as osp

from spot_rl.utils.construct_configs import load_config

PATH_TO_CONFIG_FILE = osp.join(
    osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))),
    "configs",
    "cg_config.yaml",
)
assert osp.exists(PATH_TO_CONFIG_FILE), "cg_config.yaml wasn't found"
cg_config = load_config(PATH_TO_CONFIG_FILE)

ROOT_PATH = cg_config["CG_ROOT_PATH"]
PATH_TO_OBJECT_RELATIONS_JSON = osp.join(
    ROOT_PATH, "sg_cache", "cfslam_object_relations.json"
)
assert osp.exists(
    PATH_TO_OBJECT_RELATIONS_JSON
), f"{PATH_TO_OBJECT_RELATIONS_JSON} doesnt exists"

selected_nodes = {
    "pineapple_plush_toy": {
        # not present in final json but present in cache
        # on top of kitchen counter
        "id": "71 but not present in json",
        "object_tag": "pineapple plush toy",
        "bbox_extent": [0.2, 0.1, 0.1],
        "bbox_center": [3.9, -4.2, 0.7],
        "category_tag": "object",
        "orginal_class_name": "pineapple plush toy",
        "on_top_of": "kitchen_counter",
    },
    "kitchen_counter": {
        # pineapple plush toy should be on top  of kitchen counter
        "id": 67,
        "object_tag": "blue cabinet",
        "bbox_extent": [1.1, 1.1, 0.1],
        "bbox_center": [3.6, -4.1, 0.2],
        "category_tag": "furniture",
        "orginal_class_name": "cabinet",
    },
    "stove_top": {
        # stove top not present in cache as well not in json but we use it in instructions
        # this is good substitue for that
        "id": 53,
        "object_tag": "furniture",
        "bbox_extent": [1.1, 1.1, 0.2],
        "bbox_center": [5.8, -4.4, 1.7],
        "on_top_of": "kitchen_counter",
        "category_tag": "furniture",
        "orginal_class_name": "cabinet",
    },
    "sink": {
        "id": 83,
        "object_tag": "white sink",
        "bbox_extent": [0.9, 0.6, 0.3],
        "bbox_center": [4.3, -2.0, 0.5],
        "on_top_of": "kitchen_island",
        "category_tag": "furniture",
        "orginal_class_name": "sink",
    },
    "can_of_food": {
        "id": 92,
        "object_tag": "can of food",
        "bbox_extent": [0.1, 0.1, 0.1],
        "bbox_center": [3.1, -1.0, 0.7],
        "on_top_of": "kitchen_island",
        "category_tag": "object",
        "orginal_class_name": "can of food",
    },
    "kitchen_island": {
        "id": 90,
        "object_tag": "chair",
        "bbox_extent": [0.4, 0.3, 0.2],
        "bbox_center": [3.6, -0.9, 0.4],
        "category_tag": "furniture",
        "orginal_class_name": "chair",
    },
    "dining_table": {
        "id": "100 but not present in json",
        "object_tag": "wooden dining table",
        "bbox_extent": [0.6, 0.3, 0.3],
        "bbox_center": [0.5, -1.6, 0.0],
        "category_tag": "furniture",
        "orginal_class_name": "dining table",
    },
    "donut_plush_toy": {  # not detected just use substitute
        "id": 77,
        "object_tag": "chair",
        "bbox_extent": [0.6, 0.5, 0.2],
        "bbox_center": [0.8, -1.9, 0.3],
        "on_top_of": "dining_table",
        "category_tag": "furniture",
        "orginal_class_name": "chair",
    },
    "sofa": {
        "id": 117,
        "object_tag": "pillow",
        "bbox_extent": [0.6, 0.6, 0.3],
        "bbox_center": [2.2, 1.8, 0.3],
        "category_tag": "furniture",
        "orginal_class_name": "pillow",
    },
    "catterpillar_plush": {
        "id": 116,
        "object_tag": "stuffed caterpillar",
        "bbox_extent": [0.4, 0.3, 0.1],
        "bbox_center": [2.2, 2.0, 0.1],
        "on_top_of": "sofa",
        "category_tag": "object",
        "orginal_class_name": "frog plush toy",
    },
    "coffee_table": {
        "id": 105,
        "object_tag": "coffee table",
        "bbox_extent": [0.9, 0.9, 0.6],
        "bbox_center": [1.1, 3.3, -0.1],
        "category_tag": "furniture",
        "orginal_class_name": "coffee table",
    },
    "bottle": {
        "id": 111,
        "object_tag": "bottle",
        "bbox_extent": [0.2, 0.2, 0.1],
        "bbox_center": [1.0, 3.3, 0.2],
        "on_top_of": "coffee_table",
        "category_tag": "object",
        "orginal_class_name": "bottle",
    },
    "bulldozer_toy_car": {
        "id": 123,
        "object_tag": "toy truck",
        "bbox_extent": [1.1, 0.9, 0.4],
        "bbox_center": [3.0, 5.1, 0.2],
        "on_top_of": "chair_with_bulldozer_car",
        "category_tag": "object",
        "orginal_class_name": "chair",
    },
    "chair_with_bulldozer_car": {
        "id": 103,
        "object_tag": "chair",
        "bbox_extent": [0.7, 0.4, 0.2],
        "bbox_center": [2.9, 4.8, 0.0],
        "category_tag": "furniture",
        "orginal_class_name": "chair",
    },
    "frog_plush_toy": {
        "id": 10,
        "object_tag": "frog plush toy",
        "bbox_extent": [0.3, 0.2, 0.1],
        "bbox_center": [5.8, 4.8, 0.1],
        "on_top_of": "chair_with_frog",
        "category_tag": "object",
        "orginal_class_name": "frog plush toy",
    },
    "chair_with_frog": {
        "id": 1,
        "object_tag": "stuffed frog",
        "bbox_extent": [1.1, 1.1, 0.6],
        "bbox_center": [5.9, 4.7, 0.1],
        "category_tag": "furniture",
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
    "avocado_plush_toy": {
        "id": 7,
        "object_tag": "lamp",
        "bbox_extent": [0.4, 0.3, 0.2],
        "bbox_center": [6.0, 4.1, 0.7],
        "category_tag": "object",
        "orginal_class_name": "lamp",
        "on_top_of": "living_room_console",
    },
    "office_cabinet": {
        "id": 35,
        "object_tag": "wooden furniture",
        "bbox_extent": [0.9, 0.8, 0.3],
        "bbox_center": [0.8, 6.0, 0.2],
        "category_tag": "furniture",
        "orginal_class_name": "cabinet",
    },
    "tajin_bottle": {
        "id": 34,
        "object_tag": "bottle",
        "bbox_extent": [0.2, 0.1, 0.1],
        "bbox_center": [0.9, 5.9, 0.4],
        "category_tag": "object",
        "orginal_class_name": "bottle",
        "on_top_of": "office_cabinet",
    },
    "brown_leather_chair_with_can_of_soda": {
        "id": 41,
        "object_tag": "can of soda",
        "bbox_extent": [1.0, 0.8, 0.5],
        "bbox_center": [2.3, 9.0, 0.0],
        "category_tag": "furniture",
        "orginal_class_name": "chair",
    },
    "can_of_soda": {
        "id": 42,
        "object_tag": "bottle",
        "bbox_extent": [0.1, 0.1, 0.1],
        "bbox_center": [2.2, 8.5, 0.0],
        "category_tag": "object",
        "orginal_class_name": "can of soda",
        "on_top_of": "brown_leather_chair_with_can_of_soda",
    },
    "brown_leather_chair_with_cup": {
        "id": 39,
        "object_tag": "cup",
        "bbox_extent": [0.9, 0.7, 0.5],
        "bbox_center": [1.4, 8.9, 0.0],
        "category_tag": "furniture",
        "orginal_class_name": "chair",
    },
    "cup": {
        "id": 40,
        "object_tag": "speaker",
        "bbox_extent": [0.2, 0.1, 0.1],
        "bbox_center": [1.4, 8.6, 0.1],
        "category_tag": "object",
        "orginal_class_name": "cup",
        "on_top_of": "brown_leather_chair_with_cup",
    },
    "dresser": {
        "id": 11,  # it approaches from left so use below waypoint if not sure
        "object_tag": "cabinet",
        "bbox_extent": [1.6, 0.9, 0.6],
        "bbox_center": [8.6, 6.2, 0.1],
        "category_tag": "furniture",
        "orginal_class_name": "cabinet",
    },
    "ball": {
        "id": 13,
        "object_tag": "blue and red ball",
        "bbox_extent": [0.2, 0.2, 0.1],
        "bbox_center": [8.5, 6.3, 0.5],
        "category_tag": "object",
        "orginal_class_name": "ball plush toy",
        "on_top_of": "dresser",
    },
    "lamp": {
        "id": 19,
        "object_tag": "lamp",
        "bbox_extent": [0.7, 0.4, 0.1],
        "bbox_center": [10.3, 2.0, 0.6],
        "category_tag": "object",
        "orginal_class_name": "lamp",
        "on_top_of": "nightstand_on_left",
    },
    "nightstand_on_left": {
        "id": "17 but not present in json",
        "object_tag": "dresser/nightstand",
        "bbox_extent": [0.8, 0.6, 0.3],
        "bbox_center": [10.4, 2.4, 0.0],
        "category_tag": "furniture",
        "orginal_class_name": "night stand",
    },
    "nightstand_on_right": {
        "id": 24,
        "object_tag": "white dresser/nightstand",
        "bbox_extent": [0.8, 0.6, 0.5],
        "bbox_center": [8.1, 2.4, 0.0],
        "category_tag": "furniture",
        "orginal_class_name": "night stand",
    },
    "pillow": {
        "id": 20,
        "object_tag": "pillow",
        "bbox_extent": [0.9, 0.8, 0.2],
        "bbox_center": [9.6, 2.3, 0.3],
        "category_tag": "object",
        "orginal_class_name": "pillow",
        "on_top_of": "bed",
    },
    "bed": {
        "id": 20,
        "object_tag": "pillow",
        "bbox_extent": [0.9, 0.8, 0.2],
        "bbox_center": [9.6, 2.3, 0.3],
        "category_tag": "furniture",
        "orginal_class_name": "pillow",
    },
    "lamp": {
        "id": 23,
        "object_tag": "lamp",
        "bbox_extent": [0.7, 0.5, 0.4],
        "bbox_center": [8.0, 2.2, 0.7],
        "category_tag": "object",
        "orginal_class_name": "lamp",
        "on_top_of": "nightstand_on_right",
    },
}

if __name__ == "__main__":

    # with open(PATH_TO_OBJECT_RELATIONS_JSON, "r") as f:
    #     cg_json_file = json.load(f)
    final_list = []
    unique_ids = list(
        set(
            [
                node_data["id"]
                for _, node_data in selected_nodes.items()
                if isinstance(node_data["id"], int)
            ]
        )
    )
    unique_ids = sorted(unique_ids)

    for node_name, node_data in selected_nodes.items():
        if "on_top_of" not in node_data:
            continue
        parent_node_name = node_data["on_top_of"]
        parent_node_data = selected_nodes[parent_node_name]

        node_data["object_tag"] = node_name.replace("_", " ")
        parent_node_data["object_tag"] = parent_node_name.replace("_", " ")

        objects = [node_data, parent_node_data]
        for i, object in enumerate(objects):
            if isinstance(object["id"], str) and "not present" in object["id"]:
                objects[i]["id"] = unique_ids[-1] + 1
                unique_ids.append(objects[i]["id"])

        relation_node = {
            "object1": objects[0],
            "object2": objects[1],
            "object_relation": "a on b",
            "room_region": "living room",
        }
        final_list.append(relation_node)

    path_to_output_file = osp.join(
        ROOT_PATH, "sg_cache", "cfslam_object_relations_cleaned.json"
    )
    with open(path_to_output_file, "w") as f:
        json.dump(final_list, f, indent=4)

    if osp.exists(path_to_output_file):
        print(f"Output written to {path_to_output_file}")
