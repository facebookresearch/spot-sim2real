import json
import os.path as osp
from typing import Dict, List, Optional

from spot_rl.utils.retrieve_robot_poses_from_cg import ROOT_PATH as CG_ROOT_PATH

CG_WAYPOINT_JSON = osp.join(
    CG_ROOT_PATH, "sg_cache", "cfslam_object_relations_mock.json"
)


class JsonHandler:
    """
    Class to handle reading and writing to json files

    How to use:
    1. Create a simple json file with the following format:

    [
        {
            "id": 1,
            "object_tag": "cabinet",
            "bbox_extent": [
                0.1,
                0.1,
                0.1
            ],
            "bbox_center": [
                3.9,
                -4.2,
                0.7
            ],
            "category_tag": "furniture",
            "orginal_class_name": "cabinet",
            "robot_pose": [
                1.0,
                1.0,
                -90.0
            ]
        },
    ]

    2. Create an instance of this class
    3. Read the json file using the read_json method as a dict
    4. Modify the cg json outside of this class object as needed
    5. Write the cg json into file using the write_json method with the created instance

    Example:
    json_handler = JsonHandler()
    cg_json = json_handler.read_json(waypoint_file=waypoint_file)
    cg_json.append(
        {
            "id": 1,
            "object_tag": "cabinet",
            "bbox_extent": [
                0.1,
                0.1,
                1.3
            ],
            "bbox_center": [
                3.9,
                -4.2,
                0.7
            ],
            "category_tag": "furniture",
            "orginal_class_name": "cabinet",
            "robot_pose": [
                1.0,
                1.0,
                -90.0
            ]
        },
    )   # Modify the json_dict
    json_handler.write_json(waypoint_file=waypoint_file, json_dict=json_dict)
    """

    def __init__(self):
        pass

    def construct_cg_json(self):
        """
        Constructs and returns an empty cg json (as list of dict) with dummy objects and furniture
        """
        init_cg_json = []  # type: List[Dict]

        return init_cg_json

    def read_json(self, waypoint_file: str):
        """
        Read a json file and returns a dict

        Args:
            waypoint_file (str): path to json file

        Returns:
            cg_json (list[dict]): Contents of the json file as a list of dicts if it exists, else an contructs a new simple cg json
        """

        cg_json = []  # type: List[Dict]

        # Read json file if it exists
        if osp.exists(waypoint_file):
            with open(waypoint_file, "r") as f:
                print(
                    f"Reading waypoints from already existing cg json file at {waypoint_file}"
                )
                cg_json = json.load(f)

        else:
            print(
                f"Creating an empty cg as cg json does not exist on path {waypoint_file}"
            )

        return cg_json

    def write_json(self, waypoint_file: str, cg_json: List[Dict]):
        """
        Write the cg json into file.
        If the file does not exist, it will be created.

        Args:
            waypoint_file (str): path to json file
            cg_json (List[dict]): dict to be written to json file
        """
        with open(waypoint_file, "w+") as f:
            json.dump(cg_json, f, indent=4)


jh = JsonHandler()
cg_json = jh.read_json(CG_WAYPOINT_JSON)
for el in cg_json:
    rr = el["room_region"]
    el["object1"]["room_region"] = rr
    el["object2"]["room_region"] = rr

breakpoint()
