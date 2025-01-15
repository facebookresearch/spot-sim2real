# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json
import os.path as osp
import sys
from typing import Dict, List

import numpy as np
from spot_rl.utils.generate_place_goal import get_global_place_target
from spot_rl.utils.retrieve_robot_poses_from_cg import ROOT_PATH as CG_ROOT_PATH
from spot_rl.utils.utils import get_default_parser
from spot_wrapper.spot import Spot

CG_WAYPOINT_JSON = osp.join(
    CG_ROOT_PATH, "sg_cache", "cfslam_object_relations_mock.json"
)


def parse_arguments(args):
    parser = get_default_parser()
    parser.add_argument("-af", "--add-furniture", help="input:string -> furniture name")
    parser.add_argument("-ao", "--add-object", help="input:string -> object name")
    parser.add_argument(
        "-x",
        "--create-file",
        action="store_true",
        help="input: not needed -> create a new cg json file",
    )
    args = parser.parse_args(args=args)

    return args


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


class CGWaypointRecorder:
    """
    Class to record object & furniture relations (i.e. CG file) for Spot robot (world model)

    How to use:
    1. Create an instance of this class
    2. Call the add_furniture method with furniture_name as an argument (str)
    3. Call the save_json method to save the waypoints to the json file


    Args:
        spot (Spot): Spot robot object
        waypoint_file_path (str): path to json file to save waypoints into


    Example:
    waypoint_recorder = CGWaypointRecorder(spot=Spot, waypoint_file_path=path)
    waypoint_recorder.add_furniture("test_furniture")
    waypoint_recorder.save_json()
    """

    def __init__(self, spot: Spot, waypoint_file_path: str):
        self.spot = spot

        # Local copy of cg json which keeps getting updated as new waypoints are added
        self.waypoint_file = waypoint_file_path
        self.json_handler = JsonHandler()
        self.cg_json = []  # type: List[Dict]

        # Initialize cg json
        self._init_json()

    def _init_json(self):
        """
        Initialize member variable `self.cg_json` with the contents of the json file as a List[Dict] if it is not initialized.
        """
        if self.cg_json == []:
            self.cg_json = self.json_handler.read_json(waypoint_file=self.waypoint_file)

    def save_json(self):
        """
        Save the waypoints (self.cg_json) to the json file if it is not empty.
        It will overwrite the existing json file if it exists and will create a new one if it does not exist.
        """
        if self.cg_json == []:
            print("No waypoints to save. Exiting...")
            return

        self.json_handler.write_json(self.waypoint_file, self.cg_json)
        print(
            f"Successfully saved(/overwrote) all waypoints to file at {self.waypoint_file}:\n"
        )

    def add_furniture(self, furniture_name: str):
        # Get new entity id. [{el0,el1}, {el2,el3}, {el4,el5}, ...]
        new_entity_id = 2 * len(self.cg_json)

        # Get current nav pose
        x, y, yaw = self.spot.get_xy_yaw()
        yaw_deg = np.rad2deg(yaw)
        robot_pose = [float(x), float(y), float(yaw_deg)]

        # Get place target as current gripper position
        place_target = [*get_global_place_target(self.spot)]

        new_furniture_relation = {
            "object1": {
                "id": new_entity_id,
                "object_tag": furniture_name,
                "bbox_extent": [0.5, 0.5, place_target[2] / 2.0],
                "bbox_center": [
                    place_target[0],
                    place_target[1],
                    place_target[2] / 2.0,
                ],
                "category_tag": "furniture",
                "orginal_class_name": furniture_name,
                "robot_pose": robot_pose,
            },
            "object2": {
                "id": new_entity_id + 1,
                "object_tag": "invalid",
                "bbox_extent": [0.1, 0.1, 0.1],
                "bbox_center": place_target,
                "category_tag": "object",
                "orginal_class_name": "object",
                "robot_pose": robot_pose,
            },
            "object_relation": "FAIL",
            "room_region": "living_room",
        }

        self.cg_json.append(new_furniture_relation)

    def add_object(self, object_name: str):
        raise NotImplementedError(
            "Adding new objects in CG file is not yet supported. Please update bbox location from add_furniture to add this support"
        )


def main(spot: Spot):
    args = parse_arguments(args=sys.argv[1:])
    arg_bools = [args.add_furniture, args.add_object, args.create_file]
    print(arg_bools)
    assert (
        len([i for i in arg_bools if i]) == 1
    ), "Must pass in either -f, -o, or -x as an arg, and not more than one."

    # Create WaypointRecorder object with default waypoint file
    waypoint_recorder = CGWaypointRecorder(
        spot=spot, waypoint_file_path=CG_WAYPOINT_JSON
    )

    if args.create_file:
        # Initial cg json is created already in constructor of CGWaypointRecorder, skip
        pass
    elif args.add_furniture:
        waypoint_recorder.add_furniture(args.add_furniture)
    elif args.add_object:
        waypoint_recorder.add_object(args.add_object)
    else:
        raise NotImplementedError

    waypoint_recorder.save_json()


if __name__ == "__main__":
    spot = Spot("WaypointRecorder")
    main(spot)

    # jh = JsonHandler()
    # cg_data = jh.read_json(CG_WAYPOINT_JSON)
    # assert cg_data == []
    # print("Init cg data after reading ", cg_data)

    # cg_data.append(
    #     {
    #         "object1": {"id": 1},
    #         "object2": {"id": 2},
    #         "object_relation": "a on b",
    #         "room_region": "living_room",
    #     }
    # )
    # print("Updated cg dict data:\n", cg_data)
    # jh.write_json(CG_WAYPOINT_JSON, cg_data)

    # cg_data = jh.read_json(CG_WAYPOINT_JSON)
    # print("Reading from file after writing: \n", cg_data)
    # assert cg_data == [
    #     {
    #         "object1": {"id": 1},
    #         "object2": {"id": 2},
    #         "object_relation": "a on b",
    #         "room_region": "living_room",
    #     }
    # ]
