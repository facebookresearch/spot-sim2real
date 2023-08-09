# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os

import numpy as np
import pytest
from spot_rl.envs.place_env import PlaceController, construct_config_for_place
from spot_rl.utils.geometry_utils import is_position_within_bounds
from spot_rl.utils.utils import get_waypoint_yaml, place_target_from_waypoint
from spot_wrapper.spot import Spot

hardware_tests_dir = os.path.dirname(os.path.abspath(__file__))
test_configs_dir = os.path.join(hardware_tests_dir, "configs")
test_data_dir = os.path.join(hardware_tests_dir, "data")
test_nav_trajectories_dir = os.path.join(test_data_dir, "nav_trajectories")
test_square_nav_trajectories_dir = os.path.join(
    test_nav_trajectories_dir, "square_of_side_200cm"
)
TEST_WAYPOINTS_YAML = os.path.join(test_configs_dir, "waypoints.yaml")
TEST_CONFIGS_YAML = os.path.join(test_configs_dir, "config.yaml")


def init_config():
    """
    Initialize config object for Nav test

    Returns:
        config: Config object
    """
    config = construct_config_for_place(file_path=TEST_CONFIGS_YAML, opts=[])

    return config


def test_place():
    config = init_config()
    test_waypoints_yaml_dict = get_waypoint_yaml(waypoint_file=TEST_WAYPOINTS_YAML)

    test_waypoints_local = [
        "test_place_front_local",
        "test_place_left_local",
        "test_place_right_local",
    ]
    test_place_targets_list = [
        place_target_from_waypoint(test_waypoint, test_waypoints_yaml_dict)
        for test_waypoint in test_waypoints_local
    ]

    test_spot = Spot("PlaceEnvHardwareTest")
    test_result = None
    with test_spot.get_lease(hijack=True):
        place_controller = PlaceController(
            config=config, spot=test_spot, use_policies=False
        )

        try:
            test_result = place_controller.execute(
                place_target_list=test_place_targets_list, is_local=True
            )
        except Exception:
            pytest.fails(
                "Pytest raised an error while executing PlaceController.execute from test_place_env.py"
            )
        finally:
            place_controller.shutdown(should_dock=True)

    print(f"Place test results : {test_result}")
    assert test_result is not []
    assert len(test_result) == len(test_waypoints_local)

    for wp_idx in range(len(test_waypoints_local)):
        # Capture test and target position of place target in base frame
        test_position = list(test_result[wp_idx].get("ee_pos"))
        target_position = list(test_result[wp_idx].get("place_target"))
        # Test that robot reached its goal successfully spatially
        assert (
            is_position_within_bounds(
                test_position,
                target_position,
                config.SUCC_XY_DIST,
                config.SUCC_Z_DIST,
            )
            is True
        )

        assert test_result[wp_idx].get("success") is True
