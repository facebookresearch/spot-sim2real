# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os

import numpy as np
import pytest
from spot_rl.skills.atomic_skills import Place
from spot_rl.utils.construct_configs import construct_config_for_place
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


def init_test_config():
    """
    Initialize config object for Nav test

    Returns:
        config: Config object
    """
    config = construct_config_for_place(file_path=TEST_CONFIGS_YAML, opts=[])

    return config


def validate_place_results(test_result, config, test_waypoints):
    print(f"Place test results : {test_result}")
    assert test_result is not []
    assert len(test_result) == len(test_waypoints)

    for wp_idx in range(len(test_waypoints)):
        # Capture test and target position of place target in base frame
        test_position = np.array(test_result[wp_idx].get("ee_pos"))
        target_position = np.array(test_result[wp_idx].get("place_target"))
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


def validate_place_feedback(feedback):
    (status, message) = feedback
    assert status is True
    assert message == "Successfully reached the target position"


def test_place():
    config = init_test_config()
    test_waypoints_yaml_dict = get_waypoint_yaml(waypoint_file=TEST_WAYPOINTS_YAML)

    test_waypoints = [
        "test_place_front",
        "test_place_left",
        "test_place_right",
    ]
    test_place_targets_list = [
        place_target_from_waypoint(test_waypoint, test_waypoints_yaml_dict)
        for test_waypoint in test_waypoints
    ]

    test_spot = Spot("PlaceEnvHardwareTest")
    test_result = None
    with test_spot.get_lease(hijack=True):
        test_spot.power_robot()
        place_controller = Place(spot=test_spot, config=config, use_policies=False)

        # Test place execution and verify the result
        try:
            test_result = place_controller.execute_place(
                place_target_list=test_place_targets_list, is_local=False
            )
            validate_place_results(
                test_result=test_result, config=config, test_waypoints=test_waypoints
            )
        except Exception:
            pytest.fail(
                "Pytest raised an error while executing Place.execute_place from atomic_skills.py"
            )

        # Test place execution and verify with feedback
        try:
            feedback = place_controller.execute(
                place_target=test_place_targets_list[0], is_local=False
            )
            validate_place_feedback(feedback)
        except Exception:
            pytest.fail(
                "Pytest raised an error while executing Place.execute from atomic_skills.py"
            )

        test_spot.shutdown(should_dock=False)


def test_place_local():
    config = init_test_config()
    test_waypoints_yaml_dict = get_waypoint_yaml(waypoint_file=TEST_WAYPOINTS_YAML)

    test_waypoints = [
        "test_place_front_local",
        "test_place_left_local",
        "test_place_right_local",
    ]
    test_place_targets_list = [
        place_target_from_waypoint(test_waypoint, test_waypoints_yaml_dict)
        for test_waypoint in test_waypoints
    ]

    test_spot = Spot("PlaceEnvHardwareTest")
    test_result = None
    with test_spot.get_lease(hijack=True):
        test_spot.power_robot()
        place_controller = Place(spot=test_spot, config=config, use_policies=False)

        # Test place execution and verify the result
        try:
            test_result = place_controller.execute_place(
                place_target_list=test_place_targets_list, is_local=True
            )
            validate_place_results(
                test_result=test_result, config=config, test_waypoints=test_waypoints
            )
        except Exception:
            pytest.fail(
                "Pytest raised an error while executing Place.execute_place for local waypoints from atomic_skills.py"
            )

        # Test place execution and verify with feedback
        try:
            feedback = place_controller.execute(
                place_target=test_place_targets_list[0], is_local=True
            )
            validate_place_feedback(feedback)
        except Exception:
            pytest.fail(
                "Pytest raised an error while executing Place.execute for local waypoints from atomic_skills.py"
            )

        test_spot.shutdown(should_dock=False)
