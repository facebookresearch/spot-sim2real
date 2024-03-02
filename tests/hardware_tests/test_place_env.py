# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
from typing import Dict, List, Tuple

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


def validate_place_feedbacks(feedbacks: List[Tuple[bool, str]]):
    for feedback in feedbacks:
        (status, message) = feedback
        assert status is True
        assert message == "Successfully reached the target position"


def test_place_with_policy():
    config = init_test_config()
    test_waypoints_yaml_dict = get_waypoint_yaml(waypoint_file=TEST_WAYPOINTS_YAML)

    test_waypoints = [
        "test_place_front",
    ]
    test_place_targets_list = [
        place_target_from_waypoint(test_waypoint, test_waypoints_yaml_dict)
        for test_waypoint in test_waypoints
    ]

    test_spot = Spot("PlaceEnvHardwareTest")
    test_feedbacks = []  # type: List[Tuple[bool, str]]
    with test_spot.get_lease(hijack=True):
        test_spot.power_robot()
        place_controller = Place(spot=test_spot, config=config, use_policies=True)

        # Test place execution and verify the result + feedback
        try:
            for place_target in test_place_targets_list:
                goal_dict = {
                    "place_target": place_target,
                    "is_local": False,
                }
                test_feedbacks.append(place_controller.execute(goal_dict=goal_dict))

        except Exception:
            pytest.fail(
                "Pytest raised an error while executing Place.execute() from atomic_skills.py"
            )
        finally:
            test_spot.shutdown(should_dock=False)

        # Validate place feedback
        validate_place_feedbacks(test_feedbacks)


def test_place_without_policy():
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
    test_feedbacks = []  # type: List[Tuple[bool, str]]
    with test_spot.get_lease(hijack=True):
        test_spot.power_robot()
        place_controller = Place(spot=test_spot, config=config, use_policies=False)

        # Test place execution and verify the result + feedback
        try:
            for place_target in test_place_targets_list:
                goal_dict = {
                    "place_target": place_target,
                    "is_local": False,
                }
                test_feedbacks.append(place_controller.execute(goal_dict=goal_dict))
        except Exception:
            pytest.fail(
                "Pytest raised an error while executing Place.execute() from atomic_skills.py"
            )
        finally:
            test_spot.shutdown(should_dock=False)

        # Validate place feedback
        validate_place_feedbacks(test_feedbacks)


def test_place_local_with_policy():
    config = init_test_config()
    test_waypoints_yaml_dict = get_waypoint_yaml(waypoint_file=TEST_WAYPOINTS_YAML)

    test_waypoints = [
        "test_place_front_local",
    ]
    test_place_targets_list = [
        place_target_from_waypoint(test_waypoint, test_waypoints_yaml_dict)
        for test_waypoint in test_waypoints
    ]

    test_spot = Spot("PlaceEnvHardwareTest")
    test_feedbacks = []  # type: List[Tuple[bool, str]]
    with test_spot.get_lease(hijack=True):
        test_spot.power_robot()
        place_controller = Place(spot=test_spot, config=config, use_policies=True)

        # Test place execution and verify the result + feedback
        try:
            for place_target in test_place_targets_list:
                goal_dict = {
                    "place_target": place_target,
                    "is_local": True,
                }
                test_feedbacks.append(place_controller.execute(goal_dict=goal_dict))
        except Exception:
            pytest.fail(
                "Pytest raised an error while executing Place.execute_place for local waypoints from atomic_skills.py"
            )
        finally:
            test_spot.shutdown(should_dock=False)

        # Validate place feedback
        validate_place_feedbacks(test_feedbacks)


def test_place_local_without_policy():
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
    test_feedbacks = []  # type: List[Tuple[bool, str]]
    with test_spot.get_lease(hijack=True):
        test_spot.power_robot()
        place_controller = Place(spot=test_spot, config=config, use_policies=False)

        # Test place execution and verify the result + feedback
        try:
            for place_target in test_place_targets_list:
                goal_dict = {
                    "place_target": place_target,
                    "is_local": True,
                }
                test_feedbacks.append(place_controller.execute(goal_dict=goal_dict))
        except Exception:
            pytest.fail(
                "Pytest raised an error while executing Place.execute_place for local waypoints from atomic_skills.py"
            )
        finally:
            test_spot.shutdown(should_dock=False)

        # Validate place feedback
        validate_place_feedbacks(test_feedbacks)
