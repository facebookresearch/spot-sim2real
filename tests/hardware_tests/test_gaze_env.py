# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os

import pytest
from spot_rl.skills.atomic_skills import Pick
from spot_rl.utils.construct_configs import construct_config_for_gaze
from spot_wrapper.spot import Spot

hardware_tests_dir = os.path.dirname(os.path.abspath(__file__))
test_configs_dir = os.path.join(hardware_tests_dir, "configs")
TEST_CONFIGS_YAML = os.path.join(test_configs_dir, "config.yaml")


def init_test_config():
    """
    Initialize config object for Nav test

    Returns:
        config: Config object
    """
    # Construct config for gaze
    dont_pick_up = True
    max_episode_steps = 150
    config = construct_config_for_gaze(
        file_path=TEST_CONFIGS_YAML,
        opts=[],
        dont_pick_up=dont_pick_up,
        max_episode_steps=max_episode_steps,
    )
    return config


def validate_pick_results(test_target_objects, test_result):
    assert test_result is not None
    assert len(test_result) == len(test_target_objects)
    # Assert for all elements in test_result, "success" is True
    print(f"Gaze Results: {test_result}")
    for each_gaze_result in test_result:
        assert each_gaze_result["success"] is True


def validate_pick_feedback(feedback):
    (status, message) = feedback
    assert status is True
    assert message == "Successfully picked the target object"


def test_gaze():
    config = init_test_config()

    test_target_objects = ["cup", "plush_lion", "plush_bear"]
    test_spot = Spot("GazeEnvHardwareTest_StaticGaze")
    test_result = None
    with test_spot.get_lease(hijack=True):
        test_spot.power_robot()
        gaze_controller = Pick(spot=test_spot, config=config, use_mobile_pick=False)

        # Test gaze execution and verify the result
        try:
            test_result = gaze_controller.execute_pick(
                target_object_list=test_target_objects, take_user_input=True
            )
            validate_pick_results(
                test_target_objects=test_target_objects, test_result=test_result
            )
        except Exception:
            pytest.fail(
                "Pytest raised an error while executing static Pick.execute_pick() from atomic_skills.py"
            )

        # Test mobile gaze execution and verify with feedback
        try:
            feedback = gaze_controller.execute(test_target_objects[0])
            validate_pick_feedback(feedback)
        except Exception:
            pytest.fail(
                "Pytest raised an error while executing static Pick.execute() from atomic_skills.py"
            )

        test_spot.shutdown(should_dock=False)


def test_mobile_gaze():
    config = init_test_config()

    test_target_objects = ["cup", "plush_lion", "plush_bear"]
    test_spot = Spot("GazeEnvHardwareTest_MobileGaze")
    test_result = None
    with test_spot.get_lease(hijack=True):
        test_spot.power_robot()
        gaze_controller = Pick(spot=test_spot, config=config, use_mobile_pick=True)

        # Test mobile gaze execution and verify the result
        try:
            test_result = gaze_controller.execute_pick(
                target_object_list=test_target_objects, take_user_input=True
            )
            validate_pick_results(
                test_target_objects=test_target_objects, test_result=test_result
            )
        except Exception:
            pytest.fail(
                "Pytest raised an error while executing mobile Pick.execute_pick() from atomic_skills.py"
            )

        # Test mobile gaze execution and verify with feedback
        try:
            feedback = gaze_controller.execute(test_target_objects[0])
            validate_pick_feedback(feedback)
        except Exception:
            pytest.fail(
                "Pytest raised an error while executing mobile Pick.execute() from atomic_skills.py"
            )

        test_spot.shutdown(should_dock=False)
