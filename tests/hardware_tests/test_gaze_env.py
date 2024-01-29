# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
from typing import Dict, List, Tuple

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


def validate_pick_feedbacks(feedbacks: List[Tuple[bool, str]]):
    for feedback in feedbacks:
        (status, message) = feedback
        assert status is True
        assert message == "Successfully picked the target object"


def test_gaze():
    config = init_test_config()

    test_target_objects = ["cup", "plush_lion", "plush_bear"]
    test_spot = Spot("GazeEnvHardwareTest_StaticGaze")
    test_feedbacks = []  # type: List[Tuple[bool, str]]
    with test_spot.get_lease(hijack=True):
        test_spot.power_robot()
        gaze_controller = Pick(spot=test_spot, config=config, use_mobile_pick=False)

        # Test gaze execution and verify the result + feedback
        try:
            for target_object in test_target_objects:
                goal_dict = {
                    "target_object": target_object,
                    "take_user_input": True,
                }
                test_feedbacks.append(gaze_controller.execute(goal_dict=goal_dict))
        except Exception:
            pytest.fail(
                "Pytest raised an error while executing static Pick.execute_rl_loop() from atomic_skills.py"
            )
        finally:
            test_spot.shutdown(should_dock=False)

        # Validate pick feedbacks
        validate_pick_feedbacks(test_feedbacks)


def test_mobile_gaze():
    config = init_test_config()

    test_target_objects = ["cup", "plush_lion", "plush_bear"]
    test_spot = Spot("GazeEnvHardwareTest_MobileGaze")
    test_feedbacks = []  # type: List[Tuple[bool, str]]
    with test_spot.get_lease(hijack=True):
        test_spot.power_robot()
        gaze_controller = Pick(spot=test_spot, config=config, use_mobile_pick=True)

        # Test mobile gaze execution and verify the result + feedback
        try:
            for target_object in test_target_objects:
                goal_dict = {
                    "target_object": target_object,
                    "take_user_input": True,
                }
                test_feedbacks.append(gaze_controller.execute(goal_dict=goal_dict))
        except Exception:
            pytest.fail(
                "Pytest raised an error while executing mobile Pick.execute_rl_loop() from atomic_skills.py"
            )
        finally:
            test_spot.shutdown(should_dock=False)

        # Validate pick feedbacks
        validate_pick_feedbacks(test_feedbacks)
