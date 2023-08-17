# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os

import pytest
from spot_rl.envs.gaze_env import GazeController, construct_config_for_gaze
from spot_rl.utils.utils import get_default_parser
from spot_wrapper.spot import Spot

hardware_tests_dir = os.path.dirname(os.path.abspath(__file__))
test_configs_dir = os.path.join(hardware_tests_dir, "configs")
TEST_CONFIGS_YAML = os.path.join(test_configs_dir, "config.yaml")


def init_config():
    """
    Initialize config object for Nav test

    Returns:
        config: Config object
    """
    args = get_default_parser().parse_args()

    # Update args for gaze locally here
    args.dont_pick_up = True
    args.max_episode_steps = 150

    # Construct config for gaze
    config = construct_config_for_gaze(args, file_path=TEST_CONFIGS_YAML)

    return config


def test_gaze():
    config = init_config()

    test_target_objects = ["cup", "plush_lion", "plush_ball"]
    test_spot = Spot("GazeEnvHardwareTest")
    test_result = None
    with test_spot.get_lease(hijack=True):
        gaze_controller = GazeController(config, test_spot)

        # Test gaze
        try:
            test_result = gaze_controller.execute(
                target_object_list=test_target_objects, take_user_input=True
            )
        except Exception:
            pytest.fails(
                "Pytest raised an error while executing GazeController.execute from test_gaze_env.py"
            )
        finally:
            gaze_controller.shutdown()

    assert test_result is not None
    assert len(test_result) == len(test_target_objects)

    # Assert for all elements in test_result, "success" is True
    print(f"Gaze Results: {test_result}")
    for each_gaze_result in test_result:
        assert each_gaze_result["success"] is True
