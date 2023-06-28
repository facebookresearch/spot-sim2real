import os

import pytest
import yaml
from spot_rl.envs.nav_env import WaypointController
from spot_rl.utils.calculate_distance import is_within_bounds
from spot_rl.utils.json_helpers import load_json_files
from spot_rl.utils.utils import construct_config, nav_target_from_waypoint
from spot_wrapper.spot import Spot

hardware_tests_dir = os.path.dirname(os.path.abspath(__file__))
test_configs_dir = os.path.join(hardware_tests_dir, "configs")
test_data_dir = os.path.join(hardware_tests_dir, "data")
test_nav_trajectories_dir = os.path.join(test_data_dir, "nav_trajectories")
test_square_nav_trajectories_dir = os.path.join(
    test_nav_trajectories_dir, "square_of_side_200cm"
)
TEST_WAYPOINTS_YAML = os.path.join(test_configs_dir, "waypoints.yaml")

with open(TEST_WAYPOINTS_YAML) as f:
    TEST_WAYPOINTS = yaml.safe_load(f)


def test_nav():

    config = construct_config([])
    # Don't need gripper camera for Nav
    config.USE_MRCNN = False
    # Record the waypoints for test
    config.RECORD_TRAJECTORY = True

    test_waypoints = [
        "test_square_vertex1",
        "test_square_vertex2",
        "test_square_vertex3",
    ]
    test_nav_targets = [
        nav_target_from_waypoint(test_waypoint, TEST_WAYPOINTS)
        for test_waypoint in test_waypoints
    ]

    test_spot = Spot("NavEnvHardwareTest")
    with test_spot.get_lease(hijack=True):
        wp_controller = WaypointController(
            config=config, spot=test_spot, should_record_trajectories=True
        )

        try:
            test_robot_trajectories = wp_controller.execute(
                nav_targets=test_nav_targets
            )
        except Exception:
            pytest.fails(
                "Pytest raised an error while executing WaypointController.execute from test_nav_env.py"
            )
        finally:
            wp_controller.shutdown(should_dock=True)

        print("test_robot_trajectories", test_robot_trajectories)
        assert test_robot_trajectories is not []
        assert len(test_robot_trajectories) == len(test_waypoints)

        test_ref_nav_trajectories = load_json_files(test_square_nav_trajectories_dir)

        # Test that test trajectory took similar amount of steps to finish execution

        # Test that robot was able to reach each waypoint for all waypoints inside test trajectory

        # Report DTW scores

        # The magic number 6.0 is by trial and error. We need a better way to check bounds
        assert (
            is_within_bounds(test_robot_trajectories, test_ref_nav_trajectories, 6.0)
            is True
        )
