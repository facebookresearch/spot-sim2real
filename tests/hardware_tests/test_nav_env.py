import os
from typing import Dict, List

import numpy as np
import pytest
import yaml
from spot_rl.envs.nav_env import WaypointController
from spot_rl.utils.calculate_distance import is_pose_within_bounds, is_within_bounds2
from spot_rl.utils.json_helpers import load_json_file, load_json_files
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

# UPDATE THIS WITH NEW METHOD OF LOADING WAYPOINTS from util.py
with open(TEST_WAYPOINTS_YAML) as f:
    TEST_WAYPOINTS = yaml.safe_load(f)


def init_config():
    config = construct_config([])
    # Don't need gripper camera for Nav
    config.USE_MRCNN = False
    # Record the waypoints for test
    config.RECORD_TRAJECTORY = True

    return config


def compute_avg_and_std_time(ref_traj_set):
    # List of time taken to reach each waypoint for each trajectory in ref_traj_set
    time_list_traj_set = []  # type: List[List[Dict]]
    for ref_traj in ref_traj_set:
        # List of time taken to reach each waypoint for a single trajectory in ref_traj; in this test, it will contain 3 elements
        time_list_traj = []  # type: List[Dict]
        prev_time = 0
        for wp_idx in range(len(ref_traj)):
            # Append time taken to reach current waypoint to list corresponding to current trajectory
            time_taken_to_reach_curr_wp = ref_traj[wp_idx][-1]["timestamp"] - prev_time
            time_list_traj.append(time_taken_to_reach_curr_wp)

            # Update prev_time to current trajectory's last waypoint's timestamp
            prev_time = ref_traj[wp_idx][-1]["timestamp"]

        # Append trajectory's waypoint's time list to list of time taken to reach each waypoint for all trajectories
        time_list_traj_set.append(time_list_traj)

    # Convert list of time taken to reach each waypoint for all trajectories to 2-dim numpy array
    time_list_traj_set = np.array(time_list_traj_set)

    # Calculate mean and standard deviation of time taken to reach each waypoint for all trajectories (size=3 as 3 trajectories in this test)
    avg_time_list_traj = np.mean(time_list_traj_set, axis=0)
    std_time_list_traj = np.std(time_list_traj_set, axis=0)

    return avg_time_list_traj, std_time_list_traj


def extract_goal_poses_and_timings_from_traj(traj):
    prev_time = 0
    test_time_list = []
    test_pose_list = []
    num_wp_in_traj = len(traj)
    # Test that test trajectory reached each waypoint successfully
    for wp_idx in range(num_wp_in_traj):
        # Capture robot's pose, timestamp when it reached each of its goals
        data_at_last_wp = traj[wp_idx][-1]

        pose_at_last_wp = data_at_last_wp["pose"]
        test_pose_list.append(pose_at_last_wp)

        time_at_last_wp = data_at_last_wp["timestamp"]
        time_delta_for_wp = time_at_last_wp - prev_time
        prev_time = time_at_last_wp
        test_time_list.append(time_delta_for_wp)

    return test_pose_list, test_time_list


def test_nav_square():

    config = init_config()
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
    test_traj = None
    with test_spot.get_lease(hijack=True):
        wp_controller = WaypointController(
            config=config, spot=test_spot, should_record_trajectories=True
        )

        try:
            test_traj = wp_controller.execute(nav_targets=test_nav_targets)
        except Exception:
            pytest.fails(
                "Pytest raised an error while executing WaypointController.execute from test_nav_env.py"
            )
        finally:
            wp_controller.shutdown(should_dock=True)

    assert test_traj is not []
    assert len(test_traj) == len(test_waypoints)

    ref_traj_set = load_json_files(test_square_nav_trajectories_dir)
    avg_time_list_traj, std_time_list_traj = compute_avg_and_std_time(ref_traj_set)

    print(
        f"Dataset: Average time taken to reach each of the waypoint - {avg_time_list_traj}"
    )
    print(
        f"Dataset: Std Dev in time taken to reach each of the waypoint - {std_time_list_traj}"
    )

    test_pose_list, test_time_list = extract_goal_poses_and_timings_from_traj(test_traj)

    print(f"Test-Nav: Time taken to reach each of the waypoint - {test_time_list}")
    print(f"Test-Nav: Pose at each of the goal waypoint - {test_pose_list}")
    # Test that robot reached its goal both spatially and temporally for all waypoints
    for wp_idx in range(len(test_waypoints)):
        # Capture target pose for each waypoint
        target_pose = list(
            nav_target_from_waypoint(test_waypoints[wp_idx], TEST_WAYPOINTS)
        )
        target_pose[-1] = np.rad2deg(target_pose[-1])
        # Test that robot reached its goal successfully spatially
        assert (
            is_pose_within_bounds(
                test_pose_list[wp_idx],
                target_pose,
                config.SUCCESS_DISTANCE,
                config.SUCCESS_ANGLE_DIST,
            )
            is True
        )

        # Test that robot reached its goal successfully temporally (within 1 std dev of mean)
        assert (
            abs(test_time_list[wp_idx] - avg_time_list_traj[wp_idx])
            < 1.0 * std_time_list_traj[wp_idx]
        )

    # Soft tests TBD
    # Test that test trajectory took similar amount of steps to finish execution

    # Test that robot was able to reach each waypoint for all waypoints inside test trajectory

    # Report DTW scores

    # The magic number 6.0 is by trial and error. We need a better way to check bounds
    # assert (
    #     is_within_bounds2(test_traj, ref_traj_set)
    #     is True
    # )
