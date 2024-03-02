# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
from typing import Dict, List, Tuple

import numpy as np
import pytest
from spot_rl.skills.atomic_skills import Navigation
from spot_rl.utils.construct_configs import construct_config_for_nav
from spot_rl.utils.geometry_utils import compute_dtw_scores, is_pose_within_bounds
from spot_rl.utils.json_helpers import load_json_files
from spot_rl.utils.utils import get_waypoint_yaml, nav_target_from_waypoint
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
    config = construct_config_for_nav(file_path=TEST_CONFIGS_YAML, opts=[])
    # Record the waypoints for test
    config.RECORD_TRAJECTORY = True

    return config


def compute_avg_and_std_time(ref_traj_set: List[List[List[Dict]]]) -> Tuple[List, List]:
    """
    Compute average and standard deviation of time taken to reach each waypoint for all trajectories in ref_traj_set

    Args:
        ref_traj_set: A list of recorded trajectories - List(List(List(Dict)))

        Returns:
            avg_time_list_traj_set: List of average time taken to reach each waypoint for all trajectories (size = num of waypoints)
            std_time_list_traj_set: List of standard deviation of time taken to reach each waypoint for all trajectories (size = num of waypoints)
    """

    # List of time taken to reach each waypoint for each trajectory in ref_traj_set
    time_list_traj_set = []  # type: List[List[Dict]]
    for ref_traj in ref_traj_set:
        # List of time taken to reach each waypoint for a single trajectory in ref_traj; in this test, it will contain 3 elements
        time_list_traj = []  # type: List[Dict]
        for wp_idx in range(len(ref_traj)):
            # Append time taken to reach current waypoint to list corresponding to current trajectory
            time_taken_to_reach_curr_wp = (
                ref_traj[wp_idx][-1]["timestamp"] - ref_traj[wp_idx][0]["timestamp"]
            )
            time_list_traj.append(time_taken_to_reach_curr_wp)

        # Append trajectory's waypoint's time list to list of time taken to reach each waypoint for all trajectories
        time_list_traj_set.append(time_list_traj)

    # Convert list of time taken to reach each waypoint for all trajectories to 2-dim numpy array
    time_list_traj_set = np.array(time_list_traj_set)

    # Calculate mean and standard deviation of time taken to reach each waypoint for all trajectories (size=3 as 3 trajectories in this test)
    avg_time_list_traj = np.mean(time_list_traj_set, axis=0)
    std_time_list_traj = np.std(time_list_traj_set, axis=0)
    return avg_time_list_traj, std_time_list_traj


def compute_avg_and_std_steps(
    ref_traj_set: List[List[List[Dict]]],
) -> Tuple[List, List]:
    """
    Computes average and standard deviation of steps taken to reach each waypoint for all trajectories in ref_traj_set

    Args:
        ref_traj_set: A list of recorded trajectories - List(List(List(Dict)))

    Returns:
        avg_steps_list_traj: list of average steps taken to reach each waypoint for all trajectories (size = num of waypoints)
        std_steps_list_traj: list of standard deviation of steps taken to reach each waypoint for all trajectories (size = num of waypoints)
    """

    # List of steps taken to reach each waypoint for each trajectory in ref_traj_set
    steps_list_traj_set = []  # type: List[List[int]]
    for ref_traj in ref_traj_set:
        # List of steps taken to reach each waypoint for a single trajectory in ref_traj; in this test, it will contain 3 elements
        steps_list_traj = []  # type: List[int]
        for wp_idx in range(len(ref_traj)):
            # Append steps taken to reach current waypoint to list corresponding to current trajectory
            steps_taken_to_reach_curr_wp = len(ref_traj[wp_idx])
            steps_list_traj.append(steps_taken_to_reach_curr_wp)

        # Append trajectory's waypoint's steps list to list of steps taken to reach each waypoint for all trajectories
        steps_list_traj_set.append(steps_list_traj)

    # Convert list of steps taken to reach each waypoint for all trajectories to 2-dim numpy array
    steps_list_traj_set = np.array(steps_list_traj_set)

    # Calculate mean and standard deviation of steps taken to reach each waypoint for all trajectories (size=3 as 3 trajectories in this test)
    avg_steps_list_traj = np.mean(steps_list_traj_set, axis=0)
    std_steps_list_traj = np.std(steps_list_traj_set, axis=0)

    return avg_steps_list_traj, std_steps_list_traj


def extract_goal_poses_timestamps_steps_from_traj(
    trajs: List[List[Dict]],
) -> Tuple[List, List, List]:
    """
    Extracts actual robot pose, timestamp, and number of steps that robot took to reach each of nav_targets

    Args:
        trajs: trajectory (a list for each waypoint) to extract goal poses, timestamps, and steps from

    Returns:
        goal_poses: list of actual robot poses at each waypoint
        timestamps: list of timestamps when robot reached each waypoint
        steps: list of steps robot took to reach each waypoint
    """
    prev_time = 0
    test_time_list = []
    test_pose_list = []
    test_step_list = []
    num_wp_in_traj = len(trajs)
    # Test that test trajectory reached each waypoint successfully
    for wp_idx in range(num_wp_in_traj):
        # Capture robot's pose, timestamp when it reached each of its goals
        data_at_last_wp = trajs[wp_idx][-1]

        pose_at_last_wp = data_at_last_wp["pose"]
        test_pose_list.append(pose_at_last_wp)

        time_at_last_wp = data_at_last_wp["timestamp"]
        time_delta_for_wp = time_at_last_wp - prev_time
        prev_time = time_at_last_wp
        test_time_list.append(time_delta_for_wp)

        test_step_list.append(len(trajs[wp_idx]))

    return test_pose_list, test_time_list, test_step_list


def validate_nav_trajectories(
    test_waypoints: List[str],
    test_waypoints_yaml_dict: Dict,
    test_trajs: List[List[Dict]],
    config,
):
    assert test_trajs is not []
    assert len(test_trajs) == len(test_waypoints)

    ref_traj_set = load_json_files(test_square_nav_trajectories_dir)

    avg_time_list_traj, std_time_list_traj = compute_avg_and_std_time(ref_traj_set)
    avg_steps_list_traj, std_steps_list_traj = compute_avg_and_std_steps(ref_traj_set)
    (
        test_pose_list,
        test_time_list,
        test_steps_list,
    ) = extract_goal_poses_timestamps_steps_from_traj(test_trajs)

    print(f"Dataset: Avg. time to reach each waypoint - {avg_time_list_traj}")
    print(f"Dataset: Std.Dev in time reach each waypoint - {std_time_list_traj}")
    print(f"Test-Nav: Time taken to reach each waypoint - {test_time_list}\n")

    print(f"Dataset: Avg. steps to reach each waypoint - {avg_steps_list_traj}")
    print(f"Dataset: Std.Dev in steps to reach each waypoint - {std_steps_list_traj}")
    print(f"Test-Nav: Steps taken to reach each waypoint - {test_steps_list}\n")

    print(f"Test-Nav: Pose at each of the goal waypoint - {test_pose_list}\n")

    allowable_std_dev_in_time = 3.0
    allowable_std_dev_in_steps = 3.0
    for wp_idx in range(len(test_waypoints)):
        # Capture target pose for each waypoint
        target_pose = list(
            nav_target_from_waypoint(test_waypoints[wp_idx], test_waypoints_yaml_dict)
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
            < allowable_std_dev_in_time * std_time_list_traj[wp_idx]
        )

        # Test that test trajectory took similar amount of steps to finish execution
        assert (
            abs(test_steps_list[wp_idx] - avg_steps_list_traj[wp_idx])
            < allowable_std_dev_in_steps * std_steps_list_traj[wp_idx]
        )

    # Report DTW scores
    dtw_score_list = compute_dtw_scores(test_trajs, ref_traj_set)
    print(f"DTW scores: {dtw_score_list}")


def validate_nav_feedbacks(feedback_list: List[Tuple[bool, str]]):
    for feedback in feedback_list:
        (status, message) = feedback
        assert status is True
        assert message == "Successfully reached the target pose by default"


def test_nav_square():
    config = init_test_config()
    test_waypoints_yaml_dict = get_waypoint_yaml(waypoint_file=TEST_WAYPOINTS_YAML)

    test_waypoints = [
        "test_square_vertex1",
        "test_square_vertex2",
        "test_square_vertex3",
    ]
    test_nav_targets_list = [
        nav_target_from_waypoint(test_waypoint, test_waypoints_yaml_dict)
        for test_waypoint in test_waypoints
    ]

    test_spot = Spot("NavEnvHardwareTest")
    test_trajs = []  # type: List[List[Dict]]
    test_feedbacks = []  # type: List[Tuple[bool,str]]
    with test_spot.get_lease(hijack=True):
        test_spot.power_robot()
        nav_controller = Navigation(spot=test_spot, config=config)

        # Test navigation execution and verify result + feedback
        try:
            for nav_target in test_nav_targets_list:
                goal_dict = {"nav_target": nav_target}
                test_feedbacks.append(nav_controller.execute(goal_dict=goal_dict))
                test_trajs.append(
                    nav_controller.get_most_recent_result_log().get("robot_trajectory")
                )
        except Exception:
            pytest.fail(
                "Pytest raised an error while executing Navigation.execute_rl_loop() from atomic_skills.py"
            )
        finally:
            test_spot.shutdown(should_dock=False)

        # Validate navigation trajectories
        validate_nav_trajectories(
            test_waypoints, test_waypoints_yaml_dict, test_trajs, config
        )

        # Validate navigation feedback
        validate_nav_feedbacks(test_feedbacks)
