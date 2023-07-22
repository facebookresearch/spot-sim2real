# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import List

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def wrap_angle_deg(angle_deg) -> float:
    """
    Wrap an angle in degrees between 0 and 360.

    Parameters:
    - angle_deg (float): The input angle in degrees.

    Returns:
    - wrapped_angle (float): The wrapped angle in degrees between 0 and 360.
    """
    wrapped_angle = angle_deg % 360
    return wrapped_angle


def calculate_normalized_euclidean_distance_between_pose(pose1, pose2) -> float:
    """
    Calculate the distance between two poses (as dicts).

    Parameters:
    - pose1 (list): First pose (x, y, yaw).
    - pose2 (list): Second pose (x, y, yaw).

    Returns:
    - distance (float): Distance between the two poses.
    """
    max_x = 3.5  # in metres
    max_y = 2.0  # in metres
    max_yaw = 360  # in degrees

    # Wrap angles
    pose1[2] = wrap_angle_deg(pose1[2])
    pose2[2] = wrap_angle_deg(pose2[2])

    # Normalizaing pose
    normalizing_pose = [max_x, max_y, max_yaw]
    normalized_pose1 = np.divide(pose1, normalizing_pose)
    normalized_pose2 = np.divide(pose2, normalizing_pose)

    return euclidean(normalized_pose1[:], normalized_pose2[:])


def calculate_dtw_distance_between_trajectories(test_traj, ref_traj):
    """
    Calculate the DTW distance between a test trajectory and a reference trajectory.

    Parameters:
    - test_traj (list): Test trajectory containing poses.
    - ref_traj (list): Reference trajectory containing poses.

    Returns:
    - distance (float): DTW distance between the test and reference trajectories.
    """
    test_poses = [data["pose"] for data in test_traj]
    reference_poses = [data["pose"] for data in ref_traj]
    distance, _ = fastdtw(
        test_poses,
        reference_poses,
        dist=calculate_normalized_euclidean_distance_between_pose,
    )
    return distance


def compute_dtw_scores(test_traj, ref_traj_set) -> List[List]:
    """
    Check if a test trajectory is within bounds of aleast one of the reference trajectories.

    Parameters:
    - test_traj (list): Test trajectory containing a list of trajectories, one for each waypoint that robot goes to.
    - ref_traj_set (list): List of reference trajectories from dataset each containing a list of trajectories, one for each waypoint that robot goes to.

    Returns:
    - dtw_dist_list_global (list): List of lists containing DTW distances between each waypoint trajectory in test trajectory and each waypoint trajectory in each reference trajectory.
    """

    # Create a list of lists (size = num_of_waypoints_in_test_traj X num_of_ref_trajs_in_dataset)
    dtw_dist_list_global = []

    # For each waypoint trajectory in test trajectory
    for wp_idx in range(len(test_traj)):
        dtw_dist_list_wp = []
        wp_test_traj = test_traj[wp_idx]
        # For each reference trajectory in dataset
        for ref_traj in ref_traj_set:
            wp_ref_traj = ref_traj[wp_idx]

            # Calculate the DTW distance between the two trajectories
            dtw_dist_wp = calculate_dtw_distance_between_trajectories(
                wp_test_traj, wp_ref_traj
            )

            # Append the DTW distance to the list of DTW distances for the current waypoint trajectory
            dtw_dist_list_wp.append(dtw_dist_wp)
        # Append the list of DTW distances for the current waypoint trajectory to the global list
        dtw_dist_list_global.append(dtw_dist_list_wp)

    return dtw_dist_list_global


def is_pose_within_bounds(
    test_pose, target_pose, linear_threshold, angular_threshold
) -> bool:
    """
    Check if a test pose is within linear and angular bounds of target pose.

    Parameters:
    - test_pose (list): Test pose, as [x,y,yaw]
    - target_pose (list): Target pose, as [x,y,yaw]

    Returns:
    - is_within (bool): True if the test pose is within both linear and angular bounds of the target pose, False otherwise.
    """
    # Linear bounds
    # print(f"Euclidean: {euclidean(test_pose[:2], target_pose[:2])}")
    # print(f"Angular: {abs(wrap_angle_deg(test_pose[2]) - wrap_angle_deg(target_pose[2]))}")
    is_within_linear_bounds = (
        euclidean(test_pose[:2], target_pose[:2]) < linear_threshold
    )
    angular_delta = abs(wrap_angle_deg(test_pose[2]) - wrap_angle_deg(target_pose[2]))
    is_within_angular_bounds = (
        min(angular_delta, 360 - angular_delta) < angular_threshold
    )

    # print(f"test_pose: {test_pose}")
    # print(f"target_pose: {target_pose}")
    # print(f"is_within_linear_bounds: {is_within_linear_bounds}")
    # print(f"is_within_angular_bounds: {is_within_angular_bounds}")
    return bool(is_within_linear_bounds and is_within_angular_bounds)
