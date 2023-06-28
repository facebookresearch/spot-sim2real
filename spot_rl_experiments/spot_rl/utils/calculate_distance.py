import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def wrap_angle_deg(angle_deg):
    """
    Wrap an angle in degrees between 0 and 360.

    Parameters:
    - angle_deg (float): The input angle in degrees.

    Returns:
    - wrapped_angle (float): The wrapped angle in degrees between 0 and 360.
    """
    wrapped_angle = angle_deg % 360
    return wrapped_angle


def calculate_normalized_euclidean_distance_between_pose(pose1, pose2):
    """
    Calculate the distance between two poses (as dicts).

    Parameters:
    - pose1 (tuple): First pose (x, y, yaw).
    - pose2 (tuple): Second pose (x, y, yaw).

    Returns:
    - distance (float): Distance between the two poses.
    """
    max_x = 3.5  # in metres
    max_y = 2.0  # in metres
    max_yaw = 360  # in degrees

    # Wrap angles
    pose1[2] = wrap_angle_deg(np.rad2deg(pose1[2]))
    pose2[2] = wrap_angle_deg(np.rad2deg(pose2[2]))

    # Normalizaing pose
    normalizing_pose = (max_x, max_y, max_yaw)
    normalized_pose1 = np.divide(pose1, normalizing_pose)
    normalized_pose2 = np.divide(pose2, normalizing_pose)

    return euclidean(normalized_pose1[:], normalized_pose2[:])


def calculate_dtw_distance_between_trajectories(test_trajectory, reference_trajectory):
    """
    Calculate the DTW distance between a test trajectory and a reference trajectory.

    Parameters:
    - test_trajectory (list): Test trajectory containing poses.
    - reference_trajectory (list): Reference trajectory containing poses.

    Returns:
    - distance (float): DTW distance between the test and reference trajectories.
    """
    test_poses = [
        (data["pose"]["x"], data["pose"]["y"], data["pose"]["yaw"])
        for data in test_trajectory
    ]
    reference_poses = [
        (data["pose"]["x"], data["pose"]["y"], data["pose"]["yaw"])
        for data in reference_trajectory
    ]
    distance, _ = fastdtw(
        test_poses,
        reference_poses,
        dist=calculate_normalized_euclidean_distance_between_pose,
    )
    return distance


def is_within_bounds(test_trajectories, reference_trajectories, threshold):
    """
    Check if a test trajectory is within bounds of aleast one of the reference trajectories.

    Parameters:
    - test_trajectory (list): Test trajectory containing poses.
    - reference_trajectories (list): List of reference trajectories containing poses.
    - threshold (float): Threshold value for distance comparison.

    Returns:
    - is_within (bool): True if the test trajectory is within bounds of any of the reference trajectories, False otherwise.
    """
    # THIS LOGIC NEEDS TO BE UPDATED TO MAKE IT WORK WITH `test_trajectories` where there is a test trajectory for each waypoint
    # for reference_trajectory in reference_trajectories:
    #     dtw_dist = calculate_dtw_distance_between_trajectories(
    #         test_trajectory, reference_trajectory
    #     )
    #     print(f"FastDTW dist: {dtw_dist}")
    #     if dtw_dist < threshold:
    #         return True

    return False
