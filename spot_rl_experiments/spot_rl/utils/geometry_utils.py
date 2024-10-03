# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Tuple

import numpy as np
from fastdtw import fastdtw
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean


def wrap_angle_deg(angle_deg, wrapping_360=True) -> float:
    """
    Wrap an angle in degrees between 0 and 360.

    Args:
        angle_deg (float): The input angle in degrees.
        wrapping_360 (bool): If True, wraps between 0 to 360 , else -180 to 180

    Returns:
        wrapped_angle (float): The wrapped angle in degrees between 0 and 360.
    """
    wrapped_angle = angle_deg % 360

    # Return as is if wrapping between 0 and 360
    if wrapping_360:
        return wrapped_angle

    # If wrapping between -180 and 180, and current angle is greater than 180 then adjust it
    if wrapped_angle > 180:
        wrapped_angle -= 360

    return wrapped_angle


def calculate_normalized_euclidean_distance_between_pose(pose1, pose2) -> float:
    """
    Calculate the distance between two poses (as dicts).

    Args:
        pose1 (list): First pose (x, y, yaw).
        pose2 (list): Second pose (x, y, yaw).

    Returns:
        distance (float): Distance between the two poses.
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


def calculate_dtw_distance_between_trajectories(test_traj, ref_traj) -> float:
    """
    Calculate the DTW distance between a test trajectory and a reference trajectory.

    Args:
        test_traj (List[Dict]): Test trajectory containing dict of poses.
        ref_traj (List[Dict]): Reference trajectory containing dict of poses.

    Returns:
        distance (float): DTW distance between the test and reference trajectory.
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

    Args:
        test_traj (list): Test trajectory containing a list of trajectories, one for each waypoint that robot goes to.
        ref_traj_set (list): List of reference trajectories from dataset each containing a list of trajectories, one for each waypoint that robot goes to.

    Returns:
        dtw_dist_list_global (list): List of lists containing DTW distances between each waypoint trajectory in test trajectory and each waypoint trajectory in each reference trajectory.
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
    Check if a test pose (x,y,yaw) is within linear and angular bounds of target pose.

    Args:
        test_pose (list): Test pose, as [x,y,yaw]
        target_pose (list): Target pose, as [x,y,yaw]
        linear_threshold (float): Threshold for linear distance (in meters)
        angular_threshold (float): Threshold for angular distance (in degrees)

    Returns:
        is_within (bool): True if the test pose is within both linear and angular bounds of the target pose, False otherwise.
    """
    # Linear bounds
    is_within_linear_bounds = (
        euclidean(test_pose[:2], target_pose[:2]) < linear_threshold
    )

    # Angular bounds
    angular_delta = abs(wrap_angle_deg(test_pose[2]) - wrap_angle_deg(target_pose[2]))
    is_within_angular_bounds = (
        min(angular_delta, 360 - angular_delta) < angular_threshold
    )

    return bool(is_within_linear_bounds and is_within_angular_bounds)


def is_position_within_bounds(
    test_position: np.array,
    target_position: np.array,
    xy_dist_threshold: float,
    z_dist_threshold: float,
    convention: str = "spot",
) -> bool:
    """
    Check if a test position(x,y,z) is within  of target pose.

    Args:
        test_position (np.array): Test position, as either [x,y,z] or [x,z,-y]
        target_position (np.array): Target position, as [x,y,z] or [x,z,-y]
        xy_dist_threshold (float): Threshold for xy distance (in meters)
        z_dist_threshold (float): Threshold for z distance (in meters)
        convention (str): Convention for reference frames, either "spot" or "habitat" (default: spot)

    Returns:
        is_within (bool): True if the test pose is within the zy & z bounds of the target pose, False otherwise.
    """

    # Linear xy dist, and z dist bounds
    is_within_linear_xy_bounds = False
    is_within_linear_z_bounds = False

    if convention == "spot":
        is_within_linear_xy_bounds = (
            np.linalg.norm(test_position[:2] - target_position[:2]) < xy_dist_threshold
        )
        is_within_linear_z_bounds = (
            abs(test_position[2] - target_position[2]) < z_dist_threshold
        )
    elif convention == "habitat":
        is_within_linear_xy_bounds = (
            np.linalg.norm(test_position[[0, 2]] - target_position[[0, 2]])
            < xy_dist_threshold
        )
        is_within_linear_z_bounds = (
            abs(test_position[1] - target_position[1]) < z_dist_threshold
        )
    else:
        raise NotImplementedError

    return bool(is_within_linear_xy_bounds and is_within_linear_z_bounds)


def interpolation_between_two_positions(point1, point2, steps=10, kind="linear"):
    """
    Interpolate between two 3D points

    Args:
        point1 (tuple): The first 3D point
        point2 (tuple): The second 3D point
        steps (int): The number of steps to interpolate
        kind (str): The type of interpolation to use (default: linear)

    Returns:
        interpolated_points (List[Tuple]): A list of interpolated points
    """
    # Define the two 3D points
    x1, y1, z1 = point1
    x2, y2, z2 = point2

    # Create an array of x,y,z values for interpolation
    x = np.array([x1, x2])
    y = np.array([y1, y2])
    z = np.array([z1, z2])

    # Create a 1d interpolation object object for each dimension
    f_x = interp1d(x, y, kind=kind)
    f_y = interp1d(x, y, kind=kind)
    f_z = interp1d(x, z, kind=kind)

    # Generate the interpolated points
    interpolated_points = []  # type: List[Tuple]
    for t in np.linspace(x1, x2, steps):
        interpolated_x = float(f_x(t))
        interpolated_y = float(f_y(t))
        interpolated_z = float(f_z(t))
        interpolated_points.append((interpolated_x, interpolated_y, interpolated_z))

    # Return the interpolated points
    return interpolated_points


def generate_intermediate_point(point1, point2, z_elevation=0.2):
    """
    Generate an intermediate point between two 3D points

    Args:
        point1 (tuple): The first 3D point
        point2 (tuple): The second 3D point
        z_elevation (float): Additional elevation (along z) to be imparted to the intermediate point

    Returns:
        intermediate_point (tuple): The intermediate 3D point
    """
    # Define the two 3D points
    x1, y1, z1 = point1
    x2, y2, z2 = point2

    # Generate the intermediate point
    intermediate_point = ((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2 + z_elevation)

    # Return the intermediate point
    return intermediate_point


def get_RPY_from_vector(vect):
    """
    Get the roll, pitch, yaw angles from a vector in radians.
    Roll=pi/2 as it cannot be computed from a vector in 3D space.

    Args:
        vect (np.array): The direction vector (linear np.array of size=3)

    Returns:
        rpy_list (List[float]): The roll, pitch, yaw angles in radians (where roll=0)
    """
    # Get the yaw angle
    yaw = np.arctan2(vect[1], vect[0])

    # Get the pitch angle
    pitch = np.arctan2(vect[2], np.sqrt(vect[0] ** 2 + vect[1] ** 2))

    # Roll cannot be computed from a vector in space, setting it as pi/2 so objects are easy to drop from gripper
    roll = 1.57

    # Return the roll, pitch, yaw angles as a list
    rpy_list = [roll, pitch, yaw]  # type: List
    return rpy_list
