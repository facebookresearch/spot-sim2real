from typing import List, Tuple

import numpy as np
from spot_rl.utils.path_planning import (
    PCD_PATH,
    angle_between_vectors,
    get_xyzxyz,
    midpoint,
    path_planning_using_a_star,
    pkl,
)

STATIC_OFFSET = 0.5


def angle_and_sign_between_vectors(a, b):
    """
    Calculate the angle and sign of the angle between two 2D vectors.

    Parameters:
    a (tuple): The first vector as (a_x, a_y).
    b (tuple): The second vector as (b_x, b_y).

    Returns:
    float: The angle between the vectors in radians.
    int: The sign of the angle (+1 for counterclockwise, -1 for clockwise).
    """
    # Convert the input tuples to numpy arrays
    a = np.array(a)
    b = np.array(b)

    # Calculate the dot product and magnitudes
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)

    # Calculate the angle using the dot product formula
    angle = np.arccos(dot_product / (magnitude_a * magnitude_b))

    # Calculate the cross product (in 2D, this is a scalar)
    cross_product = a[0] * b[1] - a[1] * b[0]

    # Determine the sign of the angle
    sign = 1 if cross_product > 0 else -1

    # Apply the sign to the angle
    signed_angle = sign * np.rad2deg(angle)

    return signed_angle


# don't remove this keep it incase new logic fails
def intersect_ray_with_aabb(ray_origin, ray_direction, box_min, box_max):
    t_min = (box_min[0] - ray_origin[0]) / ray_direction[0]
    t_max = (box_max[0] - ray_origin[0]) / ray_direction[0]

    if t_min > t_max:
        t_min, t_max = t_max, t_min

    ty_min = (box_min[1] - ray_origin[1]) / ray_direction[1]
    ty_max = (box_max[1] - ray_origin[1]) / ray_direction[1]

    if ty_min > ty_max:
        ty_min, ty_max = ty_max, ty_min

    if (t_min > ty_max) or (ty_min > t_max):
        return False, None, None

    if ty_min > t_min:
        t_min = ty_min

    if ty_max < t_max:
        t_max = ty_max

    tz_min = (box_min[2] - ray_origin[2]) / ray_direction[2]
    tz_max = (box_max[2] - ray_origin[2]) / ray_direction[2]

    if tz_min > tz_max:
        tz_min, tz_max = tz_max, tz_min

    if (t_min > tz_max) or (tz_min > t_max):
        return False, None, None

    if tz_min > t_min:
        t_min = tz_min

    if tz_max < t_max:
        t_max = tz_max

    # If t_min is negative, the intersection is behind the ray origin
    if t_min < 0 and t_max < 0:
        return False, None, None

    STATIC_OFFSET = 0.0
    print("t_min", t_min)
    t_min -= STATIC_OFFSET if t_min > STATIC_OFFSET else 0.0
    # Return the intersection points (if needed)
    intersection_point_1 = ray_origin + t_min * ray_direction
    intersection_point_2 = ray_origin + t_max * ray_direction

    return True, intersection_point_1, intersection_point_2, t_min, t_max


def determin_nearest_edge(
    robot_xy, bbox_centers, boxMin, boxMax, static_offset=0.0, nonreachable_indices=[]
):
    if len(bbox_centers) == 3:
        bbox_centers = bbox_centers[:2]
    assert len(nonreachable_indices) < 4, "all edges seem to be non reachable"
    raydir = (bbox_centers - robot_xy) / np.linalg.norm(bbox_centers - robot_xy)
    STATIC_OFFSET = static_offset
    (x1, y1), (x2, y2) = boxMin[:2], boxMax[:2]
    face_1 = np.array([midpoint(x1, x2), y1 - 0.0])
    face_1_vector = bbox_centers - face_1  # x1,y1, x2,y1
    face_1_vector = face_1_vector / np.linalg.norm(face_1_vector)
    face_1_adjusted = np.array([midpoint(x1, x2), y1 - STATIC_OFFSET])

    face_2 = np.array([midpoint(x1, x2), y2 + 0.0])
    face_2_vector = bbox_centers - face_2  # x1,y2, x2,y2
    face_2_vector = face_2_vector / np.linalg.norm(face_2_vector)
    face_2_adjusted = np.array([midpoint(x1, x2), y2 + STATIC_OFFSET])

    face_3 = np.array([x1 - 0.0, midpoint(y1, y2)])
    face_3_vector = bbox_centers - face_3  # x1, y1, x1, y2
    face_3_vector = face_3_vector / np.linalg.norm(face_3_vector)
    face_3_adjusted = np.array([x1 - STATIC_OFFSET, midpoint(y1, y2)])

    face_4 = np.array([x2 + 0.0, midpoint(y1, y2)])
    face_4_vector = bbox_centers - face_4  # x2, y1, x2, y2
    face_4_vector = face_4_vector / np.linalg.norm(face_4_vector)
    face_4_adjusted = np.array([x2 + STATIC_OFFSET, midpoint(y1, y2)])

    faces = [
        (face_1_vector, face_1_adjusted),
        (face_2_vector, face_2_adjusted),
        (face_3_vector, face_3_adjusted),
        (face_4_vector, face_4_adjusted),
    ]
    angles_betwn_approach_vector_and_faces = [
        angle_between_vectors(raydir[:2], face[0])[1]
        if idx not in nonreachable_indices
        else 180
        for idx, face in enumerate(faces)
    ]
    min_idx = np.argmin(angles_betwn_approach_vector_and_faces)
    _ = angles_betwn_approach_vector_and_faces[min_idx]
    nearestfacevector, nearestface = faces[min_idx]

    yaw_calc = angle_and_sign_between_vectors(np.array([1, 0]), nearestfacevector)
    other_waypoints = [
        face[1].tolist() + [angle_and_sign_between_vectors(np.array([1, 0]), face[0])]
        for face in faces
    ]
    return nearestface, nearestfacevector, yaw_calc, other_waypoints


def get_max_diag_dist_using_pcd():
    with open(PCD_PATH, "rb") as file:
        pcd_numpy = pkl.load(file)
        min_x, max_x = pcd_numpy[:, 0].min(), pcd_numpy[:, 0].max()
        min_y, max_y = pcd_numpy[:, 1].min(), pcd_numpy[:, 1].max()
        min_z, max_z = pcd_numpy[:, -1].min(), pcd_numpy[:, -1].max()
        min_end = np.array([min_x, min_y, min_z])
        max_end = np.array([max_x, max_y, max_z])
        return np.linalg.norm(max_end - min_end)


def sort_robot_view_poses_from_cg(
    robot_view_poses, bbox_centers: np.ndarray, alpha: float, beta: float, gamma: float
):
    # robot view pose [(x,y,yaw), detectionconf, pixel_area]
    # bbox centers, bbox extents of a receptacle from CG
    rank_array = [float("-inf")] * len(robot_view_poses)
    max_pixel_area = 640 * 480
    max_diag_dist = get_max_diag_dist_using_pcd()
    for i, robot_view_pose in enumerate(robot_view_poses):
        (x, y, yaw), detectionconf, pixel_area = robot_view_pose
        # Compute the rank based on the detection confidence, euclidean distance and pixel area
        euclidean_dist = np.linalg.norm(bbox_centers[:2] - np.array([x, y]))
        rank = (
            -alpha * euclidean_dist / max_diag_dist
            + beta * detectionconf
            + gamma * (pixel_area / max_pixel_area)
        )
        rank_array[i] = (rank, i)  # type: ignore
    rank_array = sorted(rank_array, reverse=True, key=lambda x: x[0])  # type: ignore
    robot_view_poses_sorted = [robot_view_poses[rank[1]] for rank in rank_array]  # type: ignore
    return robot_view_poses_sorted


def get_waypoint_from_robot_view_poses(
    robot_view_poses,
    bbox_centers: np.ndarray,
    bbox_extents: np.ndarray,
    nonreachable_edges_indices=[],
):
    """Select the best robot navigation waypoint from the robot view poses"""

    # Sort the robot view poses based on the detection confidence, euclidean distance and pixel area
    sorted_robot_view_poses = sort_robot_view_poses_from_cg(
        robot_view_poses, bbox_centers, 0.33, 0.33, 0.33
    )

    # Select the best robot view pose
    best_robot_view_pos = sorted_robot_view_poses[0]
    print(f"Best robot view pose {best_robot_view_pos}")

    boxMin, boxMax = get_xyzxyz(bbox_centers, bbox_extents)

    # Determine the nearest edge from the best_robot_view_pos to the reachable edges
    nearestedge, facevector, yaw_calc, other_waypoints = determin_nearest_edge(
        best_robot_view_pos[0][:2],
        bbox_centers,
        boxMin,
        boxMax,
        STATIC_OFFSET,
        nonreachable_edges_indices,
    )
    waypoint = nearestedge.tolist()
    waypoint.append(yaw_calc)
    return waypoint, best_robot_view_pos, other_waypoints


def get_navigation_points(
    robot_view_pose_data=None,
    bbox_centers=np.array([8.2, 6.0, 0.1]),
    bbox_extents=np.array([1.3, 1.0, 0.8]),
    cur_robot_xy=[0, 0],
    visualize=False,
    savefigname=None,
):
    """Get navigation points for a given object in the scene."""

    # Get the bounding box center and extents
    boxMin, boxMax = get_xyzxyz(bbox_centers, bbox_extents)

    assert robot_view_pose_data is not None, "Do not have robot view poses"

    robot_view_poses = []
    for robot_view_pose in robot_view_pose_data:
        robot_view_poses.append(
            [
                tuple(
                    robot_view_pose["robot_xy_yaw"].tolist()
                ),  # the pose of the robot
                float(robot_view_pose["conf"]),  # prediction confidence
                int(
                    robot_view_pose["pixel_area"]
                ),  # the pixel location of the object in the image
            ]
        )

    # Get the locations of the four edge point of the bounding box from top view
    _, _, _, four_waypoint_edges = determin_nearest_edge(
        np.array([0, 0]), bbox_centers, boxMin, boxMax, static_offset=STATIC_OFFSET
    )

    # filter edges based on reachability from the current robot location to four_waypoint_edges
    nonreachable_edges_indices = []
    reachable_indices = []
    reachable_paths = []
    for waypoint_i, waypoint_edge in enumerate(four_waypoint_edges):
        # Do a* path planning from current robot location to the edge point
        path_edge = path_planning_using_a_star(cur_robot_xy, waypoint_edge[:2])
        if not len(path_edge):
            # Do not find the path, mark the edge as non reachable (the edge in inside the clutter)
            nonreachable_edges_indices.append(waypoint_i)
        else:
            reachable_indices.append(waypoint_i)
            reachable_paths.append(path_edge)
    print(f"non reachable indices {nonreachable_edges_indices}")

    if len(nonreachable_edges_indices) == 4:
        # either raise error or consider it path planning failure & continue
        nonreachable_edges_indices = []

    if len(nonreachable_edges_indices) == 3:
        # no need to test robot view poses since only 1 edge is reachable
        waypoint = four_waypoint_edges[reachable_indices[0]]
        path = reachable_paths[0]
        best_robot_view_pos = robot_view_poses[0]
    else:
        # Do view poses selection if there are more than 1 reachable edges
        (
            waypoint,
            best_robot_view_pos,
            other_waypoints,
        ) = get_waypoint_from_robot_view_poses(
            robot_view_poses, bbox_centers, bbox_extents, nonreachable_edges_indices
        )

    # Finally do the path planning from current robot location to the selected waypoint
    path = path_planning_using_a_star(
        cur_robot_xy,
        waypoint[:2],
        savefigname,
        visualize=visualize,
        other_view_poses=[view_pose[0][:2] for view_pose in robot_view_poses],  # type: ignore
        all_faces=four_waypoint_edges if visualize else None,
        best_view_pose=best_robot_view_pos[0][:2] if visualize else None,
    )
    waypoint[-1] = np.deg2rad(waypoint[-1])
    path.append(waypoint)
    print(f"Final path x y yaw: {path}")

    return path


if __name__ == "__main__":
    vector_x = np.array([1, 0])
    vector_y = np.array([1, -1])
    print(angle_and_sign_between_vectors(vector_x, vector_y))
