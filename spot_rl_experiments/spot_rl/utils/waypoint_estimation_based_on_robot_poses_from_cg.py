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


def determin_nearest_edge(robot_xy, bbox_centers, boxMin, boxMax):
    if len(bbox_centers) == 3:
        bbox_centers = bbox_centers[:2]
    raydir = (bbox_centers - robot_xy) / np.linalg.norm(bbox_centers - robot_xy)
    STATIC_OFFSET = 0.7
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
        angle_between_vectors(raydir[:2], face[0])[1] for face in faces
    ]
    min_idx = np.argmin(angles_betwn_approach_vector_and_faces)
    _ = angles_betwn_approach_vector_and_faces[min_idx]
    nearestfacevector, nearestface = faces[min_idx]
    yaw_calc = angle_between_vectors(np.array([1, 0]), nearestfacevector)[1]
    return nearestface, nearestfacevector, yaw_calc


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
    robot_view_poses, bbox_centers: np.ndarray, bbox_extents: np.ndarray
):
    sorted_robot_view_poses = sort_robot_view_poses_from_cg(
        robot_view_poses, bbox_centers, 0.6, 0.1, 0.3
    )
    best_robot_view_pos = sorted_robot_view_poses[0]
    print(f"Best robot view pose {best_robot_view_pos}")
    boxMin, boxMax = get_xyzxyz(bbox_centers, bbox_extents)
    nearestedge, facevector, yaw_calc = determin_nearest_edge(
        best_robot_view_pos[0][:2], bbox_centers, boxMin, boxMax
    )
    waypoint = nearestedge.tolist()
    waypoint.append(yaw_calc)
    return waypoint, best_robot_view_pos


if __name__ == "__main__":
    # receptacle details from CG
    bbox_extents = np.array([1.3, 1.0, 0.8])
    bbox_centers = np.array([8.2, 6.0, 0.1])

    boxMin, boxMax = get_xyzxyz(bbox_centers, bbox_extents)
    print("boxMin", boxMin, "boxMax", boxMax)

    # PASTE your waypoint in robot_xy; only x,y, keep z as it is
    # obtain xy,yaw from Concept graph based on detection confs & area. Rank top 5 etc.
    robot_xy = np.array(
        [4.651140979172928, -3.7375389203182516, bbox_centers[-1] + 0.1]
    )
    yaw_cg = 88.59926264693141

    robot_view_pose_data = pkl.load(
        open("robot_view_poses_for_bedroom_dresser.pkl", "rb")
    )
    robot_view_poses = []
    for robot_view_pose in robot_view_pose_data:
        robot_view_poses.append(
            [
                tuple(robot_view_pose["robot_xy_yaw"].tolist()),
                float(robot_view_pose["conf"]),
                int(robot_view_pose["pixel_area"]),
            ]
        )

    # waypoint_dock = [(2.8363019401919116, 0.18846130298868974, -39.047862044229156), 0.94, 50*50]
    # waypoint_stairs = [( 6.7563028035139485, -1.1514989785168246, -131.4607976353374), 0.89, 100*100]
    # waypoint_kitchen = [( 4.714151978378424, -3.413209498770333, 102.83945805400889), 0.93, 100*150]
    # Target is sink
    # near dock, 0.94, detection (50, 50)
    # kitchen counter 0.93, detection (100, 150)
    # from stairs 0.86, detection (100, 100)
    waypoint, best_robot_view_pos = get_waypoint_from_robot_view_poses(
        robot_view_poses, bbox_centers, bbox_extents
    )
    path = path_planning_using_a_star(
        [0, 0],  # best_robot_view_pos[0][:2]
        waypoint[:2],
        other_view_poses=[view_pose[0][:2] for view_pose in robot_view_poses],  # type: ignore
    )
    waypoint[-1] = np.deg2rad(waypoint[-1])
    path.append(waypoint)
    print(f"Final path x y yaw: {path}")
    with open("path.pkl", "wb") as file:
        pkl.dump(path, file)
