import math
import os.path as osp
from math import floor

import cv2
import matplotlib.pyplot as plt
import numpy as np
from spot_rl.utils.a_star import astar
from spot_rl.utils.occupancy_grid import (
    buil_occupancy_grid,
    map_x_from_cg_to_grid,
    map_y_from_cg_to_grid,
    pkl,
)
from spot_rl.utils.waypoint_adjustment import (
    angle_between_vectors,
    get_xyzxyz,
    intersect_ray_with_aabb,
    midpoint,
)

CACHE_PATH = osp.join(osp.dirname(osp.abspath(__file__)), "occupancy_grid_cache.pkl")

PCD_PATH = osp.join(osp.dirname(osp.abspath(__file__)), "point_cloud_fre.pkl")

OCCUPANCY_SCALE = 10.0

# def path_planning_based_on_raymarching(occupancy_grid, occupancy_max_x, occupancy_max_y, occupancy_scale, xy_position_robot_in_cg, bbox_centers, bbox_extents):

#     boxMin, boxMax = get_xyzxyz(bbox_centers, bbox_extents)
#     robot_position = np.array([*xy_position_robot_in_cg, bbox_centers[-1]+0.1])
#     rayDir = (bbox_centers - robot_position) / np.linalg.norm(bbox_centers - robot_position)
#     intersects, pt1, _, t_min, _ = intersect_ray_with_aabb(robot_position, rayDir, boxMin, boxMax)
#     assert intersects, "couldn't find the intersection"
#     if intersects:
#         point_of_intersection = pt1[:2]
#         newRayDir = (point_of_intersection - xy_position_robot_in_cg)/np.linalg.norm(point_of_intersection - xy_position_robot_in_cg)
#         rayorigin = xy_position_robot_in_cg
#         t = 0.0
#         accum = 0.0
#         num_steps = 0
#         while t < t_min:
#             current_ray_cast_position = rayorigin + t*newRayDir
#             X = floor(map_x_from_cg_to_grid(current_ray_cast_position[0], occupancy_max_x)*occupancy_scale)
#             Y = floor(map_y_from_cg_to_grid(current_ray_cast_position[1], occupancy_max_y)*occupancy_scale)
#             accum += occupancy_grid[X, Y]
#             t += 0.1
#             num_steps += 1
#             #breakpoint()
#         print(accum, num_steps)
#     return accum/num_steps #x,y,yaw


def get_occupancy_grid():
    if False:  # osp.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as file:
            return pkl.load(file)
    elif osp.exists(PCD_PATH):
        (
            occupancy_grid,
            occupancy_scale,
            max_x,
            max_y,
            min_x,
            min_y,
        ) = buil_occupancy_grid(PCD_PATH, OCCUPANCY_SCALE)
        occupancy_grid_cache = {
            "occupancy_grid": occupancy_grid,
            "scale": occupancy_scale,
            "max_x": max_x,
            "max_y": max_y,
            "min_x": min_x,
            "min_y": min_y,
        }
        print("couldnot find occupancy cache, building from pcd & saving a cache")
        with open(CACHE_PATH, "wb") as file:
            pkl.dump(occupancy_grid_cache, file)
        return occupancy_grid_cache
    else:
        raise Exception(f"{PCD_PATH} not found")


def binary_to_image(binary_array):
    """
    Convert a binary numpy array of shape (H, W) with dtype np.uint8
    into an image of shape (H, W, C) with dtype np.uint8 and range 0-255.

    Parameters:
    binary_array (numpy.ndarray): Binary array of shape (H, W) and dtype np.uint8.

    Returns:
    numpy.ndarray: Image array of shape (H, W, 3) with dtype np.uint8 and values in range 0-255.
    """
    # Ensure the input is binary (0 or 1)
    binary_array = np.clip(binary_array, 0, 1)

    # Scale to range 0-255
    scaled_array = binary_array * 255

    # Convert to shape (H, W, 3) for an RGB image
    image_array = np.stack((scaled_array,) * 3, axis=-1)

    return image_array.astype(np.uint8)


def compute_rotation_angle(point1, point2):
    # Calculate the vector between the two points
    vector_x = point2[0] - point1[0]
    vector_y = point2[1] - point1[1]

    # Calculate the angle using atan2, which handles all quadrants
    angle_radians = math.atan2(vector_y, vector_x)

    return angle_radians


def convert_path_to_real_waypoints(path, min_x, min_y):
    path_converted = []
    for waypoint in path:
        new_waypoint = [waypoint[0] / 10.0, waypoint[1] / 10.0]
        new_waypoint[0] = np.round(min_x + new_waypoint[0], 1)
        new_waypoint[1] = np.round(min_y + new_waypoint[1], 1)
        # print(new_waypoint)
        path_converted.append(new_waypoint)

    assert len(path_converted) > 2

    filter_path = []
    for ii in range(len(path_converted) - 2):
        cur_pt = np.array(path_converted[ii])
        next_pt = np.array(path_converted[ii + 1])
        next_next_pt = np.array(path_converted[ii + 2])

        next_yaw = compute_rotation_angle(cur_pt, next_pt)
        next_next_yaw = compute_rotation_angle(next_pt, next_next_pt)

        if next_yaw != next_next_yaw:
            filter_path.append(next_next_pt.tolist() + [next_next_yaw])

    return filter_path


def path_planning_using_a_star(
    xy_position_robot_in_cg,
    bbox_centers,
    bbox_extents,
    occupancy_grid=None,
    occupancy_max_x=None,
    occupancy_max_y=None,
    occupancy_min_x=None,
    occupancy_min_y=None,
    occupancy_scale=None,
):
    if occupancy_grid is None:
        occupancy_cache = get_occupancy_grid()
        (
            occupancy_grid,
            occupancy_scale,
            occupancy_max_x,
            occupancy_max_y,
            occupancy_min_x,
            occupancy_min_y,
        ) = list(occupancy_cache.values())

    boxMin, boxMax = get_xyzxyz(bbox_centers, bbox_extents)
    STATIC_OFFSET = 0.7  # adjustment for base 0.5 + extra offset
    (x1, y1), (x2, y2) = boxMin[:2], boxMax[:2]
    face_1 = np.array([midpoint(x1, x2), y1 - STATIC_OFFSET])
    face_2 = np.array([midpoint(x1, x2), y2 + STATIC_OFFSET])
    face_3 = np.array([x1 - STATIC_OFFSET, midpoint(y1, y2)])
    face_4 = np.array([x2 + STATIC_OFFSET, midpoint(y1, y2)])
    faces = [face_1, face_2, face_3, face_4]
    X = floor(
        map_x_from_cg_to_grid(
            xy_position_robot_in_cg[0], occupancy_min_x, occupancy_max_x
        )
        * occupancy_scale
    )
    Y = floor(
        map_y_from_cg_to_grid(
            xy_position_robot_in_cg[1], occupancy_min_y, occupancy_max_y
        )
        * occupancy_scale
    )
    start_in_grid = np.array([X, Y])
    min_steps = float("inf")
    min_idx = -1
    occupancy_grid_visualization = binary_to_image(occupancy_grid.copy())
    bestpath = None
    # occupancy_grid_visualization = occupancy_grid_visualization.transpose((1, 2, 0))
    for face_idx, face in enumerate(faces):
        X = floor(
            map_x_from_cg_to_grid(face[0], occupancy_min_x, occupancy_max_x)
            * occupancy_scale
        )
        Y = floor(
            map_y_from_cg_to_grid(face[1], occupancy_min_y, occupancy_max_y)
            * occupancy_scale
        )
        goal_in_grid = np.array([X, Y])
        print(f"start pos {start_in_grid}, goal in grid {goal_in_grid}")
        path = astar(
            occupancy_grid,
            (start_in_grid[0], start_in_grid[1]),
            (goal_in_grid[0], goal_in_grid[1]),
        )
        if len(path) > 0:
            print("Paths", path, occupancy_grid_visualization.shape)

            occupancy_grid_visualization = cv2.circle(
                occupancy_grid_visualization,
                (start_in_grid[1], start_in_grid[0]),
                1,
                (255, 0, 0),
                1,
            )
            occupancy_grid_visualization = cv2.circle(
                occupancy_grid_visualization,
                (goal_in_grid[1], goal_in_grid[0]),
                1,
                (0, 0, 255),
                1,
            )
            for iter, waypoint in enumerate(path):
                if iter % 2 == 0:
                    occupancy_grid_visualization = cv2.circle(
                        occupancy_grid_visualization,
                        (waypoint[1], waypoint[0]),
                        1,
                        (0, 255, 0),
                        1,
                    )
            min_steps = min(min_steps, len(path))
            bestpath = path
            min_idx = face_idx
        else:
            print(f"No solution for Edge_{face_idx+1}")
    assert min_idx > -1, "couldnt find solution for given path problem"
    plt.imshow(occupancy_grid_visualization, cmap="gray")
    plt.title("Path Planning")
    plt.show()
    # plt.savefig("occupancy_grid_fre.png")
    # breakpoint()
    if min_idx > -1:
        print(f"Min steps required to reach {min_steps}, {min_idx}")
        face = faces[min_idx]
        center = np.array([midpoint(x1, x2), midpoint(y1, y2)])
        dirVector = (center - face) / np.linalg.norm(center - face)
        yaw_calc = angle_between_vectors(np.array([1, 0]), dirVector)[1]
        final_point = face.tolist() + [np.deg2rad(yaw_calc)]
        return convert_path_to_real_waypoints(
            bestpath, occupancy_min_x, occupancy_min_y
        ) + [final_point]
    return []


if __name__ == "__main__":
    bbox_extents = np.array([0.9, 0.7, 0.5])
    bbox_centers = np.array([4.0, -1.9, 0.5])
    robot_pose = [0.0, 0.0]
    print(f"Current robot pose {robot_pose}")
    print(path_planning_using_a_star(np.array(robot_pose), bbox_centers, bbox_extents))
