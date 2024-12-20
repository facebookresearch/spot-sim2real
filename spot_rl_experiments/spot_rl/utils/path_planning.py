import math
import os.path as osp
import pickle
from math import floor

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import yaml
from scipy.ndimage import binary_dilation
from spot_rl.utils.a_star import astar
from spot_rl.utils.construct_configs import load_config
from spot_rl.utils.occupancy_grid import (
    buil_occupancy_grid,
    map_x_from_cg_to_grid,
    map_y_from_cg_to_grid,
    pkl,
)

PATH_TO_CONFIG_FILE = osp.join(
    osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))),
    "configs",
    "cg_config.yaml",
)
assert osp.exists(PATH_TO_CONFIG_FILE), "cg_config.yaml wasn't found"
cg_config = load_config(PATH_TO_CONFIG_FILE)

ROOT_PATH = cg_config["CG_ROOT_PATH"]

FILL_UP_LOCATION = cg_config["FILL_UP_GRID_LOCATION"]

# Define the structure element (the neighborhood for dilation)
DILATION_MAT = np.ones((cg_config["DILATION_SIZE"], cg_config["DILATION_SIZE"]))

CACHE_PATH = osp.join(ROOT_PATH, "occupancy_grid_cache.pkl")

PCD_PATH = osp.join(ROOT_PATH, "point_cloud.pkl")

CG_PCD_PATH = osp.join(ROOT_PATH, "rgb_cloud", "pointcloud.pcd")

OCCUPANCY_SCALE = cg_config["OCCUPANCY_SCALE"]

DISTANCE_THRESHOLD_TO_ADD_POINT = cg_config["DISTANCE_THRESHOLD_TO_ADD_POINT"]


def pick_points(pcd):
    print("")
    print("1) Please pick at least three correspondences using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


def convert_cg_pcd_to_pkl(
    input_cg_pcd_path: str, output_pkl_path: str, visualize=False
):

    assert osp.exists(input_cg_pcd_path), f"{input_cg_pcd_path} no such file found"

    pcd = o3d.io.read_point_cloud(input_cg_pcd_path)
    # print(f"Loading point cloud from: {input_cg_pcd_path}, {pcd}")

    assert not pcd.is_empty(), "Error: The point cloud is empty or the file is empty."

    # Downsample the point cloud
    voxel_size = 0.05
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    print(f"Downsampling the point cloud... {downsampled_pcd}")

    # Visualize if the flag is set
    if visualize:
        print("Visualizing the downsampled point cloud...")
        _ = pick_points(pcd)

    # Save the downsampled points to a pickle file
    points = np.array(downsampled_pcd.points)
    # Convert Open3D vector of points to a list of lists

    # Dump the points to a pickle file
    with open(output_pkl_path, "wb") as f:
        pickle.dump(points, f)

    print(f"Saving the downsampled points to {output_pkl_path}...")


def get_xyzxyz(centroid, extents):
    return np.array(centroid) - (np.array(extents) / 2.0), np.array(centroid) + (
        np.array(extents) / 2.0
    )


def midpoint(x1, x2):
    return (x1 + x2) / 2.0


def angle_between_vectors(v1, v2):
    # Ensure the vectors are numpy arrays
    v1 = np.array(v1)
    v2 = np.array(v2)

    # Compute the dot product
    dot_product = np.dot(v1, v2)

    # Compute the magnitudes of the vectors
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    # Compute the cosine of the angle
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)

    # Clip the cosine value to the range [-1, 1] to avoid numerical issues
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Compute the angle in radians
    angle_radians = np.arccos(cos_angle)

    # Convert the angle to degrees (optional)
    angle_degrees = np.degrees(angle_radians)

    return angle_radians, angle_degrees


def fill_up_occupancy_grid(occupancy_grid):
    """This function is used to fill up the occupancy grid with"""
    grid = occupancy_grid["occupancy_grid"]
    for fill_up_location in FILL_UP_LOCATION:
        grid[
            fill_up_location[0] : fill_up_location[1],
            fill_up_location[2] : fill_up_location[3],
        ] = 1
    occupancy_grid["occupancy_grid"] = grid
    return occupancy_grid


def get_occupancy_grid():
    if osp.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as file:
            data = pkl.load(file)
            filled_data = fill_up_occupancy_grid(data)
            return filled_data

    if not osp.exists(PCD_PATH):
        convert_cg_pcd_to_pkl(CG_PCD_PATH, PCD_PATH)

    assert osp.exists(PCD_PATH), f"Couldn't find {PCD_PATH}"
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
    occupancy_grid_cache = fill_up_occupancy_grid(occupancy_grid_cache)
    print("couldnot find occupancy cache, building from pcd & saving a cache")
    with open(CACHE_PATH, "wb") as file:
        pkl.dump(occupancy_grid_cache, file)
    return occupancy_grid_cache


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

    assert len(path_converted) >= 2

    filter_path = []
    for ii in range(len(path_converted) - 2):
        cur_pt = np.array(path_converted[ii])
        next_pt = np.array(path_converted[ii + 1])
        next_next_pt = np.array(path_converted[ii + 2])

        next_yaw = compute_rotation_angle(cur_pt, next_pt)
        next_next_yaw = compute_rotation_angle(next_pt, next_next_pt)

        # We add waypoint only if the direction of the robot changes
        if next_yaw != next_next_yaw:
            # We check the distance
            if filter_path != []:
                if (
                    np.linalg.norm(next_next_pt - np.array(filter_path[-1][0:2]))
                    >= DISTANCE_THRESHOLD_TO_ADD_POINT
                ):
                    filter_path.append(next_next_pt.tolist() + [next_next_yaw])
            else:
                filter_path.append(next_next_pt.tolist() + [next_next_yaw])

    return filter_path


def convert_real_waypoint_to_occupancy_pixel(
    x,
    y,
    occupancy_min_x,
    occupancy_max_x,
    occupancy_min_y,
    occupancy_max_y,
    occupancy_scale,
):
    X = floor(
        map_x_from_cg_to_grid(x, occupancy_min_x, occupancy_max_x) * occupancy_scale
    )
    Y = floor(
        map_y_from_cg_to_grid(y, occupancy_min_y, occupancy_max_y) * occupancy_scale
    )
    return X, Y


def path_planning_using_a_star(
    xy_position_robot_in_cg,
    goal_xy,
    save_fig_name=None,
    visualize=False,
    other_view_poses=None,
    all_faces=None,
    best_view_pose=None,
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
        dilated_occupancy_grid = binary_dilation(
            occupancy_grid, structure=DILATION_MAT
        ).astype(int)

    start_in_grid = convert_real_waypoint_to_occupancy_pixel(
        *xy_position_robot_in_cg,
        occupancy_min_x,
        occupancy_max_x,
        occupancy_min_y,
        occupancy_max_y,
        occupancy_scale,
    )
    start_in_grid = np.array(start_in_grid)

    occupancy_grid_visualization = binary_to_image(occupancy_grid.copy())
    bestpath = None
    goal_in_grid = convert_real_waypoint_to_occupancy_pixel(
        *goal_xy,
        occupancy_min_x,
        occupancy_max_x,
        occupancy_min_y,
        occupancy_max_y,
        occupancy_scale,
    )
    goal_in_grid = np.array(goal_in_grid)
    print(f"start pos {start_in_grid}, goal in grid {goal_in_grid}")
    path, recursion_depth = astar(
        dilated_occupancy_grid,
        (start_in_grid[0], start_in_grid[1]),
        (goal_in_grid[0], goal_in_grid[1]),
    )

    occupancy_grid_visualization = cv2.circle(
        occupancy_grid_visualization,
        (start_in_grid[1], start_in_grid[0]),
        1,
        (255, 0, 0),
        1,
    )

    if all_faces is not None:
        for face in all_faces:
            u, v = face[:2]
            u, v = convert_real_waypoint_to_occupancy_pixel(
                u,
                v,
                occupancy_min_x,
                occupancy_max_x,
                occupancy_min_y,
                occupancy_max_y,
                occupancy_scale,
            )
            occupancy_grid_visualization = cv2.circle(
                occupancy_grid_visualization,
                (v, u),
                1,
                (0, 0, 255),
                1,
            )
    occupancy_grid_visualization = cv2.circle(
        occupancy_grid_visualization,
        (goal_in_grid[1], goal_in_grid[0]),
        1,
        (238, 211, 14),
        1,
    )
    if len(path) > 2:
        filter_path = convert_path_to_real_waypoints(
            path, occupancy_min_x, occupancy_min_y
        )
        for iter, waypoint in enumerate(filter_path):
            x = floor(
                map_x_from_cg_to_grid(waypoint[0], occupancy_min_x, occupancy_max_x)
                * occupancy_scale
            )
            y = floor(
                map_y_from_cg_to_grid(waypoint[1], occupancy_min_y, occupancy_max_y)
                * occupancy_scale
            )
            occupancy_grid_visualization = cv2.circle(
                occupancy_grid_visualization,
                (y, x),
                0,
                (0, 255, 0),
                -1,
            )
            bestpath = path
    if other_view_poses:
        for view_pose in other_view_poses:
            x = floor(
                map_x_from_cg_to_grid(view_pose[0], occupancy_min_x, occupancy_max_x)
                * occupancy_scale
            )
            y = floor(
                map_y_from_cg_to_grid(view_pose[1], occupancy_min_y, occupancy_max_y)
                * occupancy_scale
            )
            occupancy_grid_visualization = cv2.circle(
                occupancy_grid_visualization,
                (y, x),
                0,
                (0, 255, 255),
                -1,
            )
    if best_view_pose is not None:
        u, v = best_view_pose
        u, v = convert_real_waypoint_to_occupancy_pixel(
            u,
            v,
            occupancy_min_x,
            occupancy_max_x,
            occupancy_min_y,
            occupancy_max_y,
            occupancy_scale,
        )
        occupancy_grid_visualization = cv2.circle(
            occupancy_grid_visualization,
            (v, u),
            1,
            (127, 94, 24),
            -1,
        )
    if visualize:
        plt.imshow(occupancy_grid_visualization, cmap="gray")
        plt.title(osp.basename(save_fig_name).split(".")[0])
        plt.savefig(save_fig_name)
        plt.show()
    # is the final destination not excatly reachable then find the closest rechable point
    is_path_psuedo_rechable = recursion_depth > 0
    if bestpath is not None:
        best_converted_path = convert_path_to_real_waypoints(
            bestpath, occupancy_min_x, occupancy_min_y
        )
        distance_to_target = np.linalg.norm(
            best_converted_path[-1][:2] - np.array(goal_xy)
        )
        return best_converted_path, is_path_psuedo_rechable, distance_to_target
    print(f"No solution for start: {xy_position_robot_in_cg}, goal {goal_xy}")
    return [], False, float("inf")


if __name__ == "__main__":
    bbox_extents = np.array([0.9, 0.8, 0.2])
    bbox_centers = np.array([9.6, 2.3, 0.3])
    robot_pose = [1.0, 0.0]
    goal_xy = bbox_centers[:2] + bbox_extents[:2]
    print(f"Current robot pose {robot_pose}")
    print(f"Goal {goal_xy}")
    print(path_planning_using_a_star(np.array(robot_pose), goal_xy))
