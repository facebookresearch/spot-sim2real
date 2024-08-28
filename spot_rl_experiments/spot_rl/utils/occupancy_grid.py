import pickle as pkl
from math import floor

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

path_to_PCD = "point_cloud_fre.pkl"


def map_x_from_cg_to_grid(x_in_cg, min_x, max_x):
    # return (x_in_cg - min_x)* (max_x - min_x + 1)/(max_x - min_x)
    return x_in_cg - min_x


def map_y_from_cg_to_grid(y_in_cg, min_y, max_y):
    return y_in_cg - min_y


def filter_and_visualize_pcd(points):
    """
    Filter points based on z > 0.5, convert them to an Open3D point cloud, and visualize.

    Parameters:
    points (numpy.ndarray): An array of shape (n, 3) representing the point cloud.
    """
    # Filter points where z > 0.5
    filtered_points = points[points[:, 2] > 0.0]

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)

    # Create a visualizer object
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])


def buil_occupancy_grid(path_to_PCD, scale=10.0):

    with open(path_to_PCD, "rb") as file:
        pcd = pkl.load(file)
        # filter_and_visualize_pcd(pcd.copy())
        min_x, max_x = np.min(pcd[:, 0]), np.max(pcd[:, 0])
        min_y, max_y = np.min(pcd[:, 1]), np.max(pcd[:, 1])
        min_z, max_z = np.min(pcd[:, 2]), np.max(pcd[:, 2])

        min_x, max_x = float(min_x), float(max_x)
        min_y, max_y = float(min_y), float(max_y)
        min_z, max_z = float(min_z), float(max_z)

        len_of_x = max_x - min_x
        len_of_y = max_y - min_y
        # len_of_z = max_z - min_z

        scale = 10.0
        occupancy_grid = np.zeros(
            (int(len_of_x * scale) + 1, int(len_of_y * scale) + 1), dtype=np.uint8
        )
        # breakpoint()
        for point in pcd:
            x, y, z = point
            if z > 0.0:
                try:
                    X = floor(map_x_from_cg_to_grid(x, min_x, max_x) * scale)
                    Y = floor(map_y_from_cg_to_grid(y, min_y, max_y) * scale)
                    occupancy_grid[X, Y] += 1
                except Exception:
                    breakpoint()

        occupancy_grid = np.where(occupancy_grid > 0, 1, 0)
        return occupancy_grid, scale, max_x, max_y, min_x, min_y


if __name__ == "__main__":
    occupancy_grid, scale = buil_occupancy_grid(path_to_PCD, 10.0)
    plt.imshow(occupancy_grid, cmap="gray")
    plt.title("Top-Down View")
    # plt.show()
    plt.savefig("occupancy_grid_fre.png")
