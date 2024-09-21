# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Code derived from https://github.com/yuecideng/Multiple_Planes_Detection/tree/master?tab=MIT-1-ov-file
import random
import time

import numpy as np
import open3d as o3d


def ReadPlyPoint(fname):
    """read point from ply

    Args:
        fname (str): path to ply file

    Returns:
        [ndarray]: N x 3 point clouds
    """

    pcd = o3d.io.read_point_cloud(fname)

    return PCDToNumpy(pcd)


def NumpyToPCD(xyz):
    """convert numpy ndarray to open3D point cloud

    Args:
        xyz (ndarray):

    Returns:
        [open3d.geometry.PointCloud]:
    """

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    return pcd


def PCDToNumpy(pcd):
    """convert open3D point cloud to numpy ndarray

    Args:
        pcd (open3d.geometry.PointCloud):

    Returns:
        [ndarray]:
    """

    return np.asarray(pcd.points)


def RemoveNan(points):
    """remove nan value of point clouds

    Args:
        points (ndarray): N x 3 point clouds

    Returns:
        [ndarray]: N x 3 point clouds
    """

    return points[~np.isnan(points[:, 0])]


def RemoveNoiseStatistical(pc, nb_neighbors=20, std_ratio=2.0):
    """remove point clouds noise using statitical noise removal method

    Args:
        pc (ndarray): N x 3 point clouds
        nb_neighbors (int, optional): Defaults to 20.
        std_ratio (float, optional): Defaults to 2.0.

    Returns:
        [ndarray]: N x 3 point clouds
    """

    pcd = NumpyToPCD(pc)
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )

    return PCDToNumpy(cl)


def DownSample(pts, voxel_size=0.003):
    """down sample the point clouds

    Args:
        pts (ndarray): N x 3 input point clouds
        voxel_size (float, optional): voxel size. Defaults to 0.003.

    Returns:
        [ndarray]:
    """

    p = NumpyToPCD(pts).voxel_down_sample(voxel_size=voxel_size)

    return PCDToNumpy(p)


def PlaneRegression(points, threshold=0.01, init_n=3, iter=1000):
    """plane regression using ransac

    Args:
        points (ndarray): N x3 point clouds
        threshold (float, optional): distance threshold. Defaults to 0.003.
        init_n (int, optional): Number of initial points to be considered inliers in each iteration
        iter (int, optional): number of iteration. Defaults to 1000.

    Returns:
        [ndarray, List]: 4 x 1 plane equation weights, List of plane point index
    """

    pcd = NumpyToPCD(points)

    w, index = pcd.segment_plane(threshold, init_n, iter)

    return w, index


def DrawResult(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])
    return pcd


def DetectMultiPlanes(points, min_ratio=0.05, threshold=0.01, iterations=1000):
    """Detect multiple planes from given point clouds

    Args:
        points (np.ndarray):
        min_ratio (float, optional): The minimum left points ratio to end the Detection. Defaults to 0.05.
        threshold (float, optional): RANSAC threshold in (m). Defaults to 0.01.

    Returns:
        [List[tuple(np.ndarray, List)]]: Plane equation and plane point index
    """

    plane_list = []
    N = len(points)
    target = points.copy()
    count = 0

    while count < (1 - min_ratio) * N:
        w, index = PlaneRegression(
            target, threshold=threshold, init_n=3, iter=iterations
        )

        count += len(index)
        plane_list.append((w, target[index]))
        target = np.delete(target, index, axis=0)

    return plane_list


def plane_detect(pcd, visualize=False):
    points = PCDToNumpy(pcd)
    points = RemoveNoiseStatistical(points, nb_neighbors=50, std_ratio=0.5)

    t0 = time.time()

    results = DetectMultiPlanes(
        points, min_ratio=0.05, threshold=0.005, iterations=2000
    )

    print("Time:", time.time() - t0)
    planes = []

    print(f"{len(results)} plane are detected")
    for i, (_, plane) in enumerate(results):

        r = 1
        g = random.random()
        b = random.random()

        color = np.zeros((plane.shape[0], plane.shape[1]))
        color[:, 0] = r
        color[:, 1] = g
        color[:, 2] = b

        plane = NumpyToPCD(plane)
        plane.colors = o3d.utility.Vector3dVector(color)
        planes.append(plane)

    # if visualize:
    # o3d.visualization.draw_geometries(planes)

    return planes
