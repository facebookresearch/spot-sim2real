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

def trying_new_segment_plane(pcd):
    
    #assert (pcd.has_normals())
    # pcd = NumpyToPCD(points)
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
    # using all defaults
    oboxes = pcd.detect_planar_patches(
        normal_variance_threshold_deg=70,
        coplanarity_deg=80,
        outlier_ratio=0.75,
        min_plane_edge_length=0,
        min_num_points=0,
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

    print("Detected {} patches".format(len(oboxes)))
    
    planes = [ pcd.select_by_index(obox.get_point_indices_within_bounding_box(pcd.points)) for obox in oboxes]
    #geometries = []
    # for obox in oboxes:
    #     plane_pcd = pcd.select_by_index(obox.get_point_indices_within_bounding_box(pcd.points))
    #     planes.append(plane_pcd)
        # mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obox, scale=[1, 1, 0.0001])
        # mesh.paint_uniform_color(obox.color)
        # geometries.append(plane)
        # geometries.append(obox)
    #geometries.append(pcd)
    o3d.visualization.draw_geometries(planes)
    # o3d.visualization.draw_geometries(geometries,
    #                                 zoom=0.62,
    #                                 front=[0.4361, -0.2632, -0.8605],
    #                                 lookat=[2.4947, 1.7728, 1.5541],
    #                                 up=[-0.1726, -0.9630, 0.2071])
    return planes

def plane_detect(pcd):
    return trying_new_segment_plane(pcd)
    points = PCDToNumpy(pcd)
    points = RemoveNoiseStatistical(points, nb_neighbors=50, std_ratio=0.5)

    # DrawPointCloud(points, color=(0.4, 0.4, 0.4))
    t0 = time.time()
    
    results = DetectMultiPlanes(
        points, min_ratio=0.05, threshold=0.005, iterations=2000
    )
    
    print("Time:", time.time() - t0)
    planes = []
    colors = []

    highest_pts, high_i = -np.inf, 0
    # lowest_dist_to_camera, low_i = np.inf, 0
    print(f"{len(results)} plane are detected")
    for i, (_, plane) in enumerate(results):

        r = 1
        g = random.random()
        b = random.random()

        color = np.zeros((plane.shape[0], plane.shape[1]))
        color[:, 0] = r
        color[:, 1] = g
        color[:, 2] = b

        planes.append(plane)
        colors.append(color)

        # check depth at centroid
        plane_pcd = NumpyToPCD(plane)
        dist = -plane_pcd.get_center()[-1] + plane.shape[0]
        if dist >= highest_pts:
            highest_pts = dist
            high_i = i

    planes_selected = [planes[high_i]]
    colors_selected = [colors[high_i]]
    planes_selected = np.concatenate(planes_selected, axis=0)
    colors_selected = np.concatenate(colors_selected, axis=0)
    return DrawResult(planes_selected, colors_selected), planes, colors
