from typing import List

import numpy as np


def sample_patch_around_point(
    cx: int, cy: int, depth_raw: np.ndarray, patch_size: int = 5
) -> int:
    """
    Samples a median depth in 5x5 patch around given x, y (pixel location in depth image array) as center in raw depth image
    """
    h, w = depth_raw.shape
    x1, x2 = cx - patch_size // 2, cx + patch_size // 2
    y1, y2 = cy - patch_size // 2, cy + patch_size // 2
    x1, x2 = np.clip([x1, x2], 0, w)
    y1, y2 = np.clip([y1, y2], 0, h)
    deph_patch = depth_raw[y1:y2, x1:x2]
    deph_patch = deph_patch[deph_patch > 0]
    return np.median(deph_patch)


def project_3d_to_pixel_uv(points_3d, cam_intrinsics):
    """
    Back projects given xyz 3d point to pixel location u,v using camera intrinsics
    """
    fx = cam_intrinsics.focal_length.x
    fy = cam_intrinsics.focal_length.y
    cx = cam_intrinsics.principal_point.x
    cy = cam_intrinsics.principal_point.y
    Z = points_3d[:, -1]
    X_Z = points_3d[:, 0] / Z
    Y_Z = points_3d[:, 1] / Z
    u = (fx * X_Z) + cx
    v = (fy * Y_Z) + cy
    return np.stack([u.flatten(), v.flatten()], axis=1).reshape(-1, 2)


def get_3d_point(cam_intrinsics, pixel_uv: List[int], z: float):
    # Get camera intrinsics
    fx = float(cam_intrinsics.focal_length.x)
    fy = float(cam_intrinsics.focal_length.y)
    cx = float(cam_intrinsics.principal_point.x)
    cy = float(cam_intrinsics.principal_point.y)

    # print(fx, fy, cx, cy)
    # Get 3D point
    x = (pixel_uv[0] - cx) * z / fx
    y = (pixel_uv[1] - cy) * z / fy
    return np.array([x, y, z])


def get_3d_points(cam_intrinsics, pixels_uv: np.ndarray, zs: np.ndarray):
    """
    Vectorized version of the above method, pass n, 2D points & get n 3D points
    """
    # pixels_uv = nx2 xs -> :, 1
    # Get camera intrinsics
    fx = cam_intrinsics.focal_length.x
    fy = cam_intrinsics.focal_length.y
    cx = cam_intrinsics.principal_point.x
    cy = cam_intrinsics.principal_point.y
    # Get 3D point
    xs = (pixels_uv[:, 1] - cx) * zs / fx  # n
    ys = (pixels_uv[:, 0] - cy) * zs / fy  # n
    return np.array([xs.flatten(), ys.flatten(), zs]).reshape(-1, 3)


def get_best_uvz_from_detection(
    unscaled_dep_img, detection, depth_scale: float = 0.001
):
    """
    Sample best z depth for the given bounding box
    """
    center_x, center_y = (detection[0] + detection[2]) / 2, (
        detection[1] + detection[3]
    ) / 2
    # select the patch of the depth
    depth_patch_in_bbox = unscaled_dep_img[
        int(detection[1]) : int(detection[3]), int(detection[0]) : int(detection[2])
    ]
    # keep only non zero values
    depth_patch_in_bbox = depth_patch_in_bbox[depth_patch_in_bbox > 0.0].flatten()
    if len(depth_patch_in_bbox) > 0:
        # find mu & sigma
        mu = np.median(depth_patch_in_bbox)
        closest_depth_to_mu = np.argmin(np.absolute(depth_patch_in_bbox - mu))
        return (center_x, center_y), depth_patch_in_bbox[
            closest_depth_to_mu
        ] * depth_scale
    return (center_x, center_y), 0
