import numpy as np
from spot_rl.utils.pixel_to_3d_conversion_utils import (
    get_3d_point,
    sample_patch_around_point,
)

# TODO: under development


def grasp_control_parmeters(object_name: str):
    """
    Given the object_name or rgb_image of object; lookup dictionary to come up with force control parameters
    These parameters consist of list of (claw_fraction_open_angle, max_torque) which controls the slow closing of the gripper & force applied to hold the object.
    """
    if object_name == "cup":
        return [(0.7, 0.0), (0.6, 0.5), (0.5, 0.7), (0.4, 0.5), (0.3, 0.6), (0.2, 1.0)]


def affordance_prediction(
    object_name: str,
    rgb_image: np.ndarray,
    depth_raw: np.ndarray,
    mask: np.ndarray,
    camera_intrinsics,
    center_pixel: np.ndarray,
) -> np.ndarray:
    """
    Accepts
    object_name:str
    rgb_image: np.array HXWXC, 0-255
    depth_raw: np.array HXW, 0.-2000.
    mask: HXW, bool mask
    camera_intrinsics:spot camera intrinsic object
    center_pixel: np.array of length 2
    Returns: Suitable point on object to grasp
    """

    mask = np.where(mask > 0, 1, 0).astype(depth_raw.dtype)
    depth_image_masked = depth_raw * mask[...].astype(depth_raw.dtype)

    non_zero_indices = np.nonzero(depth_image_masked)
    # Calculate the bounding box coordinates
    y_min, y_max = non_zero_indices[0].min(), non_zero_indices[0].max()
    x_min, x_max = non_zero_indices[1].min(), non_zero_indices[1].max()
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    Z = float(sample_patch_around_point(int(cx), int(cy), depth_raw) * 1e-3)
    point_in_gripper = get_3d_point(camera_intrinsics, center_pixel, Z)

    return point_in_gripper
