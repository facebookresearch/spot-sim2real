# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, List, Optional, Tuple

import numpy as np
import sophus as sp
from scipy.spatial.transform import Rotation

FILTER_DIST = 2.4  # in meters (distance for valid detection)
FIXED_LIST_LENGTH = 10  # Fixed length of the a_T_b_{position,orientation} lists used for average computation


def compute_avg_sophus_SE3_from_nplist(
    a_T_b_position_list: List[np.ndarray], a_T_b_quaternion_list: List[np.ndarray]
) -> Optional[sp.SE3]:
    """
    Computes the average transformation of frame b expressed in frame a
    Args:
        a_T_b_position_list (List[np.ndarray]): List of positions of frame b in frame a
        a_T_b_quaternion_list (List[np.ndarray]): List of quaternions of frame b in frame a

    Returns:
        avg_sp_se3 (sp.SE3): Average of all the transformations as a Sophus SE3 object
    """
    assert len(a_T_b_position_list) == len(
        a_T_b_quaternion_list
    ), "Position and Quaternion lists must be equal length"

    if len(a_T_b_position_list) == 0:
        return None

    a_T_b_position_np = np.array(a_T_b_position_list)
    avg_a_T_b_position = np.mean(a_T_b_position_np, axis=0)

    a_T_b_quaternion_np = np.array(a_T_b_quaternion_list)
    avg_a_T_b_quaternion = np.mean(a_T_b_quaternion_np, axis=0)

    avg_sp_se3 = sp.SE3(
        Rotation.from_quat(avg_a_T_b_quaternion).as_matrix(), avg_a_T_b_position
    )

    return avg_sp_se3


def get_running_avg_a_T_b(
    current_avg_a_T_b: sp.SE3,
    a_T_b_position_list: List[np.ndarray],
    a_T_b_quaternion_list: List[np.ndarray],
    a_T_intermediate: sp.SE3,
    intermediate_T_b: sp.SE3,
    filter_dist: float = FILTER_DIST,
    fixed_list_length: int = FIXED_LIST_LENGTH,
) -> Tuple[List[np.ndarray], List[np.ndarray], Optional[sp.SE3]]:
    """
     Computes and returns average transformation of frame a to frame b via an intermediate frame (a_T_intermediate & intermediate_T_b).

     NOTE: To compute average of SE3 matrix, we find the average of translation and rotation separately.
             The average rotation is obtained by averaging the quaternions.
     NOTE: Since multiple quaternions can represent the same rotation, we ensure that the 'w' component of the
             quaternion is always positive for effective averaging.

     Args:
         current_avg_a_T_b (Optional[sp.SE3]): Current average transformation of frame a to frame b. Used as a cache which gets updated
                                               Could be None if no previous average exists i.e. before any valid data was cached.
         a_T_b_position_list (List[np.ndarray]): List of positions of frame b in frame a. Used as a cache which gets updated
         a_T_b_quaternion_list (List[np.ndarray]): List of quaternions of frame b in frame a. Used as a cache which gets updated
         a_T_intermediate (sp.SE3): Sophus SE3 object representing transformation of frame intermediate in frame a. This is most recent data.
         intermediate_T_b (sp.SE3): Sophus SE3 object representing transformation of frame b in frame intermediate. This is most recent data.
         filter_dist (float, optional): Distance threshold for valid detections. Defaults to FILTER_DIST.
         fixed_list_length (int, optional): Maximum size of the list. Defaults to FIXED_LIST_LENGTH.

    # Returns:
         a_T_b_position_list (List[np.ndarray]): List of positions of frame b in frame a. Will be updated
                                                     if the latest b frame is within the filter distance of intermediate frame.
         a_T_b_quaternion_list (List[np.ndarray]): List of quaternions of frame b in frame world. Will be updated
                                                     if the latest b frame is within the filter distance of intermediate frame.
         avg_a_T_b (Optional[sp.SE3]): Average of all the transformations as a Sophus SE3 object.

    """
    a_T_b = a_T_intermediate * intermediate_T_b
    b_position = a_T_b.translation()
    intermediate_position = a_T_intermediate.translation()
    delta = b_position - intermediate_position
    dist = np.linalg.norm(delta)

    # Consider only those detections where frame b is within a certain distance of the frame a measured in frame world
    if dist < filter_dist:
        # If the number of detections exceeds the fix list length, remove the first element
        if len(a_T_b_position_list) >= fixed_list_length:
            a_T_b_position_list.pop(0)
            a_T_b_quaternion_list.pop(0)

        a_T_b_position_list.append(b_position)

        # Ensure quaternion's w is always positive for effective averaging as multiple quaternions can represent the same rotation
        quat = Rotation.from_matrix(a_T_b.rotationMatrix()).as_quat()
        if quat[3] > 0:
            quat = -1.0 * quat
        a_T_b_quaternion_list.append(quat)

        # Compute the average transformation as new data got appended
        avg_a_T_b = compute_avg_sophus_SE3_from_nplist(
            a_T_b_position_list=a_T_b_position_list,
            a_T_b_quaternion_list=a_T_b_quaternion_list,
        )
    else:
        # If the latest detection is not within the filter distance, then do not update the list
        avg_a_T_b = current_avg_a_T_b

    return a_T_b_position_list, a_T_b_quaternion_list, avg_a_T_b
