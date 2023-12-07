from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rospy
import sophus as sp
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
from scipy.spatial.transform import Rotation

###################################################################################################
# Taken from:
# https://github.com/facebookresearch/home-robot/blob/1f393f3a244af023684668539d075e7acabde774/src/home_robot/home_robot/utils/geometry/_base.py#L58
# Thanks home-robot team!


def xyt2sophus(xyt: np.ndarray) -> sp.SE3:
    """
    Converts SE2 coordinates (x, y, rz) to an sophus SE3 pose object.
    """
    x = np.array([xyt[0], xyt[1], 0.0])
    r_mat = sp.SO3.exp([0.0, 0.0, xyt[2]]).matrix()
    return sp.SE3(r_mat, x)


def sophus2xyt(se3: sp.SE3) -> np.ndarray:
    """
    Converts an sophus SE3 pose object to SE2 coordinates (x, y, rz).
    """
    x_vec = se3.translation()
    r_vec = se3.so3().log()
    return np.array([x_vec[0], x_vec[1], r_vec[2]])


###################################################################################################


def ros_pose_to_sophus(pose: Pose) -> sp.SE3:
    """
    Convert a ROS pose to a Sophus SE3
    """
    position = pose.position
    orientation = pose.orientation
    translation = np.array([position.x, position.y, position.z])
    quaternion = np.array([orientation.x, orientation.y, orientation.z, orientation.w])
    rotation = Rotation.from_quat(quaternion)
    return sp.SE3(rotation.as_matrix(), np.reshape(translation, (3, 1)))


def sophus_to_ros_pose(se3: sp.SE3) -> Pose:
    """
    Convert a Sophus SE3 to a ROS pose
    """
    pose = Pose()
    translation = se3.translation()
    rotation = se3.rotationMatrix()
    quat = Rotation.from_matrix(rotation).as_quat()
    pose.position.x = translation[0]
    pose.position.y = translation[1]
    pose.position.z = translation[2]
    pose.orientation.x = quat[0]
    pose.orientation.y = quat[1]
    pose.orientation.z = quat[2]
    pose.orientation.w = quat[3]
    return pose


# # Move to transform_utils.py
# def generate_TransformStamped_a_T_b_from_SE3Pose(
#     a_Tform_b: SE3Pose, parent_frame: str, child_frame: str
# ) -> TransformStamped:
#     """Generate the transform from spotWorld to spot frame

#     Returns:
#         Transform: Transform from spotWorld to spot
#     """
#     transform_a_T_b = TransformStamped()

#     transform_a_T_b.header.stamp = rospy.Time.now()
#     transform_a_T_b.header.frame_id = f"/{parent_frame}"
#     transform_a_T_b.child_frame_id = f"/{child_frame}"

#     trans = a_Tform_b.position
#     transform_a_T_b.transform.translation.x = float(trans.x)
#     transform_a_T_b.transform.translation.y = float(trans.y)
#     transform_a_T_b.transform.translation.z = float(trans.z)

#     quat = a_Tform_b.rotation
#     transform_a_T_b.transform.rotation.x = float(quat.x)
#     transform_a_T_b.transform.rotation.y = float(quat.y)
#     transform_a_T_b.transform.rotation.z = float(quat.z)
#     transform_a_T_b.transform.rotation.w = float(quat.w)
#     return transform_a_T_b


# Move to transform_utils.py
def generate_TransformStamped_a_T_b_from_spSE3(
    a_Tform_b: sp.SE3, parent_frame: str, child_frame: str
) -> TransformStamped:
    """Generate the transform from spotWorld to spot frame

    Returns:
        Transform: Transform from spotWorld to spot
    """
    transform_a_T_b = TransformStamped()

    transform_a_T_b.header.stamp = rospy.Time.now()
    transform_a_T_b.header.frame_id = f"/{parent_frame}"
    transform_a_T_b.child_frame_id = f"/{child_frame}"

    trans = a_Tform_b.translation()
    print(trans.shape)
    transform_a_T_b.transform.translation.x = float(trans[0])
    transform_a_T_b.transform.translation.y = float(trans[1])
    transform_a_T_b.transform.translation.z = float(trans[2])

    quat = Rotation.from_matrix(a_Tform_b.rotationMatrix()).as_quat()
    transform_a_T_b.transform.rotation.x = float(quat[0])
    transform_a_T_b.transform.rotation.y = float(quat[1])
    transform_a_T_b.transform.rotation.z = float(quat[2])
    transform_a_T_b.transform.rotation.w = float(quat[3])
    return transform_a_T_b


# Move to transform_utils.py
def generate_spSE3_a_T_b_from_TransformStamped(a_Tform_b: TransformStamped) -> sp.SE3:
    """Generate the transform from spotWorld to spot frame

    Returns:
        Transform: Transform from spotWorld to spot
    """

    trans = np.array([0.0, 0.0, 0.0])
    trans[0] = a_Tform_b.transform.translation.x  # Test this
    trans[1] = a_Tform_b.transform.translation.y  # Test this
    trans[2] = a_Tform_b.transform.translation.z  # Test this

    quat = np.array([0.0, 0.0, 0.0, 1.0])
    quat[0] = a_Tform_b.transform.rotation.x  # Test this
    quat[1] = a_Tform_b.transform.rotation.y  # Test this
    quat[2] = a_Tform_b.transform.rotation.z  # Test this
    quat[3] = a_Tform_b.transform.rotation.w  # Test this
    return sp.SE3(
        Rotation.from_quat(quat).as_matrix(), trans
    )  # Rotation.from_quat() takes a quaternion in the form of [x, y, z, w]


# Move to transform_utils.py
def generate_spSE3_a_T_b_from_PoseStamped(a_Tform_b: PoseStamped) -> sp.SE3:
    """Generate the transform from spotWorld to spot frame

    Returns:
        Transform: Transform from spotWorld to spot
    """

    trans = np.array([0.0, 0.0, 0.0])
    trans[0] = a_Tform_b.pose.position.x  # Test this
    trans[1] = a_Tform_b.pose.position.y  # Test this
    trans[2] = a_Tform_b.pose.position.z  # Test this

    quat = np.array([0.0, 0.0, 0.0, 1.0])
    quat[0] = a_Tform_b.pose.orientation.x  # Test this
    quat[1] = a_Tform_b.pose.orientation.y  # Test this
    quat[2] = a_Tform_b.pose.orientation.z  # Test this
    quat[3] = a_Tform_b.pose.orientation.w  # Test this
    return sp.SE3(
        Rotation.from_quat(quat).as_matrix(), trans
    )  # Rotation.from_quat() takes a quaternion in the form of [x, y, z, w]


def matrix3x4_to_sophus(np_matrix: np.ndarray) -> sp.SE3:
    """
    Convert a 3x4 matrix to a Sophus SE3
    """
    rotation = np_matrix[:, :3]
    translation = np_matrix[:, 3]
    return sp.SE3(rotation, translation)


def compute_avg_spSE3_from_nplist(
    a_T_b_position_list: List[np.ndarray], a_T_b_quaternion_list: List[np.ndarray]
) -> Optional[sp.SE3]:
    """
    Computes the average transformation of aria world frame to marker frame
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

    avg_a_T_b = sp.SE3(
        Rotation.from_quat(avg_a_T_b_quaternion).as_matrix(), avg_a_T_b_position
    )

    return avg_a_T_b
