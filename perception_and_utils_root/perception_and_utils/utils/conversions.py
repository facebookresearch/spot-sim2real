# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import rospy
import sophus as sp
from bosdyn.api.geometry_pb2 import SE3Pose
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
from scipy.spatial.transform import Rotation

"""
Use the following naming convention for transforms:

def <library>_<dtype1>_to_<library>_<dtype2>(name: <dtype1>) -> <dtype2>:

Example:
def ros_PoseStamped_to_sophus_SE3(pose: PoseStamped) -> sp.SE3:
"""

###################################################################################################
# Taken from:
# https://github.com/facebookresearch/home-robot/blob/1f393f3a244af023684668539d075e7acabde774/src/home_robot/home_robot/utils/geometry/_base.py#L58
# Thanks home-robot team!


def xyt_to_sophus_SE3(xyt: np.ndarray) -> sp.SE3:
    """
    Converts SE2 coordinates (x, y, theta) to an sophus SE3 pose object.
    """
    x = np.array([xyt[0], xyt[1], 0.0])
    r_mat = sp.SO3.exp([0.0, 0.0, xyt[2]]).matrix()
    return sp.SE3(r_mat, x)


def sophus_SE3_to_xyt(se3: sp.SE3) -> np.ndarray:
    """
    Converts an sophus SE3 pose object to SE2 coordinates (x, y, theta).
    """
    x_vec = se3.translation()
    r_vec = se3.so3().log()
    return np.array([x_vec[0], x_vec[1], r_vec[2]])


###################################################################################################


def ros_Pose_to_sophus_SE3(pose: Pose) -> sp.SE3:
    """
    Convert a ROS pose to a Sophus SE3

    Args:
        pose (Pose): ROS Pose object

    Returns:
        sp.SE3: Sophus SE3 object
    """
    position = pose.position
    orientation = pose.orientation
    translation = np.array([position.x, position.y, position.z])
    quaternion = np.array([orientation.x, orientation.y, orientation.z, orientation.w])
    rotation = Rotation.from_quat(quaternion)
    return sp.SE3(rotation.as_matrix(), np.reshape(translation, (3, 1)))


def sophus_SE3_to_ros_Pose(sp_se3: sp.SE3) -> Pose:
    """
    Convert a Sophus SE3 to a ROS pose

    Args:
        sp_se3 (sp.SE3): Sophus SE3 object

    Returns:
        Pose: ROS Pose object
    """
    pose = Pose()
    translation = sp_se3.translation()
    rotation = sp_se3.rotationMatrix()
    quat = Rotation.from_matrix(rotation).as_quat()
    pose.position.x = translation[0]
    pose.position.y = translation[1]
    pose.position.z = translation[2]
    pose.orientation.x = quat[0]
    pose.orientation.y = quat[1]
    pose.orientation.z = quat[2]
    pose.orientation.w = quat[3]
    return pose


def bd_SE3Pose_to_ros_Pose(bd_se3: SE3Pose) -> Pose:
    """
    Convert a SE3Pose to a ROS pose

    Args:
        bd_se3 (SE3Pose): Boston Dynamics SE3Pose object

    Returns:
        Pose: ROS Pose object
    """
    pose = Pose()
    translation = bd_se3.position
    quat = bd_se3.rotation
    pose.position.x = float(translation.x)
    pose.position.y = float(translation.y)
    pose.position.z = float(translation.z)
    pose.orientation.x = float(quat.x)
    pose.orientation.y = float(quat.y)
    pose.orientation.z = float(quat.z)
    pose.orientation.w = float(quat.w)
    return pose


def bd_SE3Pose_to_ros_TransformStamped(
    bd_se3: SE3Pose, parent_frame: str, child_frame: str
) -> TransformStamped:
    """
    Convert a BD SE3Pose to a ROS TransformStamped

    Args:
        bd_se3 (SE3Pose): Boston Dynamics SE3Pose object
        parent_frame (str): Parent frame name
        child_frame (str): Child frame name

    Returns:
        ros_trf_stamped (TransformStamped): TransformStamped object
    """
    ros_trf_stamped = TransformStamped()

    ros_trf_stamped.header.stamp = rospy.Time.now()
    ros_trf_stamped.header.frame_id = f"/{parent_frame}"
    ros_trf_stamped.child_frame_id = f"/{child_frame}"

    trans = bd_se3.position
    ros_trf_stamped.transform.translation.x = float(trans.x)
    ros_trf_stamped.transform.translation.y = float(trans.y)
    ros_trf_stamped.transform.translation.z = float(trans.z)

    quat = bd_se3.rotation
    ros_trf_stamped.transform.rotation.x = float(quat.x)
    ros_trf_stamped.transform.rotation.y = float(quat.y)
    ros_trf_stamped.transform.rotation.z = float(quat.z)
    ros_trf_stamped.transform.rotation.w = float(quat.w)
    return ros_trf_stamped


def sophus_SE3_to_ros_TransformStamped(
    sp_se3: sp.SE3, parent_frame: str, child_frame: str
) -> TransformStamped:
    """
    Convert a Sophus SE3 to a ROS TransformStamped

    Args:
        sp_se3 (sp.SE3): Sophus SE3 object
        parent_frame (str): Parent frame name
        child_frame (str): Child frame name

    Returns:
        ros_trf_stamped (TransformStamped): TransformStamped object
    """
    ros_trf_stamped = TransformStamped()

    ros_trf_stamped.header.stamp = rospy.Time.now()
    ros_trf_stamped.header.frame_id = f"/{parent_frame}"
    ros_trf_stamped.child_frame_id = f"/{child_frame}"

    trans = sp_se3.translation()
    ros_trf_stamped.transform.translation.x = float(trans[0])
    ros_trf_stamped.transform.translation.y = float(trans[1])
    ros_trf_stamped.transform.translation.z = float(trans[2])

    quat = Rotation.from_matrix(sp_se3.rotationMatrix()).as_quat()
    ros_trf_stamped.transform.rotation.x = float(quat[0])
    ros_trf_stamped.transform.rotation.y = float(quat[1])
    ros_trf_stamped.transform.rotation.z = float(quat[2])
    ros_trf_stamped.transform.rotation.w = float(quat[3])
    return ros_trf_stamped


def ros_TransformStamped_to_sophus_SE3(ros_trf_stamped: TransformStamped) -> sp.SE3:
    """
    Convert a ROS TransformStamped to a Sophus SE3

    Args:
        ros_trf_stamped (TransformStamped): ROS TransformStamped object

    Returns:
        sp_SE3 (sp.SE3): Sophus SE3 object
    """

    trans = np.array([0.0, 0.0, 0.0])
    trans[0] = ros_trf_stamped.transform.translation.x
    trans[1] = ros_trf_stamped.transform.translation.y
    trans[2] = ros_trf_stamped.transform.translation.z

    quat = np.array([0.0, 0.0, 0.0, 1.0])
    quat[0] = ros_trf_stamped.transform.rotation.x
    quat[1] = ros_trf_stamped.transform.rotation.y
    quat[2] = ros_trf_stamped.transform.rotation.z
    quat[3] = ros_trf_stamped.transform.rotation.w
    return sp.SE3(
        Rotation.from_quat(quat).as_matrix(), trans
    )  # Rotation.from_quat() takes a quaternion in the form of [x, y, z, w]


def sophus_SE3_to_ros_PoseStamped(sp_se3: sp.SE3, parent_frame: str) -> PoseStamped:
    """
    Convert a Sophus SE3 to a ROS PoseStamped

    Args:
        sp_se3 (sp.SE3): Sophus SE3 object
        parent_frame (str): Parent frame name

    Returns:
        ros_pse_stamped (PoseStamped): ROS PoseStamped object
    """
    ros_pse_stamped = PoseStamped()

    # pose_a_T_b.header.seq = 0 # TODO: Not sure if this is needed
    ros_pse_stamped.header.stamp = rospy.Time.now()
    ros_pse_stamped.header.frame_id = f"{parent_frame}"

    trans = sp_se3.translation()
    ros_pse_stamped.pose.position.x = float(trans[0])
    ros_pse_stamped.pose.position.y = float(trans[1])
    ros_pse_stamped.pose.position.z = float(trans[2])

    quat = Rotation.from_matrix(sp_se3.rotationMatrix()).as_quat()
    ros_pse_stamped.pose.orientation.x = float(quat[0])
    ros_pse_stamped.pose.orientation.y = float(quat[1])
    ros_pse_stamped.pose.orientation.z = float(quat[2])
    ros_pse_stamped.pose.orientation.w = float(quat[3])
    return ros_pse_stamped


def ros_PoseStamped_to_sp_SE3(ros_pse_stamped: PoseStamped) -> sp.SE3:
    """
    Convert a ROS PoseStamped to a Sophus SE3

    Args:
        ros_pse_stamped (PoseStamped): ROS PoseStamped object

    Returns:
        sp_SE3 (sp.SE3): Sophus SE3 object
    """

    trans = np.array([0.0, 0.0, 0.0])
    trans[0] = ros_pse_stamped.pose.position.x
    trans[1] = ros_pse_stamped.pose.position.y
    trans[2] = ros_pse_stamped.pose.position.z

    quat = np.array([0.0, 0.0, 0.0, 1.0])
    quat[0] = ros_pse_stamped.pose.orientation.x
    quat[1] = ros_pse_stamped.pose.orientation.y
    quat[2] = ros_pse_stamped.pose.orientation.z
    quat[3] = ros_pse_stamped.pose.orientation.w
    return sp.SE3(
        Rotation.from_quat(quat).as_matrix(), trans
    )  # Rotation.from_quat() takes a quaternion in the form of [x, y, z, w]


def np_matrix3x4_to_sophus_SE3(np_matrix: np.ndarray) -> sp.SE3:
    """
    Convert a 3x4 matrix to a Sophus SE3

    Args:
        np_matrix (np.ndarray): 3x4 matrix

    Returns:
        sp_SE3 (sp.SE3): Sophus SE3 object
    """
    rotation = np_matrix[:, :3]
    translation = np_matrix[:, 3]
    return sp.SE3(rotation, translation)
