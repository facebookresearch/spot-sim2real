import numpy as np
import sophus as sp
from geometry_msgs.msg import Pose, PoseStamped
from scipy.spatial.transform import Rotation


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
