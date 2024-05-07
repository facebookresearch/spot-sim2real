import time

import cv2
import magnum as mn
import numpy as np
import rospy
import zmq
from scipy.spatial.transform import Rotation as R
from spot_rl.utils.utils import ros_topics as rt
from std_msgs.msg import String


class DetectionSubscriber:
    def __init__(self):
        self.latest_message = None
        rospy.Subscriber(rt.DETECTIONS_TOPIC, String, self.callback)

    def callback(self, data):
        self.latest_message = data.data

    def get_latest_message(self):
        return self.latest_message


def connect_socket(port):

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://localhost:{port}")
    print(f"Socket Connected at {port}")
    return socket


def detect_with_rospy_subscriber(object_name, image_scale=0.7):
    """Fetch the detection result"""
    # We use rospy approach reac the detection string from topic
    rospy.set_param("object_target", object_name)
    subscriber = DetectionSubscriber()
    fetch_time_threshold = 1.0
    time.sleep(1.0)
    begin_time = time.time()
    while (time.time() - begin_time) < fetch_time_threshold:
        latest_message = subscriber.get_latest_message()
        if "None" in latest_message:
            continue
        try:
            bbox_str = latest_message.split(",")[-4:]
            break
        except Exception:
            pass

    prediction = [int(float(num) / image_scale) for num in bbox_str]
    return prediction


def segment_with_socket(rgb_image, bbox, port=21001):
    socket = connect_socket(port)
    socket.send_pyobj((rgb_image, bbox))
    return socket.recv_pyobj()


def detect_orientation(pose_magnum, to_origin, camera_pose):
    # zyx_angles = R.from_matrix(np.array(camera_pose)[:3, :3]).as_euler("zyx", True)
    # print(zyx_angles)
    # z, y, x = zyx_angles
    # intel_zyx = [x, -1.*z, y]
    # camera_pose = R.from_euler("zyx", intel_zyx, True).as_matrix()
    # trans = np.eye(4)
    # trans[:3, :3] = camera_pose
    # camera_pose = mn.Matrix4(trans)

    z_axis: np.ndarray = np.array([0, 0, 1])
    y_axis: np.ndarray = np.array([0, 1, 0])
    x_axis: np.ndarray = np.array([1, 0, 0])

    z_axis = (
        mn.Matrix4(to_origin)
        .inverted()
        .transform_vector(mn.Vector3(*z_axis))
        .normalized()
    )
    y_axis = (
        mn.Matrix4(to_origin)
        .inverted()
        .transform_vector(mn.Vector3(*y_axis))
        .normalized()
    )
    x_axis = (
        mn.Matrix4(to_origin)
        .inverted()
        .transform_vector(mn.Vector3(*x_axis))
        .normalized()
    )

    # breakpoint()

    z_axis_transformed = (
        (camera_pose @ pose_magnum).transform_vector(z_axis).normalized()
    )
    y_axis_transformed = (
        (camera_pose @ pose_magnum).transform_vector(y_axis).normalized()
    )
    x_axis_transformed = (
        (camera_pose @ pose_magnum).transform_vector(x_axis).normalized()
    )

    theta = np.rad2deg(
        np.arccos(
            np.clip(np.dot(np.array(z_axis), np.array(z_axis_transformed)), -1, 1)
        )
    )

    gamma = np.rad2deg(
        np.arccos(
            np.clip(np.dot(np.array(y_axis), np.array(y_axis_transformed)), -1, 1)
        )
    )
    alpha = np.rad2deg(
        np.arccos(
            np.clip(np.dot(np.array(x_axis), np.array(x_axis_transformed)), -1, 1)
        )
    )
    return (
        f"vertical {format(theta, '.2f')}, {format(gamma, '.2f')}, {format(alpha, '.2f')}"
        if theta < 50
        else f"horizontal {format(theta, '.2f')}, {format(gamma, '.2f')}, {format(alpha, '.2f')}"
    )


def pose_estimation(
    rgb_image,
    depth_raw,
    object_name,
    cam_intrinsics,
    body_T_intel,
    image_scale=0.7,
    port_seg=21001,
    port_pose=2100,
):

    fx = cam_intrinsics.focal_length.x
    fy = cam_intrinsics.focal_length.y
    cx = cam_intrinsics.principal_point.x
    cy = cam_intrinsics.principal_point.y

    bbox = detect_with_rospy_subscriber(object_name, image_scale)
    mask = segment_with_socket(rgb_image, bbox, port_seg)
    mask = np.dstack([mask, mask, mask]).astype(np.uint8) * 255

    K = np.eye(3)
    K[0, 0] = fx
    K[0, -1] = cx
    K[1, 1] = fy
    K[1, -1] = cy

    rgb_image = rgb_image[..., ::-1]
    pose_args = rgb_image, depth_raw, mask, K.astype(np.double)
    pose_socket = connect_socket(port_pose)
    pose_socket.send_pyobj(pose_args)

    pose, ob_in_cam_pose, to_origin, visualization = pose_socket.recv_pyobj()
    pose_magnum = mn.Matrix4(ob_in_cam_pose)  # @mn.Matrix4(to_origin).inverted()
    classification_text = detect_orientation(pose_magnum, to_origin, body_T_intel)

    visualization = cv2.putText(
        visualization,
        classification_text,
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.imshow("orientation", visualization[..., ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("pose.png", visualization[..., ::-1])
    return "side" if "vertical" in classification_text else "topdown"
