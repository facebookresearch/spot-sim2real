import time
from typing import Tuple

import cv2
import magnum as mn
import numpy as np
import quaternion
import rospy
import zmq
from scipy.spatial.transform import Rotation as R
from spot_rl.utils.utils import ros_topics as rt
from spot_wrapper.utils import angle_between_quat
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


def detect_orientation(cam_T_obj, to_origin, body_T_cam):
    # zyx_angles = R.from_matrix(np.array(camera_pose)[:3, :3]).as_euler("zyx", True)
    # print(zyx_angles)
    # z, y, x = zyx_angles
    # intel_zyx = [x, -1.*z, y]
    # camera_pose = R.from_euler("zyx", intel_zyx, True).as_matrix()
    # trans = np.eye(4)
    # trans[:3, :3] = camera_pose
    # camera_pose = mn.Matrix4(trans)
    to_origin = mn.Matrix4(to_origin)
    
    z_axis: np.ndarray = np.array([0, 0, 1])
    y_axis: np.ndarray = np.array([0, 1, 0])
    x_axis: np.ndarray = np.array([1, 0, 0])

    # z_axis = mn.Vector3(*z_axis)
    z_axis = (
        to_origin
        .inverted()
        .transform_vector(mn.Vector3(*z_axis))
        .normalized()
    )
    y_axis = (
        to_origin
        .inverted()
        .transform_vector(mn.Vector3(*y_axis))
        .normalized()
    )
    x_axis = (
        to_origin
        .inverted()
        .transform_vector(mn.Vector3(*x_axis))
        .normalized()
    )

    z_axis_transformed_in_body = (
        (body_T_cam @ cam_T_obj).transform_vector(z_axis).normalized()
    )
    y_axis_transformed_in_body = (
        (body_T_cam @ cam_T_obj).transform_vector(y_axis).normalized()
    )
    x_axis_transformed_in_body = (
        (body_T_cam @ cam_T_obj).transform_vector(x_axis).normalized()
    )

    z_axis_transformed_in_body_for_nominal_pose = mn.Vector3(0, 0, -1)
    y_axis_transformed_in_body_for_nominal_pose = mn.Vector3(0., -1., 0.)
    x_axis_transformed_in_body_for_nominal_pose = mn.Vector3(1., 0., 0.)

    z_axis_transformed_in_camera_for_nominal_pose = mn.Vector3(0., 1., 0.)
    y_axis_transformed_in_camera_for_nominal_pose = mn.Vector3(1., 0., 0.)
    x_axis_transformed_in_camera_for_nominal_pose = mn.Vector3(0, 0, 1.)

    #z_axis is the merudand (spinal axis)

    z_axis_transformed_in_camera = cam_T_obj.transform_vector(z_axis).normalized()
    y_axis_transformed_in_camera = cam_T_obj.transform_vector(y_axis).normalized()
    x_axis_transformed_in_camera = cam_T_obj.transform_vector(x_axis).normalized()
    

    #breakpoint()
    # 
    # TODO: Subtract theta from 180 if greater than 100

    theta = np.rad2deg(np.arccos(mn.math.dot(z_axis_transformed_in_camera_for_nominal_pose, z_axis_transformed_in_camera)))
    gamma = np.rad2deg(np.arccos(mn.math.dot(y_axis_transformed_in_camera_for_nominal_pose, y_axis_transformed_in_camera)))
    alpha = np.rad2deg(np.arccos(mn.math.dot(x_axis_transformed_in_camera_for_nominal_pose, x_axis_transformed_in_camera)))
    

    theta_signed = (
        -1.0
        * np.sign(mn.math.cross(z_axis_transformed_in_camera_for_nominal_pose, z_axis_transformed_in_camera).normalized().z)
        * theta
    )
    gamma_signed = (
        -1
        * np.sign(mn.math.cross(y_axis_transformed_in_camera_for_nominal_pose, y_axis_transformed_in_camera).normalized().z)
        * gamma
    )
    alpha_signed = (
        -1
        * np.sign(mn.math.cross(x_axis_transformed_in_camera_for_nominal_pose, x_axis_transformed_in_camera).normalized().z)
        * alpha
    )

    # if theta < 30:

    return (
        (
            f"vertical {format(theta_signed, '.2f')}, {format(gamma_signed, '.2f')}, {format(alpha_signed, '.2f')}"
            if theta < 50
            else f"horizontal {format(theta_signed, '.2f')}, {format(gamma_signed, '.2f')}, {format(alpha_signed, '.2f')}"
        ),
        z_axis_transformed_in_camera
    )


# left -> [ 0.97110635 -0.10340448  0.01247738]
# right -> [-0.92076164 -0.12999853 -0.01665543]
def pose_estimation(
    rgb_image,
    depth_raw,
    object_name,
    cam_intrinsics,
    image_src,
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
    pose_args = rgb_image, depth_raw, mask, K.astype(np.double), image_src, object_name
    pose_socket = connect_socket(port_pose)
    pose_socket.send_pyobj(pose_args)

    pose, ob_in_cam_pose, to_origin, visualization = pose_socket.recv_pyobj()
    pose_magnum = mn.Matrix4(ob_in_cam_pose)  # @mn.Matrix4(to_origin).inverted()
    classification_text, meru_dand_of_object_in_cam = detect_orientation(
        pose_magnum, to_origin, body_T_intel
    )

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
    cv2.namedWindow("orientation", cv2.WINDOW_NORMAL) 
    cv2.imshow("orientation", visualization[..., ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("pose.png", visualization[..., ::-1])
    orientation = "side" if "vertical" in classification_text else "topdown"

    # Correct orientation

    return orientation, meru_dand_of_object_in_cam


class OrientationSolver:
    def __init__(self):
        # we aim to support 10 anchor object poses, 5 orientations facing the camera & 5 with back of the object facing the camera
        # with 8 different grasp positions 4 in top-down & 4 in side
        # we save quaternion of the anchor positions described above & at run time we find the alignment of the observed quaternion with the stored one to infer approx anchor pose.
        # given the anchor object pose & anchor grasp orientation only one unique solution exists.
        grasp_orientation_1: quaternion = quaternion.quaternion(
            7.58561254e-01, -4.43358993e-04, 6.51600122e-01, 1.37600256e-03
        )  # top-down, staright-down grasp
        grasp_orientation_2: quaternion = quaternion.quaternion(
            0.0264411, 0.71446186, 0.06067035, -0.69653732
        )  # top-down, inverse-down-grab
        grasp_orientation_3: quaternion = quaternion.quaternion(
            0.50645834, 0.50158179, 0.49495414, -0.49692667
        )  # top-down, intel cam on left of the spot's vision
        grasp_orientation_4: quaternion = quaternion.quaternion(
            0.50674087, -0.5018484, 0.49400634, 0.49731243
        )  # top-down, intel cam on right of the spot's vision

        # "name_of_the_pose":(pose_quat_in_body, "horizontal/vertical")
        self.object_orientations = {
            "object_orientation_1": (mn.Vector3(0, 1., 0.), "vertical"), #object in normal vertical position
            "object_orientation_2": (mn.Vector3(1, 0., 0.), "horizontal"), #object head to the left of spot; object in horizontal position
            "object_orientation_3": (mn.Vector3(-1, 0., 0.), "horizontal"), #object head to the right of spot; object in horizontal position
            "object_orientation_4": (mn.Vector3(0, 0, -1), "horizontal"),  #object base towards the spot; object in horizontal position
            "object_orientation_5": (mn.Vector3(0, 0, 1), "horizontal"), #object head towards the spot; object in horizontal position
            "object_orientation_6": (mn.Vector3(0, -1., 0.), "vertical") #object in upside down vertical position

        }
        self.object_orientation_by_axes_transform = {
            "object_orientation_5": ((np.array([0., 0., 1.]) , np.array([0, 1., 0]), np.array([1., 0, 0])), "horizontal"),

        }
        # "name_of_the_pose":(pose_quat_in_body, "topdown/side")
        self.grasp_orientations = {
            "grasp_orientation_1": (grasp_orientation_1, "topdown"),
            "grasp_orientation_2": (grasp_orientation_2, "topdown"),
            "grasp_orientation_3": (grasp_orientation_3, "topdown"),
            "grasp_orientation_4": (grasp_orientation_4, "topdown"),
        }
        # "object_pose_name_gripper_pose_name":np.ndarray([0, 0, 0])
        self.solution_space = {
            "object_orientation_1_grasp_orientation_1" : np.array([0, 0, 0]),
            "object_orientation_1_grasp_orientation_2" : np.array([0, 0, 0]),
            "object_orientation_1_grasp_orientation_3" : np.array([0, 0, 0]),
            "object_orientation_1_grasp_orientation_4" : np.array([0, 0, 0]),

            "object_orientation_2_grasp_orientation_1" : np.array([90, 0, 0]),
            "object_orientation_2_grasp_orientation_2" : np.array([-90, 0, 0]),
            "object_orientation_2_grasp_orientation_3" : np.array([-180, 0, 0]),
            "object_orientation_2_grasp_orientation_4" : np.array([0, 0, 0]),

            "object_orientation_3_grasp_orientation_1" : np.array([-90, 0, 0]),
            "object_orientation_3_grasp_orientation_2" : np.array([90, 0, 0]),
            "object_orientation_3_grasp_orientation_3" : np.array([0, 0, 0]),
            "object_orientation_3_grasp_orientation_4" : np.array([180, 0, 0]),

            "object_orientation_4_grasp_orientation_1" : np.array([0, 0, 0]),
            "object_orientation_4_grasp_orientation_2" : np.array([180, 0, 0]),
            "object_orientation_4_grasp_orientation_3" : np.array([90, 0, 0]),
            "object_orientation_4_grasp_orientation_4" : np.array([-90, 0, 0]),

            "object_orientation_5_grasp_orientation_1" : np.array([180, 0, 0]),
            "object_orientation_5_grasp_orientation_2" : np.array([0, 0, 0]),
            "object_orientation_5_grasp_orientation_3" : np.array([-90, 0, 0]),
            "object_orientation_5_grasp_orientation_4" : np.array([90, 0, 0]),

            "object_orientation_6_grasp_orientation_1" : np.array([0, 0, 0]), #infeasible to correct, upside down object can't be corrected with top-down grasp
            "object_orientation_6_grasp_orientation_2" : np.array([0, 0, 0]), #infeasible to correct
            "object_orientation_6_grasp_orientation_3" : np.array([0, 0, 0]), #infeasible to correct
            "object_orientation_6_grasp_orientation_4" : np.array([0, 0, 0]), #infeasible to correct

        }

    def _determine_anchor_grasp_pose(
        self, observed_grasp_orientation_in_quat: np.ndarray #pick has executed & gripper is holding the object but not retracted, spot.get_ee_quaternion_in_body_frame
    ) -> str:
        # find angle with all the grasp orientations stored
        # returns name of the grasp pose
        grasp_orientation_names = list(self.grasp_orientations.keys())
        grasp_orientations, _ = zip(*list(self.grasp_orientations.values()))
        grasp_orientations = list(grasp_orientations)
        angles = []
        for grasp_orientation in grasp_orientations:
            angles.append(
                np.rad2deg(
                    angle_between_quat(
                        grasp_orientation,
                        quaternion.quaternion(*observed_grasp_orientation_in_quat),
                    )
                )
            )
        print(f"similarity angles for grasp orientation {angles}")
        return grasp_orientation_names[np.argmin(angles)]

    def _determine_anchor_object_pose(
        self, meru_dand_of_object_in_cam:mn.Vector3
    ) -> Tuple[str, float]:
        # find angle with all the object orientation axes stored
        # returns name of the object_pose & delta angle to the nearest anchor pose
        
        object_orientation_names = list(self.object_orientations.keys())
        object_pose_axes, _ = zip(*list(self.object_orientations.values()))
        object_pose_axes = list(object_pose_axes)
        angles = []
    
        angles = []
        for object_pose_axis in object_pose_axes:
            sign = np.sign(mn.math.cross(meru_dand_of_object_in_cam, object_pose_axis).normalized().z)
            sign = 1. if sign == 0. else sign
            angles.append(sign*np.rad2deg(np.arccos(mn.math.dot(meru_dand_of_object_in_cam, object_pose_axis))))
            
        print(f"similarity angles for object orientation {angles}")
        index = np.argmin(np.abs(angles))
        return object_orientation_names[index], angles[index]
      
    # def determine_object_orientation_from_object_pose(
    #     self, observed_object_pose_in_quat: np.ndarray
    # ) -> str:
    #     # finds whether the observed object is horizontal or vertical based on anchor poses
    #     # return in str as "horizontal" or "vertical"
    #     object_pose_name = self._determine_anchor_object_pose(
    #         observed_object_pose_in_quat
    #     )
    #     return self.object_orientations.get(object_pose_name, "")[1]

    def get_correction_angle(
        self, observed_grasp_pose_in_quat:np.ndarray, meru_dand_of_object_in_cam:mn.Vector3
    ):
        # all poses are to be in body frame
        current_grasp_orientation = self._determine_anchor_grasp_pose(
            observed_grasp_pose_in_quat
        )
        current_object_orientation, delta_to_pose = self._determine_anchor_object_pose(
            meru_dand_of_object_in_cam
        )
        print(f"Object pose {current_object_orientation}, delta to anchor pose {delta_to_pose}, grasp orientation {current_grasp_orientation}")
        assert f"{current_object_orientation}_{current_grasp_orientation}" in self.solution_space, f"Couldn't find {current_object_orientation}_{current_grasp_orientation} in solution space"
        
        # convert solution from degtorad
        return self.solution_space.get(
            f"{current_object_orientation}_{current_grasp_orientation}",
            np.array([0, 0, 0]),
        )


# transform = np.eye(4)
# transform[:3, :3] = R.from_quat(current_orientation_at_grasp_in_quat).as_matrix()
# transform:mn.Matrix4 = mn.Matrix4(transform)

# x_axis_transformed = transform.transform_vector(mn.Vector3(1, 0, 0)).normalized()
# y_axis_transformed = transform.transform_vector(mn.Vector3(0, 1, 0)).normalized()
# z_axis_transformed = transform.transform_vector(mn.Vector3(0, 0, 1)).normalized()

# angle_with_x = np.sign(mn.math.cross(mn.Vector3(1, 0, 0), x_axis_transformed).normalized().z)*np.rad2deg(np.arccos(mn.math.dot(mn.Vector3(1, 0, 0), x_axis_transformed)))
# angle_with_y = np.sign(mn.math.cross(mn.Vector3(0, 1, 0), y_axis_transformed).normalized().z)*np.rad2deg(np.arccos(mn.math.dot(mn.Vector3(0, 1, 0), y_axis_transformed)))
# angle_with_z = np.sign(mn.math.cross(mn.Vector3(0, 0, 1), z_axis_transformed).normalized().z)*np.rad2deg(np.arccos(mn.math.dot(mn.Vector3(0, 0, 1), z_axis_transformed)))
