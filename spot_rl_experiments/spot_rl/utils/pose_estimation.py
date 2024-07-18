import time
from typing import List, Tuple

import cv2
import magnum as mn
import numpy as np
import quaternion
import rospy
import zmq
from scipy.spatial.transform import Rotation as R
from spot_rl.utils.rospy_light_detection import detect_with_rospy_subscriber
from spot_rl.utils.segmentation_service import connect_socket, segment_with_socket
from spot_wrapper.spot import Spot
from spot_wrapper.utils import angle_between_quat


def get_signed_angle_in_degrees_between(A: mn.Vector3, B: mn.Vector3):
    theta = np.rad2deg(
        np.arccos(
            mn.math.dot(
                A,
                B,
            )
        )
    )
    theta_signd = (
        -1.0
        * np.sign(
            mn.math.cross(
                A,
                B,
            )
            .normalized()
            .z
        )
        * theta
    )
    return theta_signd


def detect_orientation(
    cam_T_obj: mn.Matrix4,
    to_origin: np.ndarray,
    body_T_cam: mn.Matrix4,
    orientation_solver: "OrientationSolver" = None,
) -> Tuple[str, mn.Vector3, float, np.ndarray, np.ndarray]:
    """
    Given the pose of object in camera, detect whether object is vertical or horizontal & predict angles
    cam_T_obj : mn.Matrix4 4x4 pose; object pose
    to_origin: np.ndarray 4x4 mesh pose; mesh transformation to port mesh to world
    returns visualization_annotation_text:str, spinal_axis:mn.Vector3, angle_to_make_object_face_camera:float
    """
    to_origin = mn.Matrix4(to_origin)
    to_origin_inverted = to_origin.inverted()
    center_pose = cam_T_obj @ to_origin_inverted
    center_pose_body = body_T_cam @ cam_T_obj @ to_origin_inverted
    # pose_body = body_T_cam @ cam_T_obj

    z_axis: mn.Vector3 = mn.Vector3(0, 0, 1)
    y_axis: mn.Vector3 = mn.Vector3(0, 1, 0)
    x_axis: mn.Vector3 = mn.Vector3(1, 0, 0)

    # z_axis = mn.Vector3(*z_axis)
    # z_axis = to_origin.inverted().transform_vector(mn.Vector3(*z_axis)).normalized()
    # y_axis = to_origin.inverted().transform_vector(mn.Vector3(*y_axis)).normalized()
    # x_axis = to_origin.inverted().transform_vector(mn.Vector3(*x_axis)).normalized()

    z_axis_transformed_in_body = center_pose_body.transform_vector(z_axis).normalized()
    # y_axis_transformed_in_body = center_pose_body.transform_vector(y_axis).normalized()
    # x_axis_transformed_in_body = center_pose_body.transform_vector(x_axis).normalized()

    # z_axis_transformed_in_body_for_nominal_pose = mn.Vector3(0, 0, -1)
    # y_axis_transformed_in_body_for_nominal_pose = mn.Vector3(0.0, -1.0, 0.0)
    # x_axis_transformed_in_body_for_nominal_pose = mn.Vector3(1.0, 0.0, 0.0)

    z_axis_transformed_in_camera_for_nominal_pose = mn.Vector3(0.0, 1.0, 0.0)
    y_axis_transformed_in_camera_for_nominal_pose = mn.Vector3(1.0, 0.0, 0.0)
    x_axis_transformed_in_camera_for_nominal_pose = mn.Vector3(0, 0, 1.0)

    # z_axis is the merudand (spinal axis)

    z_axis_transformed_in_camera = center_pose.transform_vector(z_axis).normalized()
    y_axis_transformed_in_camera = center_pose.transform_vector(y_axis).normalized()
    x_axis_transformed_in_camera = center_pose.transform_vector(x_axis).normalized()

    #
    # TODO: Subtract theta from 180 if greater than 100

    theta_signed = get_signed_angle_in_degrees_between(
        z_axis_transformed_in_camera_for_nominal_pose, z_axis_transformed_in_camera
    )
    gamma_signed = get_signed_angle_in_degrees_between(
        y_axis_transformed_in_camera_for_nominal_pose, y_axis_transformed_in_camera
    )
    alpha_signed = get_signed_angle_in_degrees_between(
        x_axis_transformed_in_camera_for_nominal_pose, x_axis_transformed_in_camera
    )

    # if theta < 30:
    (
        object_anchor_pose_name,
        danchor_angle,
        orientation_type,
    ) = orientation_solver._determine_anchor_object_pose(
        z_axis_transformed_in_body, "body"
    )
    (
        gripper_pose_quat,
        solution_angles,
    ) = orientation_solver.get_target_grasp_quat_based_on_object_pose(
        object_anchor_pose_name, danchor_angle
    )
    # Old condition (np.abs(theta) < 50 or np.abs(180 - np.abs(theta)) < 50) or
    return (
        (
            f"{orientation_type} {format(theta_signed, '.2f')}, {format(gamma_signed, '.2f')}, {format(alpha_signed, '.2f')}"
            # if anchor_index in [1, 6]
            # else f"horizontal {format(theta_signed, '.2f')}, {format(gamma_signed, '.2f')}, {format(alpha_signed, '.2f')}"
        ),
        z_axis_transformed_in_body,
        gamma_signed,
        gripper_pose_quat,
        solution_angles,
    )


def pose_estimation(
    rgb_image: np.ndarray,
    depth_raw: np.ndarray,
    object_name: str,
    cam_intrinsics,
    body_T_cam: mn.Matrix4,
    image_src: int,
    image_scale: float = 0.7,
    port_seg: int = 21001,
    port_pose: int = 2100,
    bbox=None,
    mask=None,
    visualizationflag: bool = True,
) -> Tuple[str, mn.Vector3, float, List[float], List[float], float]:
    """
    Gets the pose estimate using zmq socket to FoundationPose thirdparty service
    rgb_image : np.ndarray [h,w,c], 0-255
    depth_image: np.ndarray [h, w], np.uin16, 0-2000.
    object_name: str,
    camera_intrinsics : spot intrinsics object
    image_src : 1 for intel & 0 for gripper camera
    image_scale: provided in config.yaml
    port_seg: socket port for segmentation service
    port_pose : socket for pose estimation service
    bbox : if provided isn't calculated
    mask: if provided segmentation  won't be run
    visualizationflag: bool, whether to show pose estimation visualization or not
    returns orientation:str (vertical or horizontal), spinal_axis:mn.vector3, angle_to_make_object_face_camera:float, t2:float time at which we get pose estimate

    """
    orientation_solver = OrientationSolver()
    fx = cam_intrinsics.focal_length.x
    fy = cam_intrinsics.focal_length.y
    cx = cam_intrinsics.principal_point.x
    cy = cam_intrinsics.principal_point.y

    bbox = (
        detect_with_rospy_subscriber(object_name, image_scale) if bbox is None else bbox
    )
    mask = segment_with_socket(rgb_image, bbox, port_seg) if mask is None else mask
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
    (
        classification_text,
        spinal_axis,
        gamma,
        gripper_pose_quat,
        solution_angles,
    ) = detect_orientation(pose_magnum, to_origin, body_T_cam, orientation_solver)
    t2 = time.time()
    if visualizationflag:
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
        # cv2.namedWindow("orientation", cv2.WINDOW_NORMAL)
        # cv2.imshow("orientation", visualization[..., ::-1])
        # cv2.waitKey(1)
        # cv2.destroyAllWindows()
        cv2.imwrite("pose.png", visualization[..., ::-1])
    orientation = "side" if "vertical" in classification_text else "topdown"
    return (
        orientation,
        spinal_axis,
        gamma,
        gripper_pose_quat.flatten().tolist(),
        solution_angles,
        t2,
    )


def quaternion_multiply(q1, q2):
    """
    Multiplies two quaternions q1 and q2.

    Args:
    q1, q2: Quaternions represented as arrays or lists of four elements [w, x, y, z].

    Returns:
    A quaternion resulting from the multiplication of q1 and q2.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return np.array([w, x, y, z])


def rotate_quaternion_around_y(
    quat: np.ndarray, grasp_orientation_name: str, gamma: float
) -> np.ndarray:
    gamma = -1.0 * np.sign(gamma) * 90
    quat_r = np.array([1.0, 0.0, 0.0, 0.0])
    if "5" in grasp_orientation_name:
        quat_r = np.array(
            [np.cos(np.deg2rad(gamma) / 2.0), 0.0, np.sin(np.deg2rad(gamma) / 2.0), 0.0]
        )  # (1./np.sqrt(2))*np.array([1., 0., 1., 0.])
    if "7" in grasp_orientation_name:
        quat_r = np.array(
            [np.cos(np.deg2rad(gamma) / 2.0), 0.0, np.sin(np.deg2rad(gamma) / 2.0), 0.0]
        )  # (1./np.sqrt(2))*np.array([1., 0., -1., 0.])
    if "6" in grasp_orientation_name:
        quat_r = np.array(
            [np.cos(np.deg2rad(gamma) / 2.0), 0.0, 0.0, np.sin(np.deg2rad(gamma) / 2.0)]
        )  # (1./np.sqrt(2))*np.array([1., 0., 0., 1.])
    if "8" in grasp_orientation_name:
        quat_r = np.array(
            [np.cos(np.deg2rad(gamma) / 2.0), 0.0, 0.0, np.sin(np.deg2rad(gamma) / 2.0)]
        )  # (1./np.sqrt(2))*np.array([1., 0., 0., -1.])
    print(f"Quat R {quat_r}, gamma {gamma} grasp name {grasp_orientation_name}")
    return quaternion_multiply(quat, quat_r)


def rotate_around_x_by_given_angle(
    quat: np.ndarray,
    grasp_orientation_name: str,
    object_anchor_pose_name: str,
    dangle: float,
):

    grasp_pose_index = int(grasp_orientation_name.split("_")[-1])
    object_pose_index = int(object_anchor_pose_name.split("_")[-1])
    if grasp_pose_index == 3 and object_pose_index == 4:
        dangle *= -1.0
    if grasp_pose_index == 3 and object_pose_index == 5:
        dangle *= 1.0
    if grasp_pose_index == 1 and object_pose_index == 2:
        dangle *= -1.0
    if grasp_pose_index == 1 and object_pose_index == 3:
        dangle *= -1.0

    if grasp_pose_index in [5, 7]:
        dangle *= 0.0
    quat_r: np.ndarray = np.array(
        [np.cos(np.deg2rad(dangle) / 2.0), np.sin(np.deg2rad(dangle) / 2.0), 0.0, 0.0]
    )
    return quaternion_multiply(quat, quat_r)


class OrientationSolver:
    """
    Performs orientation correction given the gripper pose & object pose
    """

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

        grasp_orientation_5: quaternion = quaternion.quaternion(
            0.71277446, 0.70131284, 0.00662719, 0.00830902
        )  # side grasp, intel cam on left of the spot's vision

        grasp_orientation_6: quaternion = quaternion.quaternion(
            0.99998224, 0.00505713, 0.00285832, 0.00132725
        )  # side grasp, stright, intel cam aligned with the spot's vision frame

        grasp_orientation_7: quaternion = quaternion.quaternion(
            0.71441466, -0.69967002, 0.00516158, -0.00684565
        )  # side grasp, intel cam on right of the spot's vision

        grasp_orientation_8: quaternion = quaternion.quaternion(
            -0.00951104, 0.99976665, -0.01812011, 0.00691935
        )  # side grasp upside down, -180

        # "name_of_the_pose":(pose_quat_in_body, "horizontal/vertical")
        self.object_orientations = {
            "object_orientation_1": (
                mn.Vector3(0, 1.0, 0.0),
                mn.Vector3(0, 0, -1.0),
                "vertical",
            ),  # object in normal vertical position
            "object_orientation_2": (
                mn.Vector3(1, 0.0, 0.0),
                mn.Vector3(0, -1.0, 0.0),
                "horizontal",
            ),  # object head to the left of spot; object in horizontal position
            "object_orientation_3": (
                mn.Vector3(-1, 0.0, 0.0),
                mn.Vector3(0.0, 1.0, 0.0),
                "horizontal",
            ),  # object head to the right of spot; object in horizontal position
            "object_orientation_4": (
                mn.Vector3(0, 0, -1),
                mn.Vector3(-1, 0, 0),
                "horizontal",
            ),  # object base towards the spot; object in horizontal position
            "object_orientation_5": (
                mn.Vector3(0, 0, 1),
                mn.Vector3(1, 0, 0),
                "horizontal",
            ),  # object head towards the spot; object in horizontal position
            "object_orientation_6": (
                mn.Vector3(0, -1.0, 0.0),
                mn.Vector3(0.0, 0.0, 1.0),
                "vertical",
            ),  # object in upside down vertical position
        }

        # "name_of_the_pose":(pose_quat_in_body, "topdown/side")
        self.grasp_orientations = {
            "grasp_orientation_1": (grasp_orientation_1, "topdown"),
            "grasp_orientation_2": (grasp_orientation_2, "topdown"),
            "grasp_orientation_3": (grasp_orientation_3, "topdown"),
            "grasp_orientation_4": (grasp_orientation_4, "topdown"),
            "grasp_orientation_5": (grasp_orientation_5, "side"),
            "grasp_orientation_6": (grasp_orientation_6, "side"),
            "grasp_orientation_7": (grasp_orientation_7, "side"),
            "grasp_orientation_8": (grasp_orientation_8, "side"),
        }
        # "object_pose_name_gripper_pose_name":np.ndarray([0, 0, 0])
        self.solution_space = {
            # top-down
            "object_orientation_1_grasp_orientation_1": np.array([0, 0, 0]),
            "object_orientation_1_grasp_orientation_2": np.array([0, 0, 0]),
            "object_orientation_1_grasp_orientation_3": np.array([0, 0, 0]),
            "object_orientation_1_grasp_orientation_4": np.array([0, 0, 0]),
            # side
            "object_orientation_1_grasp_orientation_5": np.array([90, 0, 0]),
            "object_orientation_1_grasp_orientation_6": np.array([0, 0, 0]),
            "object_orientation_1_grasp_orientation_7": np.array([-90, 0, 0]),
            "object_orientation_1_grasp_orientation_8": np.array([180, 0, 0]),
            # top-down
            "object_orientation_2_grasp_orientation_1": np.array([90, 0, 0]),
            "object_orientation_2_grasp_orientation_2": np.array([-90, 0, 0]),
            "object_orientation_2_grasp_orientation_3": np.array([-180, 0, 0]),
            "object_orientation_2_grasp_orientation_4": np.array([0, 0, 0]),
            # side
            "object_orientation_2_grasp_orientation_5": np.array([180, 0, 0]),
            "object_orientation_2_grasp_orientation_6": np.array([90, 0, 0]),
            "object_orientation_2_grasp_orientation_7": np.array([0, 0, 0]),
            "object_orientation_2_grasp_orientation_8": np.array([-90, 0, 0]),
            # top-down
            "object_orientation_3_grasp_orientation_1": np.array([-90, 0, 0]),
            "object_orientation_3_grasp_orientation_2": np.array([90, 0, 0]),
            "object_orientation_3_grasp_orientation_3": np.array([0, 0, 0]),
            "object_orientation_3_grasp_orientation_4": np.array([180, 0, 0]),
            # side
            "object_orientation_3_grasp_orientation_5": np.array([0, 0, 0]),
            "object_orientation_3_grasp_orientation_6": np.array([-90, 0, 0]),
            "object_orientation_3_grasp_orientation_7": np.array([180, 0, 0]),
            "object_orientation_3_grasp_orientation_8": np.array([90, 0, 0]),
            # top-down
            "object_orientation_4_grasp_orientation_1": np.array([0, 0, 0]),
            "object_orientation_4_grasp_orientation_2": np.array([180, 0, 0]),
            "object_orientation_4_grasp_orientation_3": np.array([90, 0, 0]),
            "object_orientation_4_grasp_orientation_4": np.array([-90, 0, 0]),
            # side
            "object_orientation_4_grasp_orientation_5": np.array(
                [0, 0, 0]
            ),  # infeasible
            "object_orientation_4_grasp_orientation_6": np.array(
                [0, 0, 0]
            ),  # infeasible
            "object_orientation_4_grasp_orientation_7": np.array(
                [0, 0, 0]
            ),  # infeasible
            "object_orientation_4_grasp_orientation_8": np.array(
                [0, 0, 0]
            ),  # infeasible
            # top-down
            "object_orientation_5_grasp_orientation_1": np.array([180, 0, 0]),
            "object_orientation_5_grasp_orientation_2": np.array([0, 0, 0]),
            "object_orientation_5_grasp_orientation_3": np.array([-90, 0, 0]),
            "object_orientation_5_grasp_orientation_4": np.array([90, 0, 0]),
            # side
            "object_orientation_5_grasp_orientation_5": np.array(
                [0, 0, 0]
            ),  # infeasible
            "object_orientation_5_grasp_orientation_6": np.array(
                [0, 0, 0]
            ),  # infeasible
            "object_orientation_5_grasp_orientation_7": np.array(
                [0, 0, 0]
            ),  # infeasible
            "object_orientation_5_grasp_orientation_8": np.array(
                [0, 0, 0]
            ),  # infeasible
            # top-down
            "object_orientation_6_grasp_orientation_1": np.array(
                [0, 0, 0]
            ),  # infeasible to correct, upside down object can't be corrected with top-down grasp
            "object_orientation_6_grasp_orientation_2": np.array(
                [0, 0, 0]
            ),  # infeasible to correct
            "object_orientation_6_grasp_orientation_3": np.array(
                [0, 0, 0]
            ),  # infeasible to correct
            "object_orientation_6_grasp_orientation_4": np.array(
                [0, 0, 0]
            ),  # infeasible to correct
            # side
            "object_orientation_6_grasp_orientation_5": np.array([-90, 0, 0]),
            "object_orientation_6_grasp_orientation_6": np.array([180, 0, 0]),
            "object_orientation_6_grasp_orientation_7": np.array([90, 0, 0]),
            "object_orientation_6_grasp_orientation_8": np.array([0, 0, 0]),
        }
        self.grasp_solution_for_object_pose = {
            "object_orientation_1": "grasp_orientation_5",
            "object_orientation_6": "grasp_orientation_5",
            "object_orientation_2": "grasp_orientation_1",
            "object_orientation_3": "grasp_orientation_1",
            "object_orientation_4": "grasp_orientation_3",
            "object_orientation_5": "grasp_orientation_3",
        }
        self.symmetric_object_dict = {"bottle": True, "penguin": False, "cup": True}

    def _determine_anchor_grasp_pose(
        self,
        observed_grasp_orientation_in_quat: np.ndarray,  # pick has executed & gripper is holding the object but not retracted, spot.get_ee_quaternion_in_body_frame
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
        self, spinal_axis: mn.Vector3, root_frame_name: str = "body"
    ) -> Tuple[str, float, str]:
        # find angle with all the object orientation axes stored
        # returns name of the object_pose & delta angle to the nearest anchor pose

        object_orientation_names = list(self.object_orientations.keys())
        object_pose_axes_camera, object_pose_axes_body, orientation_type = zip(
            *list(self.object_orientations.values())
        )

        object_pose_axes = (
            object_pose_axes_body
            if root_frame_name == "body"
            else object_pose_axes_camera
        )
        object_pose_axes = list(object_pose_axes)
        angles = []  # type: ignore

        for object_pose_axis in object_pose_axes:
            sign = np.sign(mn.math.cross(spinal_axis, object_pose_axis).normalized().z)
            sign = 1.0 if sign == 0.0 else sign
            angles.append(
                sign * np.rad2deg(np.arccos(mn.math.dot(spinal_axis, object_pose_axis)))
            )

        index = np.argmin(np.abs(angles))
        print(
            f"similarity angles for object orientation {angles}, angle {angles[index]} pose name {object_orientation_names[index]}"
        )

        return (
            object_orientation_names[index],
            angles[index],
            list(orientation_type)[index],
        )

    # def determine_object_orientation_from_object_pose(
    #     self, observed_object_pose_in_quat: np.ndarray
    # ) -> str:
    #     # finds whether the observed object is horizontal or vertical based on anchor poses
    #     # return in str as "horizontal" or "vertical"
    #     object_pose_name = self._determine_anchor_object_pose(
    #         observed_object_pose_in_quat
    #     )
    #     return self.object_orientations.get(object_pose_name, "")[1]
    def get_target_grasp_quat_based_on_object_pose(
        self, object_anchor_pose_name: str, danchor_angle: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given object pose tell us, how to orient the gripper to pickup & tells us whether it's top-down or side
        """
        grasp_orientation_name = self.grasp_solution_for_object_pose.get(
            object_anchor_pose_name
        )
        grasp_orientation_quat, _ = self.grasp_orientations.get(grasp_orientation_name)
        grasp_orientation_quat = rotate_around_x_by_given_angle(
            grasp_orientation_quat.view((np.double, 4)),
            grasp_orientation_name,
            object_anchor_pose_name,
            -1 * danchor_angle,
        )
        solution_angle = self.solution_space.get(
            f"{object_anchor_pose_name}_{grasp_orientation_name}",
            np.array([0, 0, 0]),
        )
        return grasp_orientation_quat, solution_angle

    def get_correction_angle(
        self,
        observed_grasp_pose_in_quat: np.ndarray,
        spinal_axis: mn.Vector3,
        root_frame_name: str = "body",
    ):
        # all poses are to be in body frame
        current_grasp_orientation = self._determine_anchor_grasp_pose(
            observed_grasp_pose_in_quat
        )
        (
            current_object_orientation,
            delta_to_pose,
            _,
        ) = self._determine_anchor_object_pose(spinal_axis, root_frame_name)
        print(
            f"Object pose {current_object_orientation}, delta to anchor pose {delta_to_pose}, grasp orientation {current_grasp_orientation}"
        )
        assert (
            f"{current_object_orientation}_{current_grasp_orientation}"
            in self.solution_space
        ), f"Couldn't find {current_object_orientation}_{current_grasp_orientation} in solution space"

        # convert solution from degtorad
        return self.solution_space.get(
            f"{current_object_orientation}_{current_grasp_orientation}",
            np.array([0, 0, 0]),
        )

    def determine_if_object_symmetric(self, object_name: str, gamma: float) -> float:
        for (key, value) in self.symmetric_object_dict.items():
            if key in object_name and value:
                return 0
        return gamma

    def perform_orientation_correction(
        self,
        spot: Spot,
        spinal_axis: mn.Vector3,
        gamma: float,
        object_name: str,
        make_face_the_object_right: bool = False,
    ):
        correction_status, put_back_object_status = False, False

        current_orientation_at_grasp_in_quat = (
            spot.get_ee_quaternion_in_body_frame().view((np.double, 4))
        )
        current_point, _ = spot.get_ee_pos_in_body_frame()
        correction_angles = self.get_correction_angle(
            current_orientation_at_grasp_in_quat, spinal_axis
        )
        print(
            f"the orientation correction, current ee pos point in body {current_point}, correction_angles {correction_angles}"
        )

        # Correct the orientation 0.1 m above the current position
        current_point[-1] += 0.10
        correction_status = spot.move_gripper_to_points(
            current_point, [np.deg2rad([0, 0, 0]), np.deg2rad(correction_angles)], 5, 10
        )
        current_orientation_after_grasp_in_quat = (
            spot.get_ee_quaternion_in_body_frame().view((np.double, 4))
        )
        current_grasp_orientation_name = self._determine_anchor_grasp_pose(
            current_orientation_after_grasp_in_quat
        )
        current_point[-1] -= 0.10

        # if gamma > 30, object needs correction along z-axis of spot frame
        gamma = self.determine_if_object_symmetric(object_name, gamma)
        q_final = (
            rotate_quaternion_around_y(
                current_orientation_after_grasp_in_quat,
                current_grasp_orientation_name,
                gamma,
            )
            if np.abs(gamma) > 30 and make_face_the_object_right
            else current_orientation_after_grasp_in_quat
        )
        put_back_object_status = spot.move_gripper_to_point(current_point, q_final)

        spot.open_gripper()
        return correction_status, put_back_object_status
