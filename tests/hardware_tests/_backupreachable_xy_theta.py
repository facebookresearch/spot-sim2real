import json

import cv2
import magnum as mn
import numpy as np
import quaternion
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Quaternion, Transform, TransformStamped, Vector3
from message_filters import ApproximateTimeSynchronizer, Subscriber
from scipy import stats as st
from sensor_msgs.msg import Image
from spot_rl.models.yolov8predictor import YOLOV8Predictor
from spot_rl.utils.utils import construct_config
from spot_rl.utils.utils import ros_topics as rt
from spot_wrapper.spot import Spot
from std_msgs.msg import String


def get_3d_point(cam_intrinsics, pixel_uv, z):
    # Get camera intrinsics
    fx = float(cam_intrinsics["focal_length"]["x"])
    fy = float(cam_intrinsics["focal_length"]["y"])
    cx = float(cam_intrinsics["principal_point"]["x"])
    cy = float(cam_intrinsics["principal_point"]["y"])

    # print(fx, fy, cx, cy)
    # Get 3D point
    x = (pixel_uv[0] - cx) * z / fx
    y = (pixel_uv[1] - cy) * z / fy
    return np.array([x, y, z])


def get_3d_points(cam_intrinsics, pixels_uv: np.ndarray, zs: np.ndarray):
    # pixels_uv = nx2 xs -> :, 1
    # Get camera intrinsics
    fx = float(cam_intrinsics["focal_length"]["x"])
    fy = float(cam_intrinsics["focal_length"]["y"])
    cx = float(cam_intrinsics["principal_point"]["x"])
    cy = float(cam_intrinsics["principal_point"]["y"])

    # Get 3D point
    xs = (pixels_uv[:, 1] - cx) * zs / fx  # n
    ys = (pixels_uv[:, 0] - cy) * zs / fy  # n
    return np.array([xs.flatten(), ys.flatten(), zs]).reshape(-1, 3)


def get_best_uvz_from_detection(unscaled_dep_img, detection):
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
        return (center_x, center_y), depth_patch_in_bbox[closest_depth_to_mu] * 0.001
    return (center_x, center_y), 0


def get_z_offset_by_corner_detection(
    rgb_depth_mixed_image, unscaled_dep_img, ball_detection, z, intrinsics
):
    # center_x = (ball_detection[0] + ball_detection[2])/2
    # center_y = (ball_detection[1] + ball_detection[3])/2
    # bottom_center_x, bottom_center_y = center_x, ball_detection[3]
    gray = cv2.cvtColor(rgb_depth_mixed_image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)
    # Threshold for an optimal value, it may vary depending on the image.
    corners_yxs = np.argwhere(dst > 0.01 * dst.max()).reshape(-1, 2)

    zs = unscaled_dep_img[corners_yxs[:, 0], corners_yxs[:, 1]] * 0.001
    non_zero_indices = np.argwhere(zs > 0.0).flatten()
    corners_yxs = corners_yxs[non_zero_indices]
    zs = zs[non_zero_indices]
    if len(corners_yxs) > 0:
        # point_3ds = get_3d_points(imgs[1].source.pinhole.intrinsics, corners_yxs, zs)
        indices_closer = np.argwhere(zs < z).flatten()
        zs = zs[indices_closer]
        corners_yxs = corners_yxs[indices_closer]
        if len(corners_yxs) > 0:
            # y_limit = get_3d_point(intrinsics, (bottom_center_x, bottom_center_y), z)[1]
            # point_3ds = get_3d_points(intrinsics, corners_yxs, zs)

            indices_below = np.argwhere(
                corners_yxs[:, 0] > ball_detection[-1]
            ).flatten()
            zs = zs[indices_below]
            corners_yxs = corners_yxs[indices_below]
            # point_3ds = point_3ds[indices_below]
            if len(corners_yxs) > 0:
                # project bottom 2D points straight
                indices_in_x_limits = np.argwhere(
                    np.logical_and(
                        (ball_detection[0] <= corners_yxs[:, 1]),
                        (corners_yxs[:, 1] <= ball_detection[2]),
                    )
                ).flatten()
                if len(indices_in_x_limits) > 0:
                    zs = zs[indices_in_x_limits]
                    corners_yxs = corners_yxs[indices_in_x_limits]
                    # point_3ds = point_3ds[indices_in_x_limits]
                    mu, std = st.norm.fit(zs)
                    combined_score = 1 * np.absolute(
                        zs - mu
                    )  # + 2*distance_from_object_to_corners

                    # print("Gaussian Depth Distribution for the corners ", mu, std)
                    min_arg = np.argmin(combined_score)
                    best_corner_yx = corners_yxs[min_arg]
                    best_offseted_z = zs[min_arg]
                    return True, best_offseted_z, best_corner_yx, corners_yxs
                else:
                    return False, "couldnot find best corner within x limit", None, None
            else:
                return False, "couldnot find best corner below y limit", None, None
        else:
            return False, "couldnot find best corner closer than z", None, None
    else:
        return False, "couldnot find best non zero depth corner", None, None


def convert_point_from_local_to_global_nav_target(
    point_in_local_3d: np.ndarray, spot, vision_T_hand: Transform
):
    pos, quat = vision_T_hand.translation, vision_T_hand.rotation
    quat = quaternion.quaternion(quat.w, quat.x, quat.y, quat.z)
    rotation_matrix = mn.Quaternion(quat.imag, quat.real).to_matrix()
    translation = mn.Vector3(pos.x, pos.y, pos.z)
    origin_T_hand = mn.Matrix4.from_(rotation_matrix, translation)
    point_in_global_3d: mn.Vector3 = origin_T_hand.transform_point(
        mn.Vector3(*point_in_local_3d)
    )
    point_in_global_3d = np.array(
        [point_in_global_3d.x, point_in_global_3d.y, point_in_global_3d.z]
    )
    # print(f"Point in global 3d, {point_in_global_3d}")
    theta = np.arctan(point_in_global_3d[1] / point_in_global_3d[0])
    # print(f"theta before transform {np.degrees(theta)}")
    global_x, global_y, transformed_theta = spot.xy_yaw_global_to_home(
        point_in_global_3d[0], point_in_global_3d[1], theta
    )
    # print(f"transformed theta {np.degrees(transformed_theta)}")
    return (global_x, global_y), transformed_theta


def euclidean(x1, y1, x2, y2):
    return np.sqrt(np.square([x2 - x1, y2 - y1]).sum())


def wrap_heading(heading):
    """Ensures input heading is between -180 an 180; can be float or np.ndarray"""
    return (heading + np.pi) % (2 * np.pi) - np.pi


def pull_back_point_along_ray_by_offset(x: float, y: float, offset: float = 0.5):
    t = np.sqrt(np.square([x, y]).sum())
    t_ = t - offset
    return (x * t_) / t, (y * t_) / t


class ReachableXYTheta:
    def __init__(self):

        self.config = config = construct_config()
        self._enable_nav_goal_change = config.ENABLE_NAV_GOAL_CHANGE
        self.use_yolov8 = False
        self.node_name = "ReachableXYTheta"
        self.image_scale = float(config.IMAGE_SCALE)
        self.spot = Spot("JustforTforms")
        self.rgb_img = np.zeros((480, 640, 3), dtype=np.uint8)
        self.prev_x, self.prev_y, self.prev_theta = None, None, None
        rospy.init_node(self.node_name, disable_signals=True)
        self.global_x, self.global_y, self.theta, self.header_stamp = (
            None,
            None,
            None,
            None,
        )
        if self._enable_nav_goal_change:

            self.cv_bridge = CvBridge()
            self.publish_topic = rt.REACHABLEXYTHETA_TOPIC
            TOPICS = [
                rt.HAND_RGB,
                rt.HAND_DEPTH,
                rt.DETECTIONS_TOPIC,
                rt.HAND_DEPTH_UNSCALED,
                rt.VISION_T_HAND,
                rt.HAND_CAM_INTRINSICS,
            ]
            TYPES = [Image, Image, String, Image, TransformStamped, String]

            self.subs = [
                Subscriber(topic, mtype) for topic, mtype in zip(TOPICS, TYPES)
            ]
            self.pubs = [
                rospy.Publisher(
                    self.publish_topic, String, queue_size=1, tcp_nodelay=True
                )
            ]
            self.ts = ApproximateTimeSynchronizer(
                self.subs, 1, 0.2, allow_headerless=True
            )
            self.ts.registerCallback(self.cb_data)
            rospy.loginfo(f"[{self.node_name}]: Waiting for images & depth ...")
            if self.use_yolov8:
                self.yolov8predictor = YOLOV8Predictor("yolov8x.torchscript")

    def object_detection(self, rgb_img):
        detections, _ = self.yolov8predictor(rgb_img, False)
        for d in detections:
            if d[-1] == 32.0:  # 32.0:
                return True, "ball", d[:5]
        return False, "", [None, None, None, None, None]

    def scale_detections_from_owlvit(self, x1, y1, x2, y2, conf):
        x1, y1, x2, y2, conf = float(x1), float(y1), float(x2), float(y2), float(conf)
        x1, y1, x2, y2 = (
            x1 / self.image_scale,
            y1 / self.image_scale,
            x2 / self.image_scale,
            y2 / self.image_scale,
        )

        return x1, y1, x2, y2, conf

    def cb_data(self, *args):
        # rospy.loginfo(
        #   f"CB Logger - timestamp (RGB) - {args[0].header.stamp}, timestamp (Depth) - {args[1].header.stamp}, Detection StrMsg - {args[2]}"
        # )
        detection_str: str = str(args[2].data)
        rgb_img = self.cv_bridge.imgmsg_to_cv2(args[0])
        self.rgb_img = rgb_img.copy()
        if "None" not in detection_str or self.use_yolov8:
            object_target = rospy.get_param("/object_target")
            self.header_stamp, detection_str = detection_str.split("|")

            if not self.use_yolov8:
                detected_object_type, conf, x1, y1, x2, y2 = detection_str.split("|")[
                    -1
                ].split(",")
                x1, y1, x2, y2, conf = self.scale_detections_from_owlvit(
                    x1, y1, x2, y2, conf
                )
            else:
                det_status, detected_object_type, det = self.object_detection(rgb_img)
                if det_status:
                    x1, y1, x2, y2, conf = det
                else:
                    return 0

            x1_int, y1_int, x2_int, y2_int = int(x1), int(y1), int(x2), int(y2)

            unscaled_depth = self.cv_bridge.imgmsg_to_cv2(args[3], "mono16")
            (u, v), z = get_best_uvz_from_detection(unscaled_depth, [x1, y1, x2, y2])
            self.rgb_img = cv2.rectangle(
                self.rgb_img,
                (x1_int, y1_int),
                (x2_int, y2_int),
                color=(0, 0, 255),
                thickness=2,
            )
            if object_target == detected_object_type and z >= 0.5 and z <= 2.5:

                hand_depth_img = self.cv_bridge.imgmsg_to_cv2(args[1])
                vision_T_hand = args[4]
                cam_intrinsics = json.loads(str(args[5].data))
                binary_depth_img = np.where(hand_depth_img > 0, 1, 0)
                binary_depth_img = np.uint8(binary_depth_img)
                mixed_image = rgb_img * binary_depth_img[:, :, None]

                point_in_local_3d = get_3d_point(cam_intrinsics, (u, v), z)
                (
                    corner_det_status,
                    best_z,
                    best_corner_yx,
                    other_best_yxs,
                ) = get_z_offset_by_corner_detection(
                    mixed_image, unscaled_depth, [x1, y1, x2, y2], z, cam_intrinsics
                )
                offset = 0
                if corner_det_status:
                    offset = point_in_local_3d[-1] - best_z
                    self.rgb_img = cv2.circle(
                        self.rgb_img,
                        (best_corner_yx[-1], best_corner_yx[0]),
                        7,
                        (0, 0, 255),
                        thickness=-1,
                    )
                    for yx in other_best_yxs:
                        self.rgb_img = cv2.circle(
                            self.rgb_img, (yx[-1], yx[0]), 1, (255, 0, 0)
                        )
                    # cv2.imwrite("RGBwithDet.png", self.rgb_img)
                # point_in_local_3d[-1] -= (offset + 1.0)
                (
                    self.global_x,
                    self.global_y,
                ), self.theta = convert_point_from_local_to_global_nav_target(
                    point_in_local_3d, self.spot, vision_T_hand.transform
                )
                org = (x1_int - 200, y1_int - 50)
                text = "({:.2f}, {:.2f}, {:.2f}), O:{:.2f}, Z:{:.2f}".format(
                    self.global_x, self.global_y, self.theta, offset, z
                )
                self.rgb_img = cv2.putText(
                    self.rgb_img,
                    text=text,
                    org=org,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.9,
                    color=(0, 0, 255),
                    thickness=2,
                )
                rospy.loginfo(
                    f"CB Logger - {self.header_stamp} Adjusted X,Y,Theta {self.global_x, self.global_y, self.theta} detected offset {offset} z : {z}"
                )
            elif object_target == detected_object_type:

                pass
                # rospy.loginfo(
                #    f"Should have worked - {self.header_stamp} detected str {detection_str}, rospy param {object_target} distance to  target {z}"
                # )

    def publish(self):
        if self._enable_nav_goal_change:
            # if self.prev_x is not None:
            #     dist = euclidean(self.prev_x, self.prev_y, self.global_x, self.global_y)
            #     ang_dist = wrap_heading(self.theta - self.prev_theta)
            #     if dist <= 0.1 or np.abs(ang_dist) < 0.0873:
            #         self.global_x = self.global_y = self.theta = None
            # if self.theta is not None:
            #     self.prev_x, self.prev_y, self.prev_theta = self.global_x, self.global_y, self.theta
            self.pubs[0].publish(
                String(
                    data=f"{self.header_stamp}|{self.global_x}, {self.global_y}, {self.theta}"
                )
            )

        return self.rgb_img


if __name__ == "__main__":

    reachablexytheta = ReachableXYTheta()
    cv2.namedWindow("corner_detection")
    while not rospy.is_shutdown():
        img = reachablexytheta.publish()
        cv2.imshow("corner_detection", img)
        if cv2.waitKey(10) == 13:
            break
    cv2.destroyAllWindows()
