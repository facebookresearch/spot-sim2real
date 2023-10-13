# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import time

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image
from spot_rl.utils.robot_subscriber import SpotRobotSubscriberMixin
from spot_rl.utils.utils import ros_topics as rt
from std_msgs.msg import String

TOPICS = [rt.HAND_RGB, rt.HAND_DEPTH, rt.DETECTIONS_TOPIC, rt.HAND_DEPTH_UNSCALED]
TYPES = [Image, Image, String, Image]


def preprocess_image(img, image_scale=0.7):
    if image_scale != 1.0:
        img = cv2.resize(
            img,
            (0, 0),
            fx=image_scale,
            fy=image_scale,
            interpolation=cv2.INTER_AREA,
        )

    # if self.deblur_gan is not None:
    #     img = self.deblur_gan(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # if self.grayscale:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


# DUPLICATE CODE
def get_3d_point(pixel_uv, z):
    # Get camera intrinsics
    # fx = cam_intrinsics.focal_length.x
    # fy = cam_intrinsics.focal_length.y
    # cx = cam_intrinsics.principal_point.x
    # cy = cam_intrinsics.principal_point.y

    fx = 552.0291012161067
    fy = 552.0291012161067
    cx = 320.0
    cy = 240.0

    # Get 3D point
    x = (pixel_uv[0] - cx) * z / fx
    y = (pixel_uv[1] - cy) * z / fy
    return np.array([x, y, z])


# DUPLICATE CODE
def label_img(
    img: np.ndarray,
    text: str,
    org: tuple,
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.8,
    color: tuple = (0, 0, 255),
    thickness: int = 2,
    line_type: int = cv2.LINE_AA,
):
    """
    Helper method to label image with text

    Args:
        img (np.ndarray): Image to be labeled
        text (str): Text to be labeled
        org (tuple): (x,y) position of text
        font_face (int, optional): Font face. Defaults to cv2.FONT_HERSHEY_SIMPLEX.
        font_scale (float, optional): Font scale. Defaults to 0.8.
        color (tuple, optional): Color of text. Defaults to (0, 0, 255).
        thickness (int, optional): Thickness of text. Defaults to 2.
        line_type (int, optional): Line type. Defaults to cv2.LINE_AA.
    """
    cv2.putText(
        img,
        text,
        org,
        font_face,
        font_scale,
        color,
        thickness,
        line_type,
    )


# DUPLICATE CODE
# @TODO: Maybe make position as Any?
def decorate_img_with_text(img: np.ndarray, frame: str, position: np.ndarray):
    """
    Helper method to label image with text

    Args:
        img (np.ndarray): Image to be labeled
        frame (str): Frame of reference (for labeling)
        position (np.ndarray): Position of object in frame of reference
    """
    # label_img(img, "Detected QR Marker", (50, 50), color=(0, 0, 255))
    label_img(img, f"Frame = {frame}", (50, 75), color=(0, 0, 255))
    label_img(img, f"X : {position[0]/10.0}", (50, 100), color=(0, 250, 0))
    label_img(img, f"Y : {position[1]/10.0}", (50, 125), color=(0, 250, 0))
    label_img(img, f"Z : {position[2]/10.0}", (50, 150), color=(0, 250, 0))

    return img


class HandCamViz:
    def __init__(self, headless=False, *args, **kwargs):
        self.node_name = "HandCamViz"
        self.rgb_img = np.ones((350, 500, 3), dtype=np.uint8)
        self.dep_img = np.ones((350, 500, 3), dtype=np.uint8)
        self.det_str = None
        self.dep_img_unscaled = None
        self.cv_bridge = CvBridge()

        self.rgb_windowname = "RGB"
        self.dep_windowname = "Depth"
        cv2.namedWindow(self.rgb_windowname, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.dep_windowname, cv2.WINDOW_NORMAL)
        # cv2.namedWindow("Dep", cv2.WINDOW_NORMAL)

        self.subs = [Subscriber(topic, mtype) for topic, mtype in zip(TOPICS, TYPES)]
        # rospy.Subscriber(rt.HAND_RGB, Image, self._rgb_callback)
        # rospy.Subscriber(rt.HAND_DEPTH, Image, self._dep_callback)
        # rospy.Subscriber(rt.DETECTIONS_TOPIC, String, self._detstr_callback)
        self.ts = ApproximateTimeSynchronizer(self.subs, 5, 0.2, allow_headerless=True)
        self.ts.registerCallback(self.cb_logger)

        rospy.loginfo(f"[{self.node_name}]: Waiting for images...")
        while self.rgb_img is None:
            pass
        rospy.loginfo(f"[{self.node_name}]: Received images!")

        while not rospy.is_shutdown():
            cv2.imshow(self.rgb_windowname, self.rgb_img)
            cv2.imshow(self.dep_windowname, self.dep_img)
            cv2.waitKey(1)

    def msg_to_cv2(self, *args, **kwargs) -> np.array:
        return self.cv_bridge.imgmsg_to_cv2(*args, **kwargs)

    def _rgb_callback(self, rgb_data):
        _ = str(rgb_data.header.stamp)
        self.rgb_img = preprocess_image(self.msg_to_cv2(rgb_data))
        # self.rgb_img = HandCamViz.overlay_text(self.rgb_img, timestamp, size=1.0)

    def _dep_callback(self, dep_data):
        _ = str(dep_data.header.stamp)
        self.dep_img = preprocess_image(self.msg_to_cv2(dep_data))
        # self.dep_img = HandCamViz.overlay_text(self.dep_img, timestamp, color=(125,125,125), size=1.0)

    def _detstr_callback(self, det_str_data):
        _, self.det_str = det_str_data.data.split("|")

    def _dep_unscaled_callback(self, dep_unscaled_data):
        _ = str(dep_unscaled_data.header.stamp)
        self.dep_img_unscaled = preprocess_image(
            self.msg_to_cv2(dep_unscaled_data, "mono16")
        )
        # self.dep_img_unscaled = HandCamViz.overlay_text(self.dep_img, timestamp, color=(125,125,125), size=1.0)

    def cb_logger(self, *args):
        rospy.loginfo(
            f"CB Logger - timestamp (RGB) - {args[1].header.stamp}, timestamp (Depth) - {args[1].header.stamp}, Detection StrMsg - {args[2]}"
        )
        self._rgb_callback(rgb_data=args[0])
        self._dep_callback(dep_data=args[1])
        self._detstr_callback(det_str_data=args[2])
        self._dep_unscaled_callback(dep_unscaled_data=args[3])

        if "None" not in self.det_str:
            print(self.det_str)
            self.det_str = self.det_str.split(",")
            x0 = int(self.det_str[2])
            y0 = int(self.det_str[3])
            x1 = int(self.det_str[4])
            y1 = int(self.det_str[5])
            cv2.rectangle(
                self.rgb_img, (x0, y0), (x1, y1), color=(255, 0, 0), thickness=2
            )
            cv2.rectangle(
                self.dep_img, (x0, y0), (x1, y1), color=(125, 125, 125), thickness=2
            )

            u_avg = int((x0 + x1) / 2)
            v_avg = int((y0 + y1) / 2)
            z = self.dep_img_unscaled[u_avg, v_avg]
            point_in_world = get_3d_point((u_avg, v_avg), z)
            rospy.loginfo(f"[{self.node_name}]: Center of BBox - {point_in_world}")
            decorate_img_with_text(self.rgb_img, "Hand Depth", point_in_world)
        else:
            rospy.loginfo("Empty Detection Str")

    @staticmethod
    def overlay_text(viz_img, text, color=(0, 0, 255), size=2.0, thickness=4):
        # viz_img = img.copy()
        line, font, font_size, font_thickness = (
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            size,
            thickness,
        )

        height, width = viz_img.shape[:2]
        y0, dy = 100, 100
        for i, line in enumerate(text.split("\n")):
            text_width, text_height = cv2.getTextSize(
                line, font, font_size, font_thickness
            )[0][:2]

            x = (width - text_width) // 2
            # y = (height - text_height) // 2
            y = y0 + i * dy
            cv2.putText(
                viz_img,
                line,
                (x, y),
                font,
                font_size,
                color,
                font_thickness,
                lineType=cv2.LINE_AA,
            )
        return viz_img

    def initializeWindow(self):
        cv2.namedWindow(self.named_window, cv2.WINDOW_NORMAL)

    def vis_imgs(self):
        pass


if __name__ == "__main__":
    rospy.init_node("hand_viz", disable_signals=True)
    hcviz = HandCamViz()
