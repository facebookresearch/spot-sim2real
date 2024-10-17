# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os.path as osp
import subprocess
import time
import traceback
from copy import deepcopy
from typing import Any, List

import blosc
import cv2
import magnum as mn
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from spot_rl.utils.depth_map_utils import filter_depth
from spot_rl.utils.tracking_service import tracking_with_socket
from spot_wrapper.spot import Spot
from spot_wrapper.spot import SpotCamIds as Cam
from spot_wrapper.spot import image_response_to_cv2, scale_depth_img
from std_msgs.msg import (
    ByteMultiArray,
    Header,
    MultiArrayDimension,
    MultiArrayLayout,
    String,
)

try:
    from spot_rl.utils.mask_rcnn_utils import (
        generate_mrcnn_detections,
        get_deblurgan_model,
        get_mrcnn_model,
        pred2string,
    )
except ModuleNotFoundError:
    pass

# owlvit
from spot_rl.models import OwlVit
from spot_rl.utils.pixel_to_3d_conversion_utils import (
    get_3d_point,
    sample_patch_around_point,
)
from spot_rl.utils.stopwatch import Stopwatch
from spot_rl.utils.utils import construct_config
from spot_rl.utils.utils import ros_topics as rt

MAX_PUBLISH_FREQ = 20
MAX_DEPTH = 3.5
MAX_HAND_DEPTH = 1.7
DETECTIONS_BUFFER_LEN = 30
LEFT_CROP = 124
RIGHT_CROP = 60
NEW_WIDTH = 228
NEW_HEIGHT = 240
ORIG_WIDTH = 640
ORIG_HEIGHT = 480
WIDTH_SCALE = 0.5
HEIGHT_SCALE = 0.5


class SpotImagePublisher:
    name = ""
    publisher_topics = List[str]
    publish_msg_type = Image

    def __init__(self):
        rospy.init_node(self.name, disable_signals=True)
        self.cv_bridge = CvBridge()
        self.last_publish = time.time()
        self.pubs = {
            k: rospy.Publisher(k, self.publish_msg_type, queue_size=1, tcp_nodelay=True)
            for k in self.publisher_topics
        }
        rospy.loginfo(f"[{self.name}]: Publisher initialized.")

    def publish(self):
        # if st < self.last_publish + 1 / MAX_PUBLISH_FREQ:
        #     time.sleep(0.01)
        #     return
        self._publish()
        self.last_publish = time.time()

    def cv2_to_msg(self, *args, **kwargs) -> Image:
        return self.cv_bridge.cv2_to_imgmsg(*args, **kwargs)

    def msg_to_cv2(self, *args, **kwargs) -> np.array:
        return self.cv_bridge.imgmsg_to_cv2(*args, **kwargs)

    def _publish(self):
        raise NotImplementedError


class SpotLocalRawImagesPublisher(SpotImagePublisher):
    name = "spot_local_raw_images_publisher"
    publisher_topics = [
        rt.HEAD_DEPTH,
        rt.HAND_DEPTH,
        rt.HAND_RGB,
        rt.HAND_DEPTH_UNSCALED,
    ]
    sources = [
        Cam.FRONTRIGHT_DEPTH,
        Cam.FRONTLEFT_DEPTH,
        Cam.HAND_COLOR,
        Cam.HAND_DEPTH_IN_HAND_COLOR_FRAME,
    ]

    def __init__(self, spot):
        super().__init__()
        self.spot = spot

    def _publish(self):
        image_responses = self.spot.get_image_responses(
            self.sources[:2], quality=100, await_the_resp=False
        )
        hand_image_responses = self.spot.get_hand_image()
        image_responses = image_responses.result()
        image_responses.extend(hand_image_responses)

        imgs_list = [image_response_to_cv2(r) for r in image_responses]
        imgs = {k: v for k, v in zip(self.sources, imgs_list)}

        head_depth = np.hstack([imgs[Cam.FRONTRIGHT_DEPTH], imgs[Cam.FRONTLEFT_DEPTH]])

        head_depth = self._scale_depth(head_depth, head_depth=True)
        hand_depth = self._scale_depth(imgs[Cam.HAND_DEPTH_IN_HAND_COLOR_FRAME])
        hand_depth_unscaled = imgs[Cam.HAND_DEPTH_IN_HAND_COLOR_FRAME]
        hand_rgb = imgs[Cam.HAND_COLOR]

        msgs = self.imgs_to_msgs(head_depth, hand_depth, hand_rgb, hand_depth_unscaled)

        for topic, msg in zip(self.pubs.keys(), msgs):
            self.pubs[topic].publish(msg)

    def imgs_to_msgs(self, head_depth, hand_depth, hand_rgb, hand_depth_unscaled):
        head_depth_msg = self.cv2_to_msg(head_depth, "mono8")
        hand_depth_msg = self.cv2_to_msg(hand_depth, "mono8")
        hand_rgb_msg = self.cv2_to_msg(hand_rgb, "bgr8")
        hand_depth_unscaled_msg = self.cv2_to_msg(hand_depth_unscaled, "mono16")

        timestamp = rospy.Time.now()
        head_depth_msg.header = Header(stamp=timestamp)
        hand_depth_msg.header = Header(stamp=timestamp)
        hand_rgb_msg.header = Header(stamp=timestamp)
        hand_depth_unscaled_msg.header = Header(stamp=timestamp)

        return head_depth_msg, hand_depth_msg, hand_rgb_msg, hand_depth_unscaled_msg

    @staticmethod
    def _scale_depth(img, head_depth=False):
        img = scale_depth_img(
            img, max_depth=MAX_DEPTH if head_depth else MAX_HAND_DEPTH
        )
        return np.uint8(img * 255.0)


class SpotLocalCompressedImagesPublisher(SpotLocalRawImagesPublisher):
    name = "spot_local_compressed_images_publisher"
    publisher_topics = [rt.COMPRESSED_IMAGES]
    publish_msg_type = ByteMultiArray

    def imgs_to_msgs(self, head_depth, hand_depth, hand_rgb):
        head_depth_bytes = blosc.pack_array(
            head_depth, cname="zstd", clevel=3, shuffle=blosc.NOSHUFFLE
        )
        hand_depth_bytes = blosc.pack_array(
            hand_depth, cname="zstd", clevel=3, shuffle=blosc.NOSHUFFLE
        )
        hand_rgb_bytes = np.array(cv2.imencode(".jpg", hand_rgb)[1])
        hand_rgb_bytes = (hand_rgb_bytes.astype(int) - 128).astype(np.int8)
        topic2bytes = {
            rt.HEAD_DEPTH: head_depth_bytes,
            rt.HAND_DEPTH: hand_depth_bytes,
            rt.HAND_RGB: hand_rgb_bytes,
        }
        topic2details = {
            topic: {
                "dims": MultiArrayDimension(label=topic, size=len(img_bytes)),
                "bytes": img_bytes,
            }
            for topic, img_bytes in topic2bytes.items()
        }

        depth_bytes = b""
        rgb_bytes, depth_dims, rgb_dims = [], [], []
        for topic, details in topic2details.items():
            if "depth" in topic:
                depth_bytes += details["bytes"]
                depth_dims.append(details["dims"])
            else:
                rgb_bytes.append(details["bytes"])
                rgb_dims.append(details["dims"])
        depth_bytes = np.frombuffer(depth_bytes, dtype=np.uint8).astype(int) - 128
        bytes_data = np.concatenate([depth_bytes, *rgb_bytes])
        timestamp = str(time.time())
        timestamp_dim = MultiArrayDimension(label=timestamp, size=0)
        dims = depth_dims + rgb_dims + [timestamp_dim]
        msg = ByteMultiArray(layout=MultiArrayLayout(dim=dims), data=bytes_data)
        return [msg]


class SpotProcessedImagesPublisher(SpotImagePublisher):
    subscriber_topic = ""
    subscriber_msg_type = Image

    def __init__(self):
        super().__init__()
        self.img_msg = None
        rospy.Subscriber(
            self.subscriber_topic, self.subscriber_msg_type, self.cb, queue_size=1
        )
        rospy.loginfo(f"[{self.name}]: is waiting for images...")
        while self.img_msg is None and self.subscriber_topic is not None:
            pass
        rospy.loginfo(f"[{self.name}]: has received images!")
        self.updated = True

    def publish(self):
        if not self.updated:
            return
        super().publish()
        self.updated = False

    def cb(self, msg: Image):
        self.img_msg = msg
        self.updated = True


class SpotDecompressingRawImagesPublisher(SpotProcessedImagesPublisher):
    name = "spot_decompressing_raw_images_publisher"
    publisher_topics = [rt.HEAD_DEPTH, rt.HAND_DEPTH, rt.HAND_RGB]
    subscriber_topic = rt.COMPRESSED_IMAGES
    subscriber_msg_type = ByteMultiArray

    def _publish(self):
        if self.img_msg is None:
            return
        img_msg = deepcopy(self.img_msg)

        py_timestamp = float(img_msg.layout.dim[-1].label)
        latency = time.time() - py_timestamp
        latency_msg = f"[{self.name}]: Latency is {latency:.2f} sec"
        if latency < 0.5:
            rospy.loginfo(latency_msg + ".")
        else:
            rospy.logwarn(latency_msg + "!")
        timestamp = rospy.Time.from_sec(py_timestamp)

        byte_data = (np.array(img_msg.data) + 128).astype(np.uint8)
        size_and_labels = [
            (int(dim.size), str(dim.label)) for dim in img_msg.layout.dim
        ]
        start = 0
        imgs = {}
        for size, label in size_and_labels:
            end = start + size
            if "depth" in label:
                img = blosc.unpack_array(byte_data[start:end].tobytes())
                imgs[label] = img
            elif "rgb" in label:
                rgb_bytes = byte_data[start:end]
                img = cv2.imdecode(rgb_bytes, cv2.IMREAD_COLOR)
            else:  # timestamp
                continue
            imgs[label] = img
            start += size

        head_depth_msg = self.cv2_to_msg(imgs[rt.HEAD_DEPTH], "mono8")
        hand_depth_msg = self.cv2_to_msg(imgs[rt.HAND_DEPTH], "mono8")
        hand_rgb_msg = self.cv2_to_msg(imgs[rt.HAND_RGB], "bgr8")

        head_depth_msg.header = Header(stamp=timestamp)
        hand_depth_msg.header = Header(stamp=timestamp)
        hand_rgb_msg.header = Header(stamp=timestamp)

        self.pubs[rt.HEAD_DEPTH].publish(head_depth_msg)
        self.pubs[rt.HAND_DEPTH].publish(hand_depth_msg)
        self.pubs[rt.HAND_RGB].publish(hand_rgb_msg)


class SpotFilteredDepthImagesPublisher(SpotProcessedImagesPublisher):
    max_depth = 0.0
    filtered_depth_topic = ""

    def _publish(self):
        depth = self.msg_to_cv2(self.img_msg)
        filtered_depth = filter_depth(depth, max_depth=self.max_depth)
        img_msg = self.cv_bridge.cv2_to_imgmsg(filtered_depth, "mono8")
        img_msg.header = self.img_msg.header
        self.pubs[self.publisher_topics[0]].publish(img_msg)


class SpotFilteredHeadDepthImagesPublisher(SpotFilteredDepthImagesPublisher):
    name = "spot_filtered_head_depth_images_publisher"
    subscriber_topic = rt.HEAD_DEPTH
    max_depth = MAX_DEPTH
    publisher_topics = [rt.FILTERED_HEAD_DEPTH]


class SpotFilteredHandDepthImagesPublisher(SpotFilteredDepthImagesPublisher):
    name = "spot_filtered_hand_depth_images_publisher"
    subscriber_topic = rt.HAND_DEPTH
    max_depth = MAX_HAND_DEPTH
    publisher_topics = [rt.FILTERED_HAND_DEPTH]


class SpotBoundingBoxPublisher(SpotProcessedImagesPublisher):

    # TODO: We eventually want to change this name as well as the publisher topic
    name = "spot_mrcnn_publisher"
    subscriber_topic = rt.HAND_RGB
    publisher_topics = [rt.MASK_RCNN_VIZ_TOPIC]

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.detection_topic = rt.DETECTIONS_TOPIC

        self.config = config = construct_config()
        self.image_scale = config.IMAGE_SCALE
        self.deblur_gan = get_deblurgan_model(config)
        self.grayscale = self.config.GRAYSCALE_MASK_RCNN

        self.pubs[self.detection_topic] = rospy.Publisher(
            self.detection_topic, String, queue_size=1, tcp_nodelay=True
        )
        self.viz_topic = rt.MASK_RCNN_VIZ_TOPIC
        self._reset_tracking_params()

    def preprocess_image(self, img):
        if self.image_scale != 1.0:
            img = cv2.resize(
                img,
                (0, 0),
                fx=self.image_scale,
                fy=self.image_scale,
                interpolation=cv2.INTER_AREA,
            )

        if self.deblur_gan is not None:
            img = self.deblur_gan(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return img

    def _reset_tracking_params(self):
        self._tracking_first_time = (
            True  # flag to indicate if it is the first time tracking
        )
        self._tracking_images = []  # cache the tracking images
        self._tracking_bboxs = []  # cache the tracking bboxs

    def _publish(self):
        stopwatch = Stopwatch()
        header = self.img_msg.header
        timestamp = header.stamp

        # size of hand_rgb: (480, 640, 3)
        # size of hand_rgb_preprocessed: (336, 448, 3)

        hand_rgb = self.msg_to_cv2(self.img_msg)
        hand_rgb_preprocessed = self.preprocess_image(hand_rgb)

        # Copy the hand_rgb_preprocessed
        tracking_rgb = hand_rgb_preprocessed.copy().astype("uint8")

        # Fetch the tracking flag
        enable_tracking = rospy.get_param("enable_tracking", False)

        # Cache the vis image
        viz_img = hand_rgb_preprocessed

        if enable_tracking:
            # Start tracking using video segmentation model
            images = np.expand_dims(
                tracking_rgb, axis=0
            )  # The size should (1, H, W, C)
            self._tracking_images.append(images)

            if len(self._tracking_images) == 1 and self._tracking_first_time:
                # To see if there is a bounding box in the first frame
                self._tracking_first_time = False
                # Get the first detection
                bbox_data, viz_img = self.model.inference(
                    hand_rgb_preprocessed, timestamp, stopwatch
                )
                _, detection_str = bbox_data.split("|")
                if detection_str == "None":
                    # Do not see the object yet, so restart the detection
                    self._reset_tracking_params()
                else:
                    # See the object at the first frame
                    cur_bbox_str = detection_str.split(",")[2:]
                    cur_bbox = [int(v) for v in cur_bbox_str]
                    self._tracking_bboxs.append(cur_bbox)
            elif len(self._tracking_images) >= 2:
                # This means that there is one object being detected in the previous frame (first frame)
                input_images = np.concatenate(self._tracking_images, axis=0)
                # Get the previous anchor
                bbox = self._tracking_bboxs[-1]
                cur_bbox = tracking_with_socket(input_images, bbox)
                if cur_bbox is None:
                    self._tracking_bboxs.append(None)
                    bbox_data = f"{str(timestamp)}|None"
                else:
                    cur_bbox = [cur_bbox[1], cur_bbox[0], cur_bbox[3], cur_bbox[2]]
                    self._tracking_bboxs.append(cur_bbox)
                    object_label = rospy.get_param("object_target")
                    bbox_data = f"{str(timestamp)}|{object_label},{1.0}," + ",".join(
                        [str(v) for v in cur_bbox]
                    )
                    # Apply the red bounding box to showcase tracking on the image
                    viz_img = cv2.rectangle(
                        viz_img, cur_bbox[:2], cur_bbox[2:], (0, 0, 255), 5
                    )

            if len(self._tracking_images) >= 2:
                # Prune the data
                # Find the frame that has tracking
                suc_frame = []
                for i, v in enumerate(self._tracking_bboxs):
                    if v is not None:
                        suc_frame.append(i)

                # Only keep the latest frame that has tracking to keep
                # buffer size 2
                self._tracking_images = [self._tracking_images[suc_frame[-1]]]
                self._tracking_bboxs = [self._tracking_bboxs[suc_frame[-1]]]
        else:
            # Normal detection mode using detection model
            bbox_data, viz_img = self.model.inference(
                hand_rgb_preprocessed, timestamp, stopwatch
            )
            time_stamp, detection_str = bbox_data.split("|")
            self._reset_tracking_params()

        # publish data
        self.publish_bbox_data(bbox_data)
        self.publish_viz_img(viz_img, header)

        # stopwatch.print_stats()

    def publish_bbox_data(self, bbox_data):
        self.pubs[self.detection_topic].publish(bbox_data)

    def publish_viz_img(self, viz_img, header):
        viz_img_msg = self.cv2_to_msg(viz_img, "bgr8")
        viz_img_msg.header = header
        self.pubs[self.viz_topic].publish(viz_img_msg)


class SpotOpenVocObjectDetectorPublisher(SpotImagePublisher):

    name = "spot_open_voc_object_detector_publisher"
    # TODO: spot-sim2real: this is a hack since it publishes images in SpotProcessedImagesPublisher
    publisher_topics = [rt.MASK_RCNN_VIZ_TOPIC]

    def __init__(self, model, spot):
        super().__init__()
        self.model = model
        self.spot = spot
        self.detection_topic = rt.OPEN_VOC_OBJECT_DETECTOR_TOPIC

        self.config = config = construct_config()
        self.image_scale = config.IMAGE_SCALE
        self.deblur_gan = get_deblurgan_model(config)
        self.grayscale = self.config.GRAYSCALE_MASK_RCNN

        self.pubs[self.detection_topic] = rospy.Publisher(
            self.detection_topic, String, queue_size=1, tcp_nodelay=True
        )

        # For the depth images
        self.img_msg_depth = None
        # rospy.Subscriber(rt.HAND_DEPTH_UNSCALED, Image, self.depth_cb, queue_size=1)
        rospy.loginfo(f"[{self.name}]: is waiting for images...")

        self.viz_topic = rt.MASK_RCNN_VIZ_TOPIC

        # while self.img_msg_depth is None:
        #     pass
        rospy.loginfo(f"[{self.name}]: has received images!")

    def depth_cb(self, msg: Image):
        self.img_msg_depth = msg

    def preprocess_image(self, img):
        if self.image_scale != 1.0:
            img = cv2.resize(
                img,
                (0, 0),
                fx=self.image_scale,
                fy=self.image_scale,
                interpolation=cv2.INTER_AREA,
            )

        if self.deblur_gan is not None:
            img = self.deblur_gan(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return img

    def convert_2d_pixel_to_3d(
        self, pixel_uv, depth_raw, camera_intrinsics, patch_size=10
    ):
        Z = float(
            sample_patch_around_point(
                int(pixel_uv[0]), int(pixel_uv[1]), depth_raw, patch_size
            )
            * 1.0
        )
        if np.isnan(Z):
            print(f"Affordance Prediction : Z is NaN = {Z}")
            return None
        point_in_3d = get_3d_point(camera_intrinsics, pixel_uv, Z)
        return point_in_3d

    def convert_2d_pixel_to_3d(self, pixel_uv, depth_raw, camera_intrinsics, patch_size=10):
        Z = float(sample_patch_around_point(int(pixel_uv[0]), int(pixel_uv[1]), depth_raw, patch_size) * 1.0)
        if np.isnan(Z):
            print(f"Affordance Prediction : Z is NaN = {Z}")
            return None
        elif Z > MAX_HAND_DEPTH:
            print(f"Affordance Prediction : Z is out of bounds = {Z}")
            return None

        point_in_3d = get_3d_point(camera_intrinsics, pixel_uv, Z)
        return point_in_3d

    def _publish(self):
        stopwatch = Stopwatch()
        header = Header(stamp=rospy.Time.now())  # self.img_msg.header
        timestamp = header.stamp

        # Get camera pose of view and the location of the robot
        # These two should be fast and limited delay
        imgs = self.spot.get_hand_image(
            force_get_gripper=True
        )  # for getting gripper intrinsics
        # Get the camera intrinsics
        cam_intrinsics = imgs[0].source.pinhole.intrinsics
        hand_rgb, arm_depth = [image_response_to_cv2(img) for img in imgs]

        # Get the vision to hand
        try:
            body_T_hand: mn.Matrix4 = self.spot.get_magnum_Matrix4_spot_a_T_b(
                "body", "hand_color_image_sensor", imgs[0].shot.transforms_snapshot
            )
            curr_x, curr_y, curr_yaw = self.spot.get_xy_yaw()
            home_T_body: mn.Matrix4 = mn.Matrix4.from_(
                mn.Matrix4.rotation_z(mn.Rad(curr_yaw)).rotation(),
                mn.Vector3(curr_x, curr_y, 0.5),
            )
            home_T_hand = home_T_body @ body_T_hand
        except Exception as e:
            print("ee:", imgs[0].shot.transforms_snapshot is None)
            print(f"Cannot get Spot T due to {e}. Online detection might be off")
            return

        # Internal model
        hand_rgb_preprocessed = self.preprocess_image(hand_rgb)
        new_detection, viz_img = self.model.inference(
            hand_rgb_preprocessed, timestamp, stopwatch
        )
        # Split the detection
        timestamp, new_detections = new_detection.split("|")
        new_detections = new_detections.split(";")
        object_info = []
        for det_i, detection_str in enumerate(new_detections):
            if detection_str == "None":
                continue
            class_label, score, x1, y1, x2, y2 = detection_str.split(",")
            # Compute the center pixel
            x1, y1, x2, y2 = [
                int(float(i) / self.image_scale) for i in [x1, y1, x2, y2]
            ]
            pixel_x = int(np.mean([x1, x2]))
            pixel_y = int(np.mean([y1, y2]))
            try:
                depth_raw = arm_depth / 1000.0
                point_in_gripper = self.convert_2d_pixel_to_3d(
                    [pixel_x, pixel_y], depth_raw, cam_intrinsics
                )
                if point_in_gripper is None:
                    print(f"Affordance Point is NaN for {class_label}, skipping")
                    continue
                # left top & bottom right are x1, y1 & x2, y2 are endpoints of detected bbox we convert those to HOME frame
                # & then use these in hab-llm to calculate iou in global space
                left_top_in_3d = self.convert_2d_pixel_to_3d(
                    [x1, y1], depth_raw, cam_intrinsics
                )
                right_bottom_in_3d = self.convert_2d_pixel_to_3d(
                    [x2, y2], depth_raw, cam_intrinsics
                )

                if np.isnan(point_in_gripper).any():
                    continue

                point_in_global_3d = np.array(
                    home_T_hand.transform_point(mn.Vector3(*point_in_gripper))
                )
                linear_dist = np.linalg.norm(
                    point_in_global_3d[:2] - np.array([curr_x, curr_y])
                )
                if linear_dist > 3.0:
                    print(f"Detected object quite out of range {linear_dist}")
                    continue
                points_in_global_3d = [
                    np.array(home_T_hand.transform_point(mn.Vector3(*point_in_3d)))[
                        :2
                    ].tolist()
                    for point_in_3d in [left_top_in_3d, right_bottom_in_3d]
                ]
                points_in_global_3d_str = ",".join(
                    [f"{p[0]},{p[1]}" for p in points_in_global_3d]
                )
                object_info.append(
                    f"{class_label},{point_in_global_3d[0]},{point_in_global_3d[1]},{point_in_global_3d[2]},{score},{points_in_global_3d_str}"
                )
                print(
                    f"{class_label}: {point_in_global_3d} {x1} {y1} {x2} {y2} {pixel_x} {pixel_y}"
                )
            except Exception as e:
                print(f"Issue of predicting location for {class_label} : {e}")
                continue

        # publish data
        self.publish_new_detection(";".join(object_info))
        self.publish_viz_img(viz_img, header)

    def publish_new_detection(self, new_object):
        self.pubs[self.detection_topic].publish(new_object)

    def publish_viz_img(self, viz_img, header):
        viz_img_msg = self.cv2_to_msg(viz_img, "bgr8")
        viz_img_msg.header = header
        self.pubs[self.viz_topic].publish(viz_img_msg)


class OWLVITModel:
    def __init__(self, score_threshold=0.1, show_img=False):
        self.config = config = construct_config()
        self.owlvit = OwlVit([["ball"]], score_threshold, show_img, 2)
        self.image_scale = config.IMAGE_SCALE
        rospy.loginfo("[OWLVIT]: Models loaded.")

    def inference(self, hand_rgb, timestamp, stopwatch):
        # print("Running inference on Owlvit")
        params = rospy.get_param("/object_target").split(",")
        self.owlvit.update_label([params])
        bbox_xy, viz_img = self.owlvit.run_inference_and_return_img(hand_rgb)

        if bbox_xy is not None and bbox_xy != []:
            detections = []
            for detection in bbox_xy:
                str_det = f'{detection[0]},{detection[1]},{",".join([str(i) for i in detection[2]])}'
                detections.append(str_det)
            bbox_xy_string = ";".join(detections)
        else:
            bbox_xy_string = "None"
        detections_str = f"{str(timestamp)}|{bbox_xy_string}"

        return detections_str, viz_img


class OWLVITModelMultiClasses(OWLVITModel):
    def inference(self, hand_rgb, timestamp, stopwatch):
        # print("Running inference on OWLVITModelMultiClasses")
        # Add new classes here to the model
        # We decide to hardcode these classes first as this will be more robust
        # and gives us the way to control the detection
        # multi_classes = [["ball", "cup", "table", "cabinet", "chair", "sofa"]]
        multi_classes = rospy.get_param("/multi_class_object_target").split(",")
        multi_classes = [str(class_name).strip() for class_name in multi_classes]
        self.owlvit.update_label([multi_classes])
        # TODO: spot-sim2real: right now for each class, we only return the most confident detection
        bbox_xy, viz_img = self.owlvit.run_inference_and_return_img(
            hand_rgb, multi_objects_per_label=True
        )
        # bbox_xy is a list of [label_without_prefix, target_scores[label], [x1, y1, x2, y2]]
        if bbox_xy is not None and bbox_xy != []:
            detections = []
            for detection in bbox_xy:
                str_det = f'{detection[0]},{detection[1]},{",".join([str(i) for i in detection[2]])}'
                detections.append(str_det)
            bbox_xy_string = ";".join(detections)
        else:
            bbox_xy_string = "None"
        detections_str = f"{str(timestamp)}|{bbox_xy_string}"

        return detections_str, viz_img


class MRCNNModel:
    def __init__(self):
        self.config = config = construct_config()
        self.mrcnn = get_mrcnn_model(config)
        rospy.loginfo("[MRCNN]: Models loaded.")

    def inference(self, hand_rgb, timestamp, stopwatch):
        img = hand_rgb
        pred = self.mrcnn.inference(img)
        if stopwatch is not None:
            stopwatch.record("mrcnn_secs")
        detections_str = f"{int(timestamp)}|{pred2string(pred)}"
        viz_img = self.mrcnn.visualize_inference(img, pred)
        return detections_str, viz_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter-head-depth", action="store_true")
    parser.add_argument("--filter-hand-depth", action="store_true")
    parser.add_argument("--decompress", action="store_true")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--owlvit", action="store_true")
    parser.add_argument("--open-voc", action="store_true")
    parser.add_argument("--mrcnn", action="store_true")
    parser.add_argument("--core", action="store_true", help="running on the Core")
    parser.add_argument("--listen", action="store_true", help="listening to Core")
    parser.add_argument(
        "--local", action="store_true", help="fully local robot connection"
    )
    parser.add_argument(
        "--bounding_box_detector",
        choices=["owlvit", "mrcnn"],
        default="owlvit",
        help="bounding box detector model to use (owlvit or maskrcnn)",
    )

    args = parser.parse_args()
    # assert (
    #    len([i[1] for i in args._get_kwargs() if i[1]]) == 1
    # ), "One and only one arg must be provided."

    filter_head_depth = args.filter_head_depth
    filter_hand_depth = args.filter_hand_depth
    decompress = args.decompress
    raw = args.raw
    compress = args.compress
    core = args.core
    listen = args.listen
    local = args.local
    bounding_box_detector = args.bounding_box_detector
    mrcnn = args.mrcnn
    owlvit = args.owlvit
    open_voc = args.open_voc

    node = None  # type: Any
    model = None  # type: Any
    if filter_head_depth:
        node = SpotFilteredHeadDepthImagesPublisher()
    elif filter_hand_depth:
        node = SpotFilteredHandDepthImagesPublisher()
    elif mrcnn:
        model = MRCNNModel()
        node = SpotBoundingBoxPublisher(model)
    elif owlvit:
        # TODO dynamic label
        rospy.set_param("object_target", "ball")
        rospy.set_param("enable_tracking", False)
        model = OWLVITModel()
        node = SpotBoundingBoxPublisher(model)
    elif open_voc:
        # Add open voc object detector here
        spot = Spot("SpotOpenVocObjectDetectorPublisher")
        # ball,donut plush toy,can of food,bottle,caterpillar plush toy,bulldozer toy ca
        rospy.set_param(
            "multi_class_object_target",
            "can of food,creamer bottle",
        )
        model = OWLVITModelMultiClasses()
        node = SpotOpenVocObjectDetectorPublisher(model, spot)
    elif decompress:
        node = SpotDecompressingRawImagesPublisher()
    elif raw or compress:
        name = "LocalRawImagesPublisher" if raw else "LocalCompressedImagesPublisher"
        spot = Spot(name)
        if raw:
            node = SpotLocalRawImagesPublisher(spot)
        else:
            node = SpotLocalCompressedImagesPublisher(spot)
    else:
        assert core or listen or local, "This should be impossible."

    if core or listen or local:
        if core:
            flags = ["--compress"]
        else:
            flags = [
                "--filter-head-depth",
                "--filter-hand-depth",
                f"--{bounding_box_detector}",
            ]
            if listen:
                flags.append("--decompress")
            elif local:
                flags.append("--raw")
                flags.append("--open-voc")
            else:
                raise RuntimeError("This should be impossible.")
        cmds = [f"python {osp.abspath(__file__)} {flag}" for flag in flags]

        processes = [subprocess.Popen(cmd, shell=True) for cmd in cmds]
        try:
            while all([p.poll() is None for p in processes]):
                pass
        finally:
            for p in processes:
                try:
                    p.kill()
                except Exception:
                    pass
    else:
        while not rospy.is_shutdown():
            node.publish()
