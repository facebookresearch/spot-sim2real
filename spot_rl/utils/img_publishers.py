import argparse
import os.path as osp
import subprocess
import time
from copy import deepcopy

import blosc
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
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

from spot_rl.utils.depth_map_utils import filter_depth

try:
    from spot_rl.utils.mask_rcnn_utils import (
        generate_mrcnn_detections,
        get_deblurgan_model,
        get_mrcnn_model,
        pred2string,
    )
except ModuleNotFoundError:
    pass
from spot_rl.utils.stopwatch import Stopwatch
from spot_rl.utils.utils import construct_config
from spot_rl.utils.utils import ros_topics as rt

# Detection using owlvit model
from spot_rl.utils.owlvit_utils import OwlVit

MAX_PUBLISH_FREQ = 20
MAX_DEPTH = 3.5
MAX_HAND_DEPTH = 1.7


class SpotImagePublisher:
    name = ""
    publisher_topics = []
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
        st = time.time()
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
    publisher_topics = [rt.HEAD_DEPTH, rt.HAND_DEPTH, rt.HAND_RGB]
    sources = [
        Cam.FRONTRIGHT_DEPTH,
        Cam.FRONTLEFT_DEPTH,
        Cam.HAND_DEPTH_IN_HAND_COLOR_FRAME,
        Cam.HAND_COLOR,
    ]

    def __init__(self, spot):
        super().__init__()
        self.spot = spot

    def _publish(self):
        image_responses = self.spot.get_image_responses(self.sources, quality=100)
        imgs_list = [image_response_to_cv2(r) for r in image_responses]
        imgs = {k: v for k, v in zip(self.sources, imgs_list)}

        head_depth = np.hstack([imgs[Cam.FRONTRIGHT_DEPTH], imgs[Cam.FRONTLEFT_DEPTH]])

        head_depth = self._scale_depth(head_depth, head_depth=True)
        hand_depth = self._scale_depth(imgs[Cam.HAND_DEPTH_IN_HAND_COLOR_FRAME])
        hand_rgb = imgs[Cam.HAND_COLOR]

        msgs = self.imgs_to_msgs(head_depth, hand_depth, hand_rgb)

        for topic, msg in zip(self.pubs.keys(), msgs):
            self.pubs[topic].publish(msg)

    def imgs_to_msgs(self, head_depth, hand_depth, hand_rgb):
        head_depth_msg = self.cv2_to_msg(head_depth, "mono8")
        hand_depth_msg = self.cv2_to_msg(hand_depth, "mono8")
        hand_rgb_msg = self.cv2_to_msg(hand_rgb, "bgr8")

        timestamp = rospy.Time.now()
        head_depth_msg.header = Header(stamp=timestamp)
        hand_depth_msg.header = Header(stamp=timestamp)
        hand_rgb_msg.header = Header(stamp=timestamp)

        return head_depth_msg, hand_depth_msg, hand_rgb_msg

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
        while self.img_msg is None:
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


class SpotMRCNNPublisher(SpotProcessedImagesPublisher):
    name = "spot_mrcnn_publisher"
    subscriber_topic = rt.HAND_RGB
    publisher_topics = [rt.MASK_RCNN_VIZ_TOPIC]

    def __init__(self):
        self.config = config = construct_config()
        self.mrcnn = get_mrcnn_model(config)
        self.deblur_gan = get_deblurgan_model(config)
        self.image_scale = config.IMAGE_SCALE
        rospy.loginfo(f"[{self.name}]: Models loaded.")
        super().__init__()
        self.pubs[rt.DETECTIONS_TOPIC] = rospy.Publisher(
            rt.DETECTIONS_TOPIC, String, queue_size=1, tcp_nodelay=True
        )

    def _publish(self):
        stopwatch = Stopwatch()

        # Publish the Mask RCNN detections
        header = self.img_msg.header
        timestamp = header.stamp
        hand_rgb = self.msg_to_cv2(self.img_msg)
        pred, viz_img = generate_mrcnn_detections(
            hand_rgb,
            scale=self.image_scale,
            mrcnn=self.mrcnn,
            grayscale=self.config.GRAYSCALE_MASK_RCNN,
            deblurgan=self.deblur_gan,
            return_img=True,
            stopwatch=stopwatch,
        )
        detections_str = f"{int(timestamp.nsecs)}|{pred2string(pred)}"

        viz_img = self.mrcnn.visualize_inference(viz_img, pred)
        if not detections_str.endswith("None"):
            print(detections_str)
        viz_img_msg = self.cv2_to_msg(viz_img)
        viz_img_msg.header = header
        stopwatch.record("vis_secs")

        stopwatch.print_stats()

        self.pubs[rt.DETECTIONS_TOPIC].publish(detections_str)
        self.pubs[rt.MASK_RCNN_VIZ_TOPIC].publish(viz_img_msg)
        print("ssss")


class SpotOWLVITPublisher(SpotProcessedImagesPublisher):
    name = "spot_owlvit_publisher"
    subscriber_topic = rt.HAND_RGB
    publisher_topics = [rt.OWLVIT_VIZ_TOPIC]

    def __init__(self, owlvit_label):
        self.config = config = construct_config()
        self.owlvit = OwlVit([[owlvit_label]], 0.05, True)
        self.image_scale = config.IMAGE_SCALE
        rospy.loginfo(f"[{self.name}]: Models loaded.")
        super().__init__()
        self.pubs[rt.OWLVIT_DETECTIONS_TOPIC] = rospy.Publisher(
            rt.OWLVIT_DETECTIONS_TOPIC, String, queue_size=1, tcp_nodelay=True
        )

    def _publish(self):
        stopwatch = Stopwatch()

        # Publish the OWLVIT detections
        header = self.img_msg.header
        timestamp = header.stamp
        hand_rgb = self.msg_to_cv2(self.img_msg)

        # Detect the image from here
        #self.owlvit.update_label([["ball"]])
        print(self.owlvit.labels)
        bbox_xy, viz_img = self.owlvit.run_inference_and_return_img(hand_rgb)
        print(bbox_xy)


        if bbox_xy is not None:
            bbox_xy_string = str(bbox_xy[0])+","+str(bbox_xy[1])+','+str(bbox_xy[2])+','+str(bbox_xy[3])
        else:
            bbox_xy_string = "None"
        detections_str = f"{int(timestamp.nsecs)}|{bbox_xy_string}"

        # We might need to do this for owlvit
        #viz_img = self.mrcnn.visualize_inference(viz_img, pred)
        if not detections_str.endswith("None"):
            print(detections_str)
        viz_img_msg = self.cv2_to_msg(viz_img)
        viz_img_msg.header = header
        stopwatch.record("vis_secs")

        stopwatch.print_stats()

        self.pubs[rt.OWLVIT_DETECTIONS_TOPIC].publish(detections_str)
        self.pubs[rt.OWLVIT_VIZ_TOPIC].publish(viz_img_msg)


class SpotOWLVITPublisher(SpotProcessedImagesPublisher):
    name = "spot_owlvit_publisher"
    subscriber_topic = rt.HAND_RGB
    publisher_topics = [rt.OWLVIT_VIZ_TOPIC]

    def __init__(self):
        self.config = config = construct_config()
        self.owlvit = OwlVit([['lion plush', 'penguin plush', 'teddy bear', 'bear plush', 'caterpilar plush', 'ball plush', 'rubiks cube']], 0.1, False)
        self.deblur_gan = get_deblurgan_model(config)
        self.image_scale = config.IMAGE_SCALE
        rospy.loginfo(f"[{self.name}]: Models loaded.")
        super().__init__()
        self.pubs[rt.OWLVIT_DETECTIONS_TOPIC] = rospy.Publisher(
            rt.OWLVIT_DETECTIONS_TOPIC, String, queue_size=1, tcp_nodelay=True
        )

    def _publish(self):
        stopwatch = Stopwatch()

        # Publish the OWLVIT detections
        header = self.img_msg.header
        timestamp = header.stamp
        hand_rgb = self.msg_to_cv2(self.img_msg)

        # Detect the image from here
        self.owlvit.update_label([["ball"]])
        bbox_xy = self.owlvit.run_inference(img)

        if bbox_xy is not None:
            bbox_xy_string = str(bbox_xy[0])+","+str(bbox_xy[1])+','+str(bbox_xy[3])+','+str(bbox_xy[4])
        else:
            bbox_xy_string = "None"
        detections_str = f"{int(timestamp.nsecs)}|{bbox_xy_string}"

        viz_img = self.mrcnn.visualize_inference(viz_img, pred)
        if not detections_str.endswith("None"):
            print(detections_str)
        viz_img_msg = self.cv2_to_msg(viz_img)
        viz_img_msg.header = header
        stopwatch.record("vis_secs")

        stopwatch.print_stats()

        self.pubs[rt.OWLVIT_DETECTIONS_TOPIC].publish(detections_str)
        self.pubs[rt.OWLVIT_VIZ_TOPIC].publish(viz_img_msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter-head-depth", action="store_true")
    parser.add_argument("--filter-hand-depth", action="store_true")
    parser.add_argument("--mrcnn", action="store_true")
    parser.add_argument("--owlvit", action="store_true")
    parser.add_argument("--decompress", action="store_true")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--core", action="store_true", help="running on the Core")
    parser.add_argument("--listen", action="store_true", help="listening to Core")
    parser.add_argument(
        "--local", action="store_true", help="fully local robot connection"
    )
    parser.add_argument("--owlvit_label")
    args = parser.parse_args()
    print(args)
    #assert (
    #    len([i[1] for i in args._get_kwargs() if i[1]]) == 1
    #), "One and only one arg must be provided."

    filter_head_depth = args.filter_head_depth
    filter_hand_depth = args.filter_hand_depth
    mrcnn = args.mrcnn
    owlvit = args.owlvit
    decompress = args.decompress
    raw = args.raw
    compress = args.compress
    core = args.core
    listen = args.listen
    local = args.local
    owlvit_label = args.owlvit_label
    #owlvit_label = 'paper roll'

    node = None
    if filter_head_depth:
        node = SpotFilteredHeadDepthImagesPublisher()
    elif filter_hand_depth:
        node = SpotFilteredHandDepthImagesPublisher()
    elif mrcnn:
        node = SpotMRCNNPublisher()
    elif owlvit:
        node = SpotOWLVITPublisher(owlvit_label)
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
            #flags = ["--filter-head-depth", "--filter-hand-depth", "--mrcnn"]
            flags = ["--filter-head-depth", "--filter-hand-depth", f'--owlvit --owlvit_label "{owlvit_label}"', "--mrcnn"]
            #flags = ["--filter-head-depth", "--filter-hand-depth", f'--owlvit', "--mrcnn"]
            if listen:
                flags.append("--decompress")
            elif local:
                flags.append("--raw")
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
                except:
                    pass
    else:
        while not rospy.is_shutdown():
            node.publish()
