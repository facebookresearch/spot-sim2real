# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json
import time
from functools import partial

import magnum as mn
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from spot_rl.utils.utils import ros_topics as rt
from std_msgs.msg import Float32MultiArray, String

IMG_TOPICS = [
    rt.MASK_RCNN_VIZ_TOPIC,
    rt.HEAD_DEPTH,
    rt.HAND_DEPTH,
    rt.HAND_RGB,
    rt.FILTERED_HEAD_DEPTH,
    rt.FILTERED_HAND_DEPTH,
]
NO_RAW_IMG_TOPICS = [
    rt.MASK_RCNN_VIZ_TOPIC,
    rt.HAND_RGB,
    rt.FILTERED_HEAD_DEPTH,
    rt.FILTERED_HAND_DEPTH,
]


class SpotRobotSubscriberMixin:
    node_name = "SpotRobotDataSubscriber"
    no_raw = False
    proprioception = True

    def __init__(self, spot=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rospy.init_node(self.node_name, disable_signals=True)
        self.cv_bridge = CvBridge()

        subscriptions = NO_RAW_IMG_TOPICS if self.no_raw else IMG_TOPICS

        # Maps a topic name to the latest msg from it
        self.msgs = {topic: None for topic in subscriptions}
        self.updated = {topic: False for topic in subscriptions}

        for img_topic in subscriptions:
            rospy.Subscriber(
                img_topic,
                Image,
                partial(self.img_callback, img_topic),
                queue_size=1,
                buff_size=2**30,
            )
        rospy.loginfo(f"[{self.node_name}]: Waiting for images...")
        while not all([self.msgs[s] is not None for s in subscriptions]):
            pass
        rospy.loginfo(f"[{self.node_name}]: Received images!")

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.current_arm_pose = None
        self.link_wr1_position, self.link_wr1_rotation = None, None
        if self.proprioception:
            rospy.Subscriber(
                rt.ROBOT_STATE,
                Float32MultiArray,
                self.robot_state_callback,
                queue_size=1,
            )
            assert spot is not None
            self.spot = spot
        else:
            self.spot = None

        # Subscribe the target object detection
        self.target_object_detection = []
        rospy.Subscriber("/open_voc_object_detector", String, self.detections_callback)

        self.pick_target = "None"
        self.pick_object = "None"
        self.place_target = "None"

        rospy.loginfo(f"[{self.node_name}]: Robot subscribing has started.")

    def img_callback(self, topic, msg):
        self.msgs[topic] = msg
        self.updated[topic] = True

    def curr_transform(self):
        # Assume body is at default height of 0.5 m
        # This is local_T_global.
        return mn.Matrix4.from_(
            mn.Matrix4.rotation_z(mn.Rad(self.yaw)).rotation(),
            mn.Vector3(self.x, self.y, 0.5),
        )

    def robot_state_callback(self, msg):
        x, y, yaw = msg.data[:3]
        self.x, self.y, self.yaw = self.spot.xy_yaw_global_to_home(x, y, yaw)
        self.current_arm_pose = msg.data[3:-7]
        link_wr1_position, self.link_wr1_rotation = (
            msg.data[-7:][:3],
            msg.data[-7:][3:],
        )
        # Transform it the home frame
        self.link_wr1_position = np.array(
            self.curr_transform().transform_point(mn.Vector3(link_wr1_position))
        )

    def detections_callback(self, msg):
        try:
            name, x, y, z = msg.data.split(",")
            self.target_object_detection = [name, float(x), float(y), float(z)]
        except Exception:
            print("No detection")

    def msg_to_cv2(self, *args, **kwargs) -> np.array:
        return self.cv_bridge.imgmsg_to_cv2(*args, **kwargs)


if __name__ == "__main__":
    from spot_wrapper.spot import Spot

    spot = Spot("get_data")
    sub = SpotRobotSubscriberMixin(spot)

    data_of_interest = ["ee_pose", "target_object_detection"]

    data = {}  # type: ignore
    for kk in data_of_interest:
        data[kk] = []

    print("start")
    while True:
        time.sleep(0.1)
        cur_time = time.time()
        # To store the ee location in the home frame
        data_dict = {
            "timestamp": str(cur_time),
            "pos_x": str(sub.link_wr1_position[0]),
            "pos_y": str(sub.link_wr1_position[1]),
            "pos_z": str(sub.link_wr1_position[2]),
            "rot_x": str(sub.link_wr1_rotation[0]),
            "rot_y": str(sub.link_wr1_rotation[1]),
            "rot_z": str(sub.link_wr1_rotation[2]),
            "rot_w": str(sub.link_wr1_rotation[3]),
        }
        data["ee_pose"].append(data_dict.copy())

        # To store the object detection in the home frame
        if sub.target_object_detection != []:
            data_dict = {
                "timestamp": str(cur_time),
                "label": str(sub.target_object_detection[0]),
                "pos_x": str(sub.target_object_detection[1]),
                "pos_y": str(sub.target_object_detection[2]),
                "pos_z": str(sub.target_object_detection[3]),
            }
            data["target_object_detection"].append(data_dict.copy())

        if len(data["ee_pose"]) == 500 or len(data["target_object_detection"]) == 500:
            break

        print(len(data["ee_pose"]))

    for kk in data_of_interest:
        file_name = f"/home/jmmy/research/Nexus_data/{kk}.json"
        # Write the JSON string to a file
        with open(file_name, "w") as outfile:
            json.dump(data[kk], outfile)
