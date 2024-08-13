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
from std_msgs.msg import Float32MultiArray

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
    node_name = "SpotRobotSubscriber"
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
        self.link_wr1_position = np.array(
            self.curr_transform().transform_point(mn.Vector3(link_wr1_position))
        )

    def msg_to_cv2(self, *args, **kwargs) -> np.array:
        return self.cv_bridge.imgmsg_to_cv2(*args, **kwargs)


if __name__ == "__main__":
    from spot_wrapper.spot import Spot

    spot = Spot("get_data")
    sub = SpotRobotSubscriberMixin(spot)

    data = []
    print("start")
    while True:
        time.sleep(0.1)
        data_dict = {
            "timestamp": str(time.time()),
            "pos_x": str(sub.link_wr1_position[0]),
            "pos_y": str(sub.link_wr1_position[1]),
            "pos_z": str(sub.link_wr1_position[2]),
            "rot_x": str(sub.link_wr1_rotation[0]),
            "rot_y": str(sub.link_wr1_rotation[1]),
            "rot_z": str(sub.link_wr1_rotation[2]),
            "rot_w": str(sub.link_wr1_rotation[3]),
        }
        data.append(data_dict)
        if len(data) == 500:
            break
        print(len(data))

    file_name = "/home/jmmy/research/Nexus_data/ee_pose.json"
    # Write the JSON string to a file
    with open(file_name, "w") as outfile:
        json.dump(data, outfile)
