# # Copyright (c) Meta Platforms, Inc. and its affiliates.
# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.


# from functools import partial

# import numpy as np
# import rospy
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image
# from spot_rl.utils.utils import ros_topics as rt
# from spot_wrapper.spot import Spot
# from std_msgs.msg import Float32MultiArray

# IMG_TOPICS = [
#     rt.MASK_RCNN_VIZ_TOPIC,
#     rt.HEAD_DEPTH,
#     rt.HAND_DEPTH,
#     rt.HAND_RGB,
#     rt.FILTERED_HEAD_DEPTH,
#     rt.FILTERED_HAND_DEPTH,
#     rt.IRS_RGB,
#     rt.IRS_DEPTH,
#     rt.GRIPPER_RGB,
#     rt.GRIPPER_DEPTH,
# ]
# NO_RAW_IMG_TOPICS = [
#     rt.MASK_RCNN_VIZ_TOPIC,
#     rt.HAND_RGB,
#     rt.FILTERED_HEAD_DEPTH,
#     rt.FILTERED_HAND_DEPTH,
# ]


# class SpotRobotSubscriberMixin:
#     node_name = "SpotRobotSubscriber"
#     no_raw = False
#     proprioception = True

#     def __init__(self, spot=None, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         rospy.init_node(self.node_name, disable_signals=True)
#         self.cv_bridge = CvBridge()

#         subscriptions = NO_RAW_IMG_TOPICS if self.no_raw else IMG_TOPICS

#         # Maps a topic name to the latest msg from it
#         self.msgs = {topic: None for topic in subscriptions}
#         self.updated = {topic: False for topic in subscriptions}

#         for img_topic in subscriptions:
#             rospy.Subscriber(
#                 img_topic,
#                 Image,
#                 partial(self.img_callback, img_topic),
#                 queue_size=1,
#                 buff_size=2**30,
#             )
#         rospy.loginfo(f"[{self.node_name}]: Waiting for images...")
#         while not all([self.msgs[s] is not None for s in subscriptions]):
#             # rospy.signal_shutdown("No subscription topics available.")
#             pass
#         rospy.loginfo(f"[{self.node_name}]: Received images!")

#         self.x = 0.0
#         self.y = 0.0
#         self.yaw = 0.0
#         self.current_arm_pose = None
#         self.link_wr1_position, self.link_wr1_rotation = None, None
#         if self.proprioception:
#             rospy.Subscriber(
#                 rt.ROBOT_STATE,
#                 Float32MultiArray,
#                 self.robot_state_callback,
#                 queue_size=1,
#             )
#             assert spot is not None
#             self.spot = spot
#         else:
#             self.spot = None

#         self.pick_target = "None"
#         self.pick_object = "None"
#         self.place_target = "None"

#         rospy.loginfo(f"[{self.node_name}]: Robot subscribing has started.")

#     def img_callback(self, topic, msg):
#         self.msgs[topic] = msg
#         self.updated[topic] = True

#     def robot_state_callback(self, msg):
#         x, y, yaw = msg.data[:3]
#         self.x, self.y, self.yaw = self.spot.xy_yaw_global_to_home(x, y, yaw)
#         self.current_arm_pose = msg.data[3:-7]
#         self.link_wr1_position, self.link_wr1_rotation = (
#             msg.data[-7:][:3],
#             msg.data[-7:][3:],
#         )

from functools import partial

import numpy as np

#     def msg_to_cv2(self, *args, **kwargs) -> np.array:
#         return self.cv_bridge.imgmsg_to_cv2(*args, **kwargs)
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from spot_rl.utils.utils import ros_topics as rt
from spot_wrapper.spot import Spot
from std_msgs.msg import Float32MultiArray

IMG_TOPICS = [
    rt.MASK_RCNN_VIZ_TOPIC,
    rt.HEAD_DEPTH,
    rt.HAND_DEPTH,
    rt.HAND_RGB,
    rt.FILTERED_HEAD_DEPTH,
    rt.FILTERED_HAND_DEPTH,
    rt.IRS_RGB,
    rt.IRS_DEPTH,
    rt.GRIPPER_RGB,
    rt.GRIPPER_DEPTH,
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
        self.wait_for_images(subscriptions)
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

    def wait_for_images(self, subscriptions, timeout=30):
        start_time = rospy.get_time()
        while not all([self.msgs[s] is not None for s in subscriptions]):
            if rospy.get_time() - start_time > timeout:
                rospy.logwarn(f"[{self.node_name}]: Timeout waiting for image topics.")
                break
            rospy.sleep(1)  # Sleep for a short time to avoid busy-waiting

    def img_callback(self, topic, msg):
        try:
            self.msgs[topic] = msg
            self.updated[topic] = True
        except Exception as e:
            rospy.logerr(
                f"[{self.node_name}]: Error processing message from topic {topic}: {e}"
            )

    def robot_state_callback(self, msg):
        try:
            x, y, yaw = msg.data[:3]
            self.x, self.y, self.yaw = self.spot.xy_yaw_global_to_home(x, y, yaw)
            self.current_arm_pose = msg.data[3:-7]
            self.link_wr1_position, self.link_wr1_rotation = (
                msg.data[-7:][:3],
                msg.data[-7:][3:],
            )
        except Exception as e:
            rospy.logerr(
                f"[{self.node_name}]: Error processing robot state message: {e}"
            )

    def msg_to_cv2(self, *args, **kwargs) -> np.array:
        try:
            return self.cv_bridge.imgmsg_to_cv2(*args, **kwargs)
        except Exception as e:
            rospy.logerr(f"[{self.node_name}]: Error converting message to CV2: {e}")
            return np.array([])
