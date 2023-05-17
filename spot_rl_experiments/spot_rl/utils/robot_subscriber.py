from functools import partial

import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from spot_wrapper.spot import Spot
from std_msgs.msg import Float32MultiArray, String

from spot_rl.utils.utils import ros_topics as rt

IMG_TOPICS = [
    #rt.MASK_RCNN_VIZ_TOPIC,
    rt.HEAD_DEPTH,
    # rt.HAND_DEPTH,
    # rt.HAND_RGB,
    rt.FILTERED_HEAD_DEPTH,
    # rt.FILTERED_HAND_DEPTH,
]
NO_RAW_IMG_TOPICS = [
    #rt.MASK_RCNN_VIZ_TOPIC,
    # rt.HAND_RGB,
    rt.FILTERED_HEAD_DEPTH,
    # rt.FILTERED_HAND_DEPTH,
]
INSTRUCTIONS_TOPICS = [
    rt.INSTRUCTIONS_TOPIC,
]

class SpotRobotSubscriberMixin:
    node_name = "SpotRobotSubscriber"
    no_raw = False
    proprioception = True
    instruction_display = True

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
                buff_size=2 ** 30,
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
        if self.instruction_display:
            rospy.Subscriber(
                rt.INSTRUCTIONS_TOPIC,
                String,
                self.instruction_callback,
                queue_size=1,
            )
        else:
            rospy.loginfo(f"[{self.node_name}]: Disabled display for robot instruction commands.")

        rospy.loginfo(f"[{self.node_name}]: Robot subscribing has started.")

    def img_callback(self, topic, msg):
        self.msgs[topic] = msg
        self.updated[topic] = True

    def robot_state_callback(self, msg):
        x, y, yaw = msg.data[:3]
        self.x, self.y, self.yaw = self.spot.xy_yaw_global_to_home(x, y, yaw)
        self.current_arm_pose = msg.data[3:-7]
        self.link_wr1_position, self.link_wr1_rotation = (
            msg.data[-7:][:3],
            msg.data[-7:][3:],
        )

    def instruction_callback(self, msg):
        rospy.loginfo(f"[{self.node_name}]: Received instruction: {msg.data}")
        instructions = msg.data.split(",")

        if len(instructions) != 3:
            rospy.logerr(f"[{self.node_name}]: Invalid instruction received: {msg.data}")
            return
        else:
            self.pick_target = instructions[0].strip()
            self.pick_object = instructions[1].strip()
            self.place_target = instructions[2].strip()

            if(self.pick_target == "" or self.pick_object == "" or self.place_target == ""):
                rospy.logerr(f"[{self.node_name}]: Atleast one of the instruction is empty: {self.pick_target}, {self.pick_object}, {self.place_target}")
                return

            rospy.loginfo(f"[{self.node_name}]: Parsed instruction: {self.pick_target}, {self.pick_object}, {self.place_target}")


    def msg_to_cv2(self, *args, **kwargs) -> np.array:
        return self.cv_bridge.imgmsg_to_cv2(*args, **kwargs)
