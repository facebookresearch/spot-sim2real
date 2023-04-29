import argparse
import time

import numpy as np
import rospy
from spot_wrapper.spot import Spot
from spot_wrapper.utils import say
from std_msgs.msg import Float32MultiArray, String

from spot_rl.utils.utils import ros_topics as rt

NAV_POSE_BUFFER_LEN = 1


class SpotRosProprioceptionPublisher:
    def __init__(self, spot):
        rospy.init_node("spot_ros_proprioception_node", disable_signals=True)
        self.spot = spot

        # Instantiate filtered image publishers
        self.pub = rospy.Publisher(rt.ROBOT_STATE, Float32MultiArray, queue_size=1)
        self.last_publish = time.time()
        rospy.loginfo("[spot_ros_proprioception_node]: Publishing has started.")

        self.nav_pose_buff = None
        self.buff_idx = 0

    def publish_msgs(self):
        st = time.time()
        robot_state = self.spot.get_robot_state()
        msg = Float32MultiArray()
        xy_yaw = self.spot.get_xy_yaw(robot_state=robot_state, use_boot_origin=True)
        if self.nav_pose_buff is None:
            self.nav_pose_buff = np.tile(xy_yaw, [NAV_POSE_BUFFER_LEN, 1])
        else:
            self.nav_pose_buff[self.buff_idx] = xy_yaw
        self.buff_idx = (self.buff_idx + 1) % NAV_POSE_BUFFER_LEN
        xy_yaw = np.mean(self.nav_pose_buff, axis=0)

        joints = self.spot.get_arm_proprioception(robot_state=robot_state).values()

        position, rotation = self.spot.get_base_transform_to("link_wr1")
        gripper_transform = [position.x, position.y, position.z] + [
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w,
        ]

        msg.data = np.array(
            list(xy_yaw) + [j.position.value for j in joints] + gripper_transform,
            dtype=np.float32,
        )

        # Limit publishing to 10 Hz max
        if time.time() - self.last_publish > 1 / 10:
            self.pub.publish(msg)
            rospy.loginfo(
                f"[spot_ros_proprioception_node]: "
                "Proprioception retrieval / publish time: "
                f"{1/(time.time() - st):.4f} / "
                f"{1/(time.time() - self.last_publish):.4f} Hz"
            )
            self.last_publish = time.time()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--proprioception", action="store_true")
    parser.add_argument("-t", "--text-to-speech", action="store_true")
    args = parser.parse_args()

    if args.text_to_speech:
        tts_callback = lambda msg: say(msg.data)
        rospy.init_node("spot_ros_tts_node", disable_signals=True)
        rospy.Subscriber(rt.TEXT_TO_SPEECH, String, tts_callback, queue_size=1)
        rospy.loginfo("[spot_ros_tts_node]: Listening for text to dictate.")
        rospy.spin()
    elif args.proprioception:
        name = "SpotRosProprioceptionPublisher"
        node = SpotRosProprioceptionPublisher(Spot(name))
        while not rospy.is_shutdown():
            node.publish_msgs()
    else:
        raise RuntimeError("One and only one arg must be provided.")
