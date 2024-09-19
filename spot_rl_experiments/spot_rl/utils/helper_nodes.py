# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import time

import numpy as np
import rospy
from nav_msgs.msg import Odometry
from spot_rl.utils.utils import ros_frames as rf
from spot_rl.utils.utils import ros_topics as rt
from spot_wrapper.spot import Spot
from spot_wrapper.utils import say
from std_msgs.msg import Float32MultiArray, String
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

NAV_POSE_BUFFER_LEN = 1


class SpotRosProprioceptionPublisher:
    def __init__(self, spot):
        rospy.init_node("spot_ros_proprioception_node", disable_signals=True)
        self.spot = spot

        # Publishers
        self.pub = rospy.Publisher(rt.ROBOT_STATE, Float32MultiArray, queue_size=1)
        self.odom_broadcaster = rospy.Publisher(
            rt.ODOM_TOPIC, Odometry, queue_size=1, latch=True
        )
        self.marker_pub = rospy.Publisher(
            "gripper_trajectory_marker_array", MarkerArray, queue_size=1
        )

        self.last_publish = time.time()
        self.nav_pose_buff = None
        self.buff_idx = 0

        # Static transform broadcaster
        self.static_tf_broadcaster = StaticTransformBroadcaster()

        self.gripper_traj = []

    def publish_msgs(self):
        st = time.time()
        robot_state = self.spot.get_robot_state()
        robot_kinematic_snapshot_tree = robot_state.kinematic_state.transforms_snapshot

        position, rotation = self.spot.get_base_transform_to("link_wr1")
        ee_pose = np.array([position.x, position.y, position.z])

        self.gripper_traj.append(ee_pose)
        if len(self.gripper_traj) > 100:
            self.gripper_traj.pop(0)

        msg = Float32MultiArray()
        xy_yaw = self.spot.get_xy_yaw(robot_state=robot_state, use_boot_origin=True)
        if self.nav_pose_buff is None:
            self.nav_pose_buff = np.tile(xy_yaw, [NAV_POSE_BUFFER_LEN, 1])
        else:
            self.nav_pose_buff[self.buff_idx] = xy_yaw
        self.buff_idx = (self.buff_idx + 1) % NAV_POSE_BUFFER_LEN
        xy_yaw = np.mean(self.nav_pose_buff, axis=0)

        joints = self.spot.get_arm_proprioception(robot_state=robot_state).values()
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

        if time.time() - self.last_publish > 1 / 10:
            self.pub.publish(msg)
            self.static_tf_broadcaster.sendTransform(
                self.spot.get_ros_TransformStamped_vision_T_body(
                    robot_kinematic_snapshot_tree
                )
            )
            pose = self.spot.get_ros_Pose_vision_T_body(robot_kinematic_snapshot_tree)
            msg = Odometry()
            msg.pose.pose = pose
            msg.header.stamp = rospy.Time.now()
            msg.child_frame_id = rf.SPOT
            msg.header.frame_id = rf.SPOT_WORLD
            self.odom_broadcaster.publish(msg)

            self.publish_ee_trajectory()

            rospy.loginfo(
                f"[spot_ros_proprioception_node]: Proprioception retrieval / publish time: {1/(time.time() - st):.4f} / {1/(time.time() - self.last_publish):.4f} Hz"
            )
            self.last_publish = time.time()

    def publish_ee_trajectory(self):
        marker_array = MarkerArray()
        for i, pose in enumerate(self.gripper_traj):
            marker = Marker()
            marker.header.frame_id = "spot_world"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "ee_trajectory"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = pose[0]
            marker.pose.position.y = pose[1]
            marker.pose.position.z = pose[2]
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.lifetime = rospy.Duration(0)
            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--proprioception", action="store_true")
    parser.add_argument("-t", "--text-to-speech", action="store_true")
    args = parser.parse_args()

    if args.text_to_speech:
        tts_callback = lambda msg: say(msg.data)  # noqa: E731
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
