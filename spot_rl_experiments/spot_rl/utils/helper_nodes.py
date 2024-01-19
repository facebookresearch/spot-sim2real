# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import csv
import curses
import os
import pickle
import signal
import time

import numpy as np
import rospy
from spot_rl.envs.base_env import SpotBaseEnv
from spot_rl.utils.utils import construct_config
from spot_rl.utils.utils import ros_topics as rt
from spot_wrapper.spot import Spot
from spot_wrapper.spot import SpotCamIds as Cam
from spot_wrapper.spot import image_response_to_cv2, scale_depth_img
from spot_wrapper.utils import say
from std_msgs.msg import Float32MultiArray, String

NAV_POSE_BUFFER_LEN = 1

MAX_PUBLISH_FREQ = 20
MAX_DEPTH = 3.5
MAX_HAND_DEPTH = 1.7


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


IMG_SOURCES = [
    Cam.FRONTRIGHT_DEPTH,
    Cam.FRONTLEFT_DEPTH,
    Cam.HAND_DEPTH_IN_HAND_COLOR_FRAME,
    Cam.HAND_COLOR,
]


class SpotRosProprioceptionSaver:
    def __init__(self, spot):
        self.spot = spot
        self.nav_pose_buff = None
        self.buff_idx = 0
        self.data_list = []
        self.data_image_list = []
        self._parallel_inference_mode = True

        if self._parallel_inference_mode:
            config = construct_config(opts=[])
            self.base_env = SpotBaseEnv(config, self.spot)

    def detections_cb(self, msg):
        timestamp, detections_str = msg.data.split("|")
        self.detections_buffer["detections"][int(timestamp)] = detections_str

    def _scale_depth(self, img, head_depth=False):
        img = scale_depth_img(
            img, max_depth=MAX_DEPTH if head_depth else MAX_HAND_DEPTH
        )
        return np.uint8(img * 255.0)

    def publish_msgs(self, start_time):
        single_process_time = time.time()
        robot_state = self.spot.get_robot_state()

        xy_yaw = self.spot.get_xy_yaw(robot_state=robot_state, use_boot_origin=True)
        if self.nav_pose_buff is None:
            self.nav_pose_buff = np.tile(xy_yaw, [NAV_POSE_BUFFER_LEN, 1])
        else:
            self.nav_pose_buff[self.buff_idx] = xy_yaw
        self.buff_idx = (self.buff_idx + 1) % NAV_POSE_BUFFER_LEN
        xy_yaw = np.mean(self.nav_pose_buff, axis=0)

        joints = self.spot.get_arm_proprioception(robot_state=robot_state).values()

        gripper = self.spot.get_gripper_proprioception(robot_state=robot_state).values()

        position, rotation = self.spot.get_base_transform_to("link_wr1")

        # Get the ee pos and rotation from Spot
        ee_xyz, ee_rpy = self.spot.get_ee_pos_in_body_frame()
        ee_xyz = ee_xyz.tolist()
        ee_rpy = ee_rpy.tolist()
        gripper_transform = ee_xyz + ee_rpy

        # Get the hand rgb/depth image
        if self._parallel_inference_mode:
            hand_depth, hand_rgb = self.base_env.get_arm_images()
            head_depth = self.base_env.get_head_depth()
            # Information about the images
            # hand_depth: (240, 228, 1); max: 1.0; min: 0.02
            # hand_rgb  : (480, 640, 3); max: 225; min: 0.0
            # head_depth: (212, 240, 1); max: 1.0; min: 0.02
        else:
            image_responses = self.spot.get_image_responses(IMG_SOURCES, quality=100)
            imgs_list = [image_response_to_cv2(r) for r in image_responses]
            imgs = {k: v for k, v in zip(IMG_SOURCES, imgs_list)}
            head_depth = np.hstack(
                [imgs[Cam.FRONTRIGHT_DEPTH], imgs[Cam.FRONTLEFT_DEPTH]]
            )
            head_depth = self._scale_depth(head_depth, head_depth=True)
            hand_depth = self._scale_depth(imgs[Cam.HAND_DEPTH_IN_HAND_COLOR_FRAME])
            hand_rgb = imgs[Cam.HAND_COLOR]

        cur_time = time.time() - start_time
        self.data_list.append(
            [cur_time]
            + list(xy_yaw)
            + [j.position.value for j in joints]
            + [j.position.value for j in gripper]
            + gripper_transform
        )
        self.data_image_list.append([cur_time, head_depth, hand_depth, hand_rgb])

        print("Freq:", 1.0 / (time.time() - single_process_time), "hz")


def raise_error(sig, frame):
    raise RuntimeError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--proprioception", action="store_true")
    parser.add_argument("-t", "--text-to-speech", action="store_true")
    parser.add_argument("-s", "--save-proprioception", action="store_true")
    args = parser.parse_args()

    if args.text_to_speech:
        tts_callback = lambda msg: say(msg.data)  # noqa
        rospy.init_node("spot_ros_tts_node", disable_signals=True)
        rospy.Subscriber(rt.TEXT_TO_SPEECH, String, tts_callback, queue_size=1)
        rospy.loginfo("[spot_ros_tts_node]: Listening for text to dictate.")
        rospy.spin()
    elif args.proprioception:
        name = "SpotRosProprioceptionPublisher"
        node = SpotRosProprioceptionPublisher(Spot(name))
        while not rospy.is_shutdown():
            node.publish_msgs()
    elif args.save_proprioception:
        name = "SpotRosProprioceptionSaver"
        saver = SpotRosProprioceptionSaver(Spot(name))
        start_time = time.time()
        # prepare the key to termination
        # Start in-terminal GUI
        stdscr = curses.initscr()
        stdscr.nodelay(True)
        curses.noecho()
        signal.signal(signal.SIGINT, raise_error)
        # TODO: Need to pip install keyboard
        start_recording = False
        print("Press s to start recording, and d to done with recording...")
        while True:
            if start_recording:
                print("recording\n")
                saver.publish_msgs(start_time)

            pressed_key = stdscr.getch()
            # Don't update if no key was pressed or we updated too recently
            if pressed_key == -1:
                continue

            pressed_key_char = chr(pressed_key)
            if pressed_key_char == "d":
                print("end recording\n")
                break
            if pressed_key_char == "s":
                print("start recording\n")
                start_recording = True

        print("Save the file...")
        save_path = "/home/jimmytyyang/Downloads/open_drawer_data/open_drawer_v0.0.2"
        with open(save_path + ".pkl", "wb") as handle:
            pickle.dump(saver.data_list, handle)

        with open(save_path + "_image.pkl", "wb") as handle:
            pickle.dump(saver.data_image_list, handle)

        with open(save_path + ".csv", "w", newline="") as csvfile:
            writer = csv.writer(
                csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
            )
            writer.writerow(
                [
                    "time",
                    "base x",
                    "base y",
                    "base yaw",
                    "arm joint 1",
                    "arm joint 2",
                    "arm joint 3",
                    "arm joint 4",
                    "arm joint 5",
                    "arm joint 6",
                    "ee joint",
                    "ee local x",
                    "ee local y",
                    "ee local z",
                    "ee local r",
                    "ee local p",
                    "ee local y",
                ]
            )
            for row_data in saver.data_list:
                writer.writerow(row_data)
        print("Finished save the file...")
        breakpoint()
    else:
        raise RuntimeError("One and only one arg must be provided.")
