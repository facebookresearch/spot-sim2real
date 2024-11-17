# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os
import os.path as osp
import time
from collections import deque
from typing import List

import cv2
import numpy as np
import rospy
import tqdm
from spot_rl.utils.robot_subscriber import SpotRobotSubscriberMixin
from spot_rl.utils.utils import ros_topics as rt
from spot_wrapper.utils import resize_to_tallest

RAW_IMG_TOPICS = [rt.HEAD_DEPTH, rt.HAND_DEPTH, rt.HAND_RGB, rt.HAND_RGB]

PROCESSED_IMG_TOPICS = [
    rt.FILTERED_HEAD_DEPTH,
    rt.FILTERED_HAND_DEPTH,
    rt.MASK_RCNN_VIZ_TOPIC,
    rt.MULTI_OBJECT_DETECTION_VIZ_TOPIC,
]

FOUR_CC = cv2.VideoWriter_fourcc(*"MP4V")
FPS = 30
TEXT_FOR_LSC_DEMO = False


class VisualizerMixin:
    def __init__(self, headless=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recording = False
        self.frames = []
        self.headless = headless
        self.curr_video_time = time.time()
        self.out_path = None
        self.video = None
        self.dim = None
        self.new_video_started = False
        self.named_window = "ROS Spot Images"

        if not TEXT_FOR_LSC_DEMO:
            self._cur_human_action = "None"
            self._has_display_since_time = time.time()

    def generate_composite(self):
        raise NotImplementedError

    @staticmethod
    def overlay_text(img, text, color=(0, 0, 255), size=2.0, thickness=4):
        viz_img = img.copy()
        line, font, font_size, font_thickness = (
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            size,
            thickness,
        )

        height, width = img.shape[:2]
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
        # Skip if no messages were updated
        currently_saving = not self.recording and self.frames
        img = self.generate_composite() if not currently_saving else None
        if not self.headless:
            if img is not None:
                if self.recording:
                    viz_img = self.overlay_text(img, "RECORDING IS ON!")
                    cv2.imshow(self.named_window, viz_img)
                else:
                    cv2.imshow(self.named_window, img)

            key = cv2.waitKey(1)
            if key != -1:
                if ord("r") == key and not currently_saving:
                    self.recording = not self.recording
                elif ord("q") == key:
                    exit()

        if img is not None:
            self.dim = img.shape[:2]

            # Video recording
            if self.recording:
                self.frames.append(time.time())
                if self.video is None:
                    height, width = img.shape[:2]
                    self.out_path = f"{time.time()}.mp4"
                    self.video = cv2.VideoWriter(
                        self.out_path, FOUR_CC, FPS, (width, height)
                    )
                self.video.write(img)

        if currently_saving and not self.recording:
            self.save_video()

    def save_video(self):
        if self.video is None:
            return
        # Close window while we work
        cv2.destroyAllWindows()

        # Save current buffer
        self.video.release()
        old_video = cv2.VideoCapture(self.out_path)
        ret, img = old_video.read()

        # Re-make video with correct timing
        height, width = self.dim
        self.new_video_started = True
        new_video = cv2.VideoWriter(
            self.out_path.replace(".mp4", "_final.mp4"),
            FOUR_CC,
            FPS,
            (width, height),
        )
        curr_video_time = self.frames[0]
        for idx, timestamp in enumerate(tqdm.tqdm(self.frames)):
            if not ret:
                break
            if idx + 1 >= len(self.frames):
                new_video.write(img)
            else:
                next_timestamp = self.frames[idx + 1]
                while curr_video_time < next_timestamp:
                    new_video.write(img)
                    curr_video_time += 1 / FPS
            ret, img = old_video.read()

        new_video.release()
        os.remove(self.out_path)
        self.video, self.out_path, self.frames = None, None, []
        self.new_video_started = False

    def delete_videos(self):
        for i in [self.out_path, self.out_path.replace(".mp4", "_final.mp4")]:
            if osp.isfile(i):
                os.remove(i)


class SpotRosVisualizer(VisualizerMixin, SpotRobotSubscriberMixin):
    node_name = "SpotRosVisualizer"
    no_raw = False
    proprioception = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_seen = {topic: time.time() for topic in self.msgs.keys()}
        self.fps = {topic: deque(maxlen=10) for topic in self.msgs.keys()}

    def pretty_parse_object_or_furniture_str(self, str_name: str):
        target_str = "Default Name"
        str_list = str_name.split("_")
        if len(str_list) == 1:
            target_str = str_list[0]
        else:
            # Check if first element of target_str_list is a number
            target_str = (
                " ".join(str_list[1:]) if str_list[0].isdigit() else " ".join(str_list)
            )

        return target_str

    def beautify_next_action_str(self, msg_list: List[str]):
        # Element0 if this list is always timestamp, Element1 is skill name and Element2 is skill target
        assert (
            len(msg_list) == 3
        ), f"Invalid {msg_list=}. Cannot have more than 3 elements in this list"

        # Get skill name
        action_name = msg_list[1]
        beautiful_str = ""

        if "nav" in action_name or "Nav" in action_name:
            beautiful_str += f"Navigating to {self.pretty_parse_object_or_furniture_str(msg_list[2])}"
        elif "explore" in action_name:
            beautiful_str += (
                f"Exploring {self.pretty_parse_object_or_furniture_str(msg_list[2])}"
            )
        elif "pick" in action_name or "Pick" in action_name:
            beautiful_str += (
                f"Picking up {self.pretty_parse_object_or_furniture_str(msg_list[2])}"
            )
        elif "place" in action_name or "Place" in action_name:
            place_inputs = msg_list[2].split(";")
            object_name = self.pretty_parse_object_or_furniture_str(place_inputs[0])
            relation = place_inputs[1]
            receptacle_name = self.pretty_parse_object_or_furniture_str(
                place_inputs[2].strip()
            )
            beautiful_str += f"Placing {object_name} {relation} {receptacle_name}"
        elif "dock" in action_name or "Dock" in action_name:
            beautiful_str += "Plan complete!!\n Going to dock"
        else:
            beautiful_str += f"{action_name} called for {msg_list[2]}"

        return beautiful_str

    def beautify_human_action_str(self, msg):
        msg = msg.split(",")
        if len(msg) < 2:
            return "None"
        else:
            msg = f"{msg[0]} {msg[1]}"
            return msg

    def generate_composite(self):
        if not any(self.updated.values()):
            # No imgs were refreshed. Skip.
            return None

        refreshed_topics = [k for k, v in self.updated.items() if v]

        # Gather latest images
        raw_msgs = [self.msgs[i] for i in RAW_IMG_TOPICS]
        processed_msgs = [self.msgs[i] for i in PROCESSED_IMG_TOPICS]

        raw_imgs = [self.msg_to_cv2(i) for i in raw_msgs if i is not None]

        # Replace any Nones with black images if raw version exists. We (safely) assume
        # here that there is no processed image anyway if the raw image does not exist.
        processed_imgs = []
        for idx, raw_msg in enumerate(raw_msgs):
            if processed_msgs[idx] is not None:
                processed_imgs.append(self.msg_to_cv2(processed_msgs[idx]))
            elif processed_msgs[idx] is None and raw_msg is not None:
                processed_imgs.append(np.zeros_like(raw_imgs[idx]))

        # Crop gripper images
        if raw_msgs[1] is not None:
            for imgs in [raw_imgs, processed_imgs]:
                imgs[1] = imgs[1][:, 124:-60]
        try:
            img = np.vstack(
                [
                    resize_to_tallest(bgrify_grayscale_imgs(i), hstack=True)
                    for i in [raw_imgs, processed_imgs]
                ]
            )
            # The normal size of the images is 480x2279x3.
        except Exception:
            print("Cannot np.vstack image, skipping...")
            return

        if TEXT_FOR_LSC_DEMO:
            # Add Pick receptacle, Object, Place receptacle information on the side
            pck = rospy.get_param("/viz_pick", "None")
            obj = rospy.get_param("/viz_object", "None")
            plc = rospy.get_param("/viz_place", "None")
            information_string = (
                "Pick from:\n"
                + pck
                + "\n\nObject Target:\n"
                + obj
                + "\n\nPlace to:\n"
                + plc
            )
            display_img = 255 * np.ones(
                (img.shape[0], int(img.shape[1] / 4), img.shape[2]), dtype=np.uint8
            )
            display_img = self.overlay_text(
                display_img,
                information_string,
                color=(255, 0, 0),
                size=0.9,
                thickness=4,
            )
            img = resize_to_tallest([img, display_img], hstack=True)
        else:
            # Add current robot action and human action on the side
            display_img = 255 * np.ones(
                (img.shape[0], int(img.shape[1] / 1.5), img.shape[2]), dtype=np.uint8
            )
            robot_action = rospy.get_param(
                "skill_name_input", f"{str(time.time())},None,None"
            )
            robot_action = robot_action.split(",")
            # Santize the nav name
            if "nav" in robot_action[1]:
                robot_action[1] = "nav"
                robot_action[2] = robot_action[2].strip("|").split(";")[-1]
            elif "explore" in robot_action[1]:
                robot_action[2] = (
                    robot_action[2]
                    .strip("|")
                    .split("|")[-1]
                    .split(";")[-1]
                    .split(":")[0]
                )
            if "None" not in robot_action:
                robot_action = self.beautify_next_action_str(robot_action)
            else:
                robot_action = "Thinking..."
            information_string = "Robot action: " + robot_action

            # Add human action
            human_action = rospy.get_param("human_action", "0,None,None,None")
            human_action = (
                "None"
                if "None" in human_action
                else ",".join(human_action.split(",")[1:])
            )

            if human_action != "None":
                # If there is a human action, we update the string immediately
                self._cur_human_action = human_action
                # Set the timer
                self._has_display_since_time = time.time()
            elif (
                human_action == "None"
                and time.time() - self._has_display_since_time < 10
            ):
                # If human action is None, we wait for this many seconds to display things
                self._cur_human_action = self._cur_human_action
            else:
                self._cur_human_action = "None"

            information_string += f"\nHuman action: {self.beautify_human_action_str(self._cur_human_action)}"

            world_graph_simple_viz = rospy.get_param("world_graph_simple_viz", "")
            world_graph_simple_viz = world_graph_simple_viz.replace(": ", " on ")
            information_string += f"\nWorld graph:\n{world_graph_simple_viz}"

            # Add human action
            display_img = self.overlay_text(
                display_img,
                information_string,
                color=(255, 0, 0),
                size=1.2,
                thickness=3,
            )
            img = resize_to_tallest([img, display_img], hstack=True)

        for topic in refreshed_topics:
            curr_time = time.time()
            self.updated[topic] = False
            self.fps[topic].append(1 / (curr_time - self.last_seen[topic]))
            self.last_seen[topic] = curr_time

        all_topics = RAW_IMG_TOPICS + PROCESSED_IMG_TOPICS
        print(" ".join([f"{k[1:]}: {np.mean(self.fps[k]):.2f}" for k in all_topics]))

        return img


def bgrify_grayscale_imgs(imgs):
    return [
        cv2.cvtColor(i, cv2.COLOR_GRAY2BGR) if i.ndim == 2 or i.shape[-1] == 1 else i
        for i in imgs
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()

    srv = None
    try:
        srv = SpotRosVisualizer(headless=args.headless)
        srv.initializeWindow()
        if args.record:
            srv.recording = True
        while not rospy.is_shutdown():
            srv.vis_imgs()
    except Exception as e:
        print("Ending script.")
        if not args.headless:
            cv2.destroyAllWindows()
        if srv is not None:
            try:
                if srv.new_video_started:
                    print("Deleting unfinished videos.")
                    srv.delete_videos()
                else:
                    srv.save_video()
            except Exception:
                print("Deleting unfinished videos")
                srv.delete_videos()
                exit()
        raise e


if __name__ == "__main__":
    main()
