# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os
import os.path as osp
import time
from collections import deque

import cv2
import numpy as np
import rospy
import tqdm
from spot_rl.utils.robot_subscriber import SpotRobotSubscriberMixin
from spot_rl.utils.utils import ros_topics as rt
from spot_wrapper.utils import resize_to_tallest

# from intel_realsense_payload_for_spotsim2real.IntelRealSenseCameraInterface import IntelRealSenseCameraInterface

RAW_IMG_TOPICS = [rt.HEAD_DEPTH, rt.GRIPPER_DEPTH, rt.GRIPPER_RGB, rt.IRS_RGB]

PROCESSED_IMG_TOPICS = [
    rt.FILTERED_HEAD_DEPTH,
    rt.FILTERED_HAND_DEPTH,
    rt.MASK_RCNN_VIZ_TOPIC,
    rt.IRS_DEPTH,
]

FOUR_CC = cv2.VideoWriter_fourcc(*"MP4V")
FPS = 30


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

    # Define a timeout duration (in seconds)
    TIMEOUT_DURATION = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_seen = {topic: time.time() for topic in self.msgs.keys()}
        self.fps = {topic: deque(maxlen=10) for topic in self.msgs.keys()}

    # Checking for empty image to make an overlay text
    def is_empty_image(self, img):
        """Determine if an image is empty or has no meaningful data"""
        if np.all(img == 0):
            return True
        return False

    def generate_composite(self):
        if not any(self.updated.values()):
            # No imgs were refreshed. Skip.
            return None

        refreshed_topics = [k for k, v in self.updated.items() if v]

        # Gather latest images
        raw_msgs = [self.msgs[i] for i in RAW_IMG_TOPICS]
        processed_msgs = [self.msgs[i] for i in PROCESSED_IMG_TOPICS]

        raw_imgs = [self.msg_to_cv2(i) for i in raw_msgs if i is not None]
        processed_imgs = []
        # Handle processed messages and fill with zeros if needed
        for idx, raw_msg in enumerate(raw_msgs):
            if processed_msgs[idx] is not None:
                processed_imgs.append(self.msg_to_cv2(processed_msgs[idx]))
            else:
                processed_imgs.append(np.zeros_like(raw_imgs[idx]))

        # Crop and process images as needed
        if raw_msgs[1] is not None:
            for imgs in [raw_imgs, processed_imgs]:
                imgs[1] = imgs[1][:, 124:-60]

        # Resizing and lowering the contrast in depth image
        raw_imgs[1] = cv2.convertScaleAbs(raw_imgs[1], alpha=0.03)
        processed_imgs[2] = cv2.resize(processed_imgs[2], (640, 480))
        processed_imgs[3] = cv2.convertScaleAbs(processed_imgs[3], alpha=0.03)

        # Check for topic in list and call is_empty_image() to check whether the image is empty and then overlay text.
        for topic in RAW_IMG_TOPICS:
            if topic in RAW_IMG_TOPICS:
                idx = RAW_IMG_TOPICS.index(topic)
                if idx < len(raw_imgs):
                    if self.is_empty_image(raw_imgs[idx]):
                        print(f"Image for topic {topic} is empty or disconnected.")
                        raw_imgs[idx] = self.overlay_text(
                            raw_imgs[idx],
                            "DISCONNECTED",
                            color=(255, 0, 0),
                            size=2.0,
                            thickness=4,
                        )
        for topic in PROCESSED_IMG_TOPICS:
            if topic in PROCESSED_IMG_TOPICS:
                idx = PROCESSED_IMG_TOPICS.index(topic)
                if idx < len(processed_imgs):
                    if self.is_empty_image(processed_imgs[idx]):
                        print(f"Image for topic {topic} is empty or disconnected.")
                        processed_imgs[idx] = self.overlay_text(
                            processed_imgs[idx],
                            "DISCONNECTED",
                            color=(255, 0, 0),
                            size=2.0,
                            thickness=4,
                        )

        # Overlay topic text
        raw_imgs = [
            self.overlay_topic_text(img, topic)
            for img, topic in zip(raw_imgs, RAW_IMG_TOPICS)
        ]
        processed_imgs = [
            self.overlay_topic_text(img, topic)
            for img, topic in zip(processed_imgs, PROCESSED_IMG_TOPICS)
        ]

        img = np.vstack(
            [
                resize_to_tallest(bgrify_grayscale_imgs(i), hstack=True)
                for i in [raw_imgs, processed_imgs]
            ]
        )

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
            display_img, information_string, color=(255, 0, 0), size=1.5, thickness=4
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

    @staticmethod
    # Method to obtain image, add a white strip on top of the image by resizing it and putting text on that white strip
    def overlay_topic_text(
        img,
        topic,
        box_color=(0, 0, 0),
        text_color=(0, 0, 0),
        font_size=1.3,
        thickness=2,
    ):
        # Original image dimensions
        topic = topic.replace("_", " ").replace("/", "")

        og_height, og_width = img.shape[:2]

        strip_height = 50
        if len(img.shape) == 3:
            white_strip = 255 * np.ones((strip_height, og_width, 3), dtype=np.uint8)
        else:
            white_strip = 255 * np.ones((strip_height, og_width), dtype=np.uint8)

        # Resize the original image height by adding the white strip height
        viz_img = np.vstack((white_strip, img))

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"{topic}"
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_size, thickness)

        margin = 50
        text_x = margin
        text_y = strip_height - margin + text_height
        cv2.putText(
            viz_img,
            text,
            (text_x, text_y),
            font,
            font_size,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

        return viz_img


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
