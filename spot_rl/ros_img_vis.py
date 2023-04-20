import argparse
import os
import os.path as osp
import time
from collections import deque

import cv2
import numpy as np
import rospy
import tqdm
from spot_wrapper.utils import resize_to_tallest

from spot_rl.utils.robot_subscriber import SpotRobotSubscriberMixin
from spot_rl.utils.utils import ros_topics as rt

RAW_IMG_TOPICS = [rt.HEAD_DEPTH, rt.HAND_DEPTH, rt.HAND_RGB]

PROCESSED_IMG_TOPICS = [
    rt.FILTERED_HEAD_DEPTH,
    rt.FILTERED_HAND_DEPTH,
    #rt.MASK_RCNN_VIZ_TOPIC,
    rt.OWLVIT_VIZ_TOPIC,
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

    def generate_composite(self):
        raise NotImplementedError

    @staticmethod
    def overlay_text(img, text, color=(0, 0, 255)):
        viz_img = img.copy()
        line, font, font_size, font_thickness = (text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)
        text_width, text_height = cv2.getTextSize(
            line, font, font_size, font_thickness
        )[0][:2]
        height, width = img.shape[:2]
        x = (width - text_width) // 2
        y = (height - text_height) // 2
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

    def vis_imgs(self):
        # Skip if no messages were updated
        currently_saving = not self.recording and self.frames
        img = self.generate_composite() if not currently_saving else None
        if not self.headless:
            if img is not None:
                if self.recording:
                    viz_img = self.overlay_text(img, "RECORDING IS ON!")
                    cv2.imshow("ROS Spot Images", viz_img)
                else:
                    cv2.imshow("ROS Spot Images", img)

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

        img = np.vstack(
            [
                resize_to_tallest(bgrify_grayscale_imgs(i), hstack=True)
                for i in [raw_imgs, processed_imgs]
            ]
        )

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
            except:
                print("Deleting unfinished videos")
                srv.delete_videos()
                exit()
        raise e


if __name__ == "__main__":
    main()
