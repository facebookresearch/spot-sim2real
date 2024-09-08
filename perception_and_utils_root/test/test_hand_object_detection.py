import os

os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/Cellar/ffmpeg/6.1.1_3/bin/ffmpeg"
import sys  # noqa: E402

sys.path.append("/Users/jimmytyyang/research/spot-sim2real/spot_rl_experiments")
import time  # noqa: E402

import cv2  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import mediapipe as mp  # noqa: E402
import moviepy.editor as moviepy  # noqa: E402
import numpy as np  # noqa: E402
from spot_rl.utils.tracking_service import tracking_with_socket  # noqa: E402
from ultralytics import YOLOWorld  # noqa: E402

# Threshold for number of times that hand point is inside bbox
THRESHOLD_HAND_IN_BBOX = 1
MAX_FRAMES = float("inf")
ENABLE_SAM2_TRACKING = True


class HandObjectDetection:
    def __init__(self):
        # Load a pretrained mediapipe hand model
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.15,
            min_tracking_confidence=0.05,
        )
        self.mpDraw = mp.solutions.drawing_utils

        # Load a pretrained YOLO model
        self.object_detection_model = YOLOWorld("yolov8x-worldv2.pt")

        # Reset the hand tracking parameters
        self._reset_tracking_params()

    def _reset_tracking_params(self):
        self._hand_tracking_first_time = (
            True  # flag to indicate if it is the first time tracking
        )
        self._hand_tracking_images = []  # cache the tracking images
        self._hand_tracking_bboxs = []  # cache the tracking bboxs

    def hand_in_bbox(self, xyxy, hand_pixel_location):
        # Check if the hand is inside the bounding box
        num_of_times_that_hand_point_is_inside_bbox = 0
        for hand_pixel in hand_pixel_location:
            if (
                hand_pixel[0] > xyxy[0]
                and hand_pixel[0] < xyxy[2]
                and hand_pixel[1] > xyxy[1]
                and hand_pixel[1] < xyxy[3]
            ):
                num_of_times_that_hand_point_is_inside_bbox += 1
        return num_of_times_that_hand_point_is_inside_bbox

    def _get_bbox_from_cxcys(self, hand_pixel_location):
        # Get the bounding box from cx cy pairs
        cxs = [hand_pixel[0] for hand_pixel in hand_pixel_location]
        cys = [hand_pixel[1] for hand_pixel in hand_pixel_location]
        return min(cxs), min(cys), max(cxs), max(cys)

    def prediction_from_video(self, video_path="", class_to_detect="toy plush"):

        # Reset hand tracking parameters
        self._reset_tracking_params()

        # Set the classes to detect
        self.object_detection_model.set_classes([class_to_detect])

        # Load a video
        cap = cv2.VideoCapture(video_path)
        vis_list = []

        frame_count = 0
        # Loop through the video
        while cap.isOpened():
            ret, img = cap.read()

            # check if the video is finished
            if img is None:
                break

            if frame_count > MAX_FRAMES:
                break

            print(f"Frame: {frame_count}")
            frame_count += 1

            # Copy the image to avoid changing the original image
            origin_img = img.copy()

            # Start tracking using video segmentation model
            images = np.expand_dims(img, axis=0)  # The size should (1, H, W, C)
            self._hand_tracking_images.append(images)

            # Hand detection timer
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            start_time = time.time()
            results = self.hands.process(imgRGB)
            end_time = time.time()

            hand_pixel_location = []
            # Draw the hand landmarks if detected
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    for id, lm in enumerate(handLms.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
                        hand_pixel_location.append([cx, cy])
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )

            if ENABLE_SAM2_TRACKING:
                # Process hand tracking
                if (
                    len(self._hand_tracking_images) == 1
                    and self._hand_tracking_first_time
                ):
                    # To see if there is a bounding box in the first frame
                    self._hand_tracking_first_time = False
                    if hand_pixel_location == []:
                        # Do not see the hand yet, so restart the detection
                        self._reset_tracking_params()
                    else:
                        # See the hand at the first frame
                        self._hand_tracking_bboxs.append(
                            self._get_bbox_from_cxcys(hand_pixel_location)
                        )
                elif len(self._hand_tracking_images) >= 2:
                    # This means that there is one hand being detected in the previous frame (first frame)
                    input_images = np.concatenate(self._hand_tracking_images, axis=0)
                    # Get the previous anchor
                    bbox = self._hand_tracking_bboxs[-1]
                    cur_bbox = tracking_with_socket(input_images, bbox)
                    if cur_bbox is None:
                        self._hand_tracking_bboxs.append(None)
                    else:
                        cur_bbox = [cur_bbox[1], cur_bbox[0], cur_bbox[3], cur_bbox[2]]
                        self._hand_tracking_bboxs.append(cur_bbox)
                        # Apply the red bounding box to showcase tracking on the image
                        img = cv2.rectangle(
                            img, cur_bbox[:2], cur_bbox[2:], (0, 0, 255), 5
                        )
                if len(self._hand_tracking_images) >= 2:
                    # Prune the data
                    # Find the frame that has tracking
                    suc_frame = []
                    for i, v in enumerate(self._hand_tracking_bboxs):
                        if v is not None:
                            suc_frame.append(i)
                    # Only keep the latest frame that has tracking to keep
                    # buffer size 2
                    self._hand_tracking_images = [
                        self._hand_tracking_images[suc_frame[-1]]
                    ]
                    self._hand_tracking_bboxs = [
                        self._hand_tracking_bboxs[suc_frame[-1]]
                    ]

            # Put the FPS on the image
            fps = 1 / (end_time - start_time)
            cv2.putText(
                img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3
            )

            # Object prediction
            num_of_times_that_hand_point_is_inside_bbox = False
            results = self.object_detection_model.predict(origin_img, stream=False)
            for _, re in enumerate(results):
                img = re.plot(img=img)
                if re.boxes.xyxy.shape[0] > 0:
                    xyxy = re.boxes.xyxy.cpu().numpy().tolist()[0]
                    num_of_times_that_hand_point_is_inside_bbox = (
                        num_of_times_that_hand_point_is_inside_bbox
                        or self.hand_in_bbox(xyxy, hand_pixel_location)
                        >= THRESHOLD_HAND_IN_BBOX
                    )

            # Put the hand touches object flag on the image
            if num_of_times_that_hand_point_is_inside_bbox:
                cv2.putText(
                    img, "hold", (390, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3
                )
            else:
                cv2.putText(
                    img, "unhold", (390, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3
                )

            # Concatenate the original image and the image with hand landmarks and object prediction
            vis = np.concatenate((origin_img, img), axis=1)
            vis_list.append(vis)

        # Generate a video from the concatenated images
        height, width, layers = vis.shape
        video = cv2.VideoWriter(
            "output_video.avi", cv2.VideoWriter_fourcc(*"XVID"), 30, (width, height)
        )
        for vis in vis_list:
            video.write(vis)
        cv2.destroyAllWindows()
        video.release()

        # Convert the video to mp4
        clip = moviepy.VideoFileClip("output_video.avi")
        clip.write_videofile(f"{video_path[0:-4]}_detection.mp4")


if __name__ == "__main__":
    model = HandObjectDetection()
    model.prediction_from_video(
        "/Users/jimmytyyang/Downloads/hand_interaction_with_bottle.mp4", "bottle"
    )
    model.prediction_from_video(
        "/Users/jimmytyyang/Downloads/hand_interaction_with_can.mp4", "can"
    )
    model.prediction_from_video(
        "/Users/jimmytyyang/Downloads/hand_interaction_with_toy_plush.mp4", "toy_plush"
    )
    model.prediction_from_video(
        "/Users/jimmytyyang/Downloads/hand_interaction_with_cup.mp4", "cup"
    )
