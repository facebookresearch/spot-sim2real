import os

os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/Cellar/ffmpeg/6.1.1_3/bin/ffmpeg"
import time  # noqa: E402

import cv2  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import mediapipe as mp  # noqa: E402
import moviepy.editor as moviepy  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLOWorld  # noqa: E402

# Threshold for number of times that hand point is inside bbox
THRESHOLD_HAND_IN_BBOX = 1
MAX_FRAMES = float("inf")


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

    def prediction(self, video_path="", class_to_detect="toy plush"):

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
    model.prediction(
        "/Users/jimmytyyang/Downloads/hand_interaction_with_bottle.mp4", "bottle"
    )
    model.prediction(
        "/Users/jimmytyyang/Downloads/hand_interaction_with_can.mp4", "can"
    )
    model.prediction(
        "/Users/jimmytyyang/Downloads/hand_interaction_with_toy_plush.mp4", "toy_plush"
    )
    model.prediction(
        "/Users/jimmytyyang/Downloads/hand_interaction_with_cup.mp4", "cup"
    )
