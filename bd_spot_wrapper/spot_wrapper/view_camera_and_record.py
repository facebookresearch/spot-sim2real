# mypy: ignore-errors
import argparse
import time
from collections import deque

import cv2
import numpy as np
from spot_wrapper.spot import (
    Spot,
    SpotCamIds,
    draw_crosshair,
    image_response_to_cv2,
    scale_depth_img,
)
from spot_wrapper.utils import color_bbox, resize_to_tallest

MAX_HAND_DEPTH = 3.0
MAX_HEAD_DEPTH = 10.0
DETECT_LARGEST_WHITE_OBJECT = False

FOUR_CC = cv2.VideoWriter_fourcc(*"MP4V")
FPS = 30


def main(spot: Spot):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--no-display", action="store_true")
    parser.add_argument("-q", "--quality", type=int)
    args = parser.parse_args()
    window_name = "Spot Camera Viewer"
    time_buffer = deque(maxlen=10)
    sources = [
        # SpotCamIds.FRONTRIGHT_DEPTH,
        # SpotCamIds.FRONTLEFT_DEPTH,
        # SpotCamIds.HAND_DEPTH,
        SpotCamIds.HAND_COLOR,
    ]
    try:
        all_imgs = []
        k = 0
        while True:
            start_time = time.time()

            # Get Spot camera image
            image_responses = spot.get_image_responses(sources, quality=args.quality)
            imgs = []
            for image_response, source in zip(image_responses, sources):
                img = image_response_to_cv2(image_response, reorient=True)
                if "depth" in source:
                    max_depth = MAX_HAND_DEPTH if "hand" in source else MAX_HEAD_DEPTH
                    img = scale_depth_img(img, max_depth=max_depth, as_img=True)
                elif source is SpotCamIds.HAND_COLOR:
                    # img = draw_crosshair(img)
                    if DETECT_LARGEST_WHITE_OBJECT:
                        x, y, w, h = color_bbox(img, just_get_bbox=True)
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                imgs.append(img)
                all_imgs.append(img)
                w, h = img.shape[:2]

            # Make sure all imgs are same height
            img = resize_to_tallest(imgs, hstack=True)

            if not args.no_display:
                cv2.imshow(window_name, img)
                cv2.waitKey(1)

            if k % 10 == 0:
                cv2.imwrite(f"videos_april/img_{k}.jpg", img)

            # if k % 50 == 0 and k > 0:
            #    new_video = cv2.VideoWriter(
            #        f'videos_april/video_{video_index}.mp4',
            #        -1,
            #        FPS,
            #        (w, h),
            #    )
            #    for img in all_imgs:
            #        new_video.write(img)

            #    new_video.release()
            #    all_imgs = []
            #    break

            k += 1
            time_buffer.append(time.time() - start_time)
            print("Avg FPS:", 1 / np.mean(time_buffer))
    finally:
        if not args.no_display:
            cv2.destroyWindow(window_name)


if __name__ == "__main__":
    spot = Spot("ViewCamera")
    # We don't need a lease because we're passively observing images (no motor ctrl)
    main(spot)
