import os

import redis
from bosdyn.api import image_pb2

REDIS_PORT = os.environ.get("REDIS_PORT", 6379)
hand_rgbd_key_name_in_redis = os.environ.get("hand_rgbd_key_name_for_redis", "HandRGBD")


class RedisClient:
    def __init__(self, host="localhost", port=None, db=0):
        port = port or REDIS_PORT
        self.redis_client = redis.StrictRedis(host=host, port=port, db=db)
        print("Redis connected")

    def get_latest_hand_rgbd(self):
        """Retrieve the latest data from Redis and check if it is recent."""
        # Get the serialized data from Redis
        latest_data = self.redis_client.get(f"{hand_rgbd_key_name_in_redis}")
        if not latest_data:
            return False  # No data available in Redis
        image_responses_serialized = latest_data.split(b"_delimeter_")
        image_responses = []
        for image_response_serialized in image_responses_serialized:
            image_response = image_pb2.ImageResponse()
            image_response.ParseFromString(image_response_serialized)
            image_responses.append(image_response)
        return image_responses


if __name__ == "__main__":
    import time

    def image_response_to_cv2(image_response, reorient=True):
        import cv2
        import numpy as np

        if (
            image_response.shot.image.pixel_format
            == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16
            and image_response.shot.image.format == image_pb2.Image.FORMAT_RAW
        ):
            dtype = np.uint16
        else:
            dtype = np.uint8

        # img = np.fromstring(image_response.shot.image.data, dtype=dtype)
        img = np.frombuffer(image_response.shot.image.data, dtype=dtype)
        if image_response.shot.image.format == image_pb2.Image.FORMAT_RAW:
            img = img.reshape(
                image_response.shot.image.rows, image_response.shot.image.cols
            )
        else:
            img = cv2.imdecode(img, -1)

        return img

    redisclientforspot = RedisClient()

    prev_frame_time, new_frame_time = 0, 0

    while True:
        image_responses = redisclientforspot.get_latest_hand_rgbd()
        if image_responses:
            # breakpoint()
            images = [
                image_response_to_cv2(image_response)
                for image_response in image_responses
            ]
        else:
            raise Exception("data doesn't exist yet or expired")
        new_frame_time = time.time()  # type: ignore
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        print(f"FPS {fps}")
        time.sleep(0.1)
