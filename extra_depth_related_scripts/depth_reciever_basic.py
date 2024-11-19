import zmq, time
import numpy as np
import cv2
import os
from bosdyn.api import image_pb2
SPOT_IP = os.environ["SPOT_IP"]

def image_response_to_cv2(image_response, reorient=True):
    if (
        image_response.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16
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

if __name__ == "__main__":
    port = "21998"
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://{SPOT_IP}:{port}")
    # Subscribe to all messages (empty filter subscribes to everything)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    print(f"Depth Server subscribed at {port}")
    
    start_time = None
    n = 0
    prev_frame_time = 0
  
    # used to record the time at which we processed current frame 
    new_frame_time = 0
    while True:
        start_time = time.time() if start_time is None else start_time
        image_responses_serialized, capture_time = socket.recv_pyobj()
        image_responses = [image_pb2.ImageResponse() for _ in range(len(image_responses_serialized))]
        for image_response, image_response_serialized in zip(image_responses, image_responses_serialized):
            image_response.ParseFromString(image_response_serialized)
        #print(f"Recieved data", encoded_image_bytes)
        images = [image_response_to_cv2(img) for img in image_responses]
        #print(f"Intrinsics {image_responses[0].source.pinhole.intrinsics}, Transform Tree {image_responses[0].shot.transforms_snapshot}")
        # encoded_image = np.frombuffer(encoded_image_bytes, dtype=np.uint8)
        # depth_image = cv2.imdecode(encoded_image, cv2.IMREAD_UNCHANGED)
        #print(images[-1].shape, images[1].dtype)
        n+=1
        new_frame_time = time.time() 
        fps = 1/(new_frame_time-prev_frame_time) 
        prev_frame_time = new_frame_time 
        #socket.send_pyobj("")
        print(f"FPS {fps}")
