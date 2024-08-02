import threading
import time

import cv2
import numpy as np
import ros_communication_client as ros

stop_publishing = False


def show_image(nparr):
    if nparr.dtype == np.uint16:
        h, w = nparr.shape[:2]
        nparr = nparr.astype(np.float32) / nparr.max()
        nparr = (nparr * 255.0).astype(np.uint8)
        nparr = np.dstack([nparr, nparr, nparr]).reshape((h, w, 3))
    if nparr.dtype == np.float32:
        nparr = (255.0 * nparr).astype(np.uint8)
    cv2.imshow("Spot Hand RGB", nparr)
    cv2.waitKey(1)


def start_publishing():
    global stop_publishing
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    publisher = ros.Publisher(
        "/new_image_pub", "sensor_msgs/Image", verbose=True
    )  # std_msgs/Float32MultiArray "sensor_msgs/Image"
    while not stop_publishing:
        publisher.publish(image)
        break
    # del(publisher)


if __name__ == "__main__":
    start_time = time.time()
    # start_publishing()
    # stop_publishing = True

    # publisher_thread = threading.Thread(target=start_publishing)
    # publisher_thread.start()

    subscriber = ros.Subscriber(
        "/mask_rcnn_visualizations",
        "sensor_msgs/Image",
        callback_fn=show_image,
        verbose=True,
    )
    # time.sleep(0.5)
    while time.time() - start_time <= 2:
        #      #print(subscriber.data)
        time.sleep(1)
    subscriber.unsubscribe()
    # stop_publishing = True
    # publisher_thread.join()
