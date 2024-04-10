import time

import rospy
from spot_rl.utils.utils import ros_topics as rt
from std_msgs.msg import String


class DetectionSubscriber:
    """
    This is a rospy subscriber which is supposed to act like a socket, it doesn't continously fetch the data,
    Its purpose is to connect to OwlVit publisher, fetch detection string as needed & then be destroyed
    """

    def __init__(self):
        self.latest_message = None
        rospy.Subscriber(rt.DETECTIONS_TOPIC, String, self.callback)

    def callback(self, data):
        self.latest_message = data.data

    def get_latest_message(self):
        return self.latest_message


def detect_with_rospy_subscriber(object_name, image_scale=0.7):
    """Fetch the detection result, creates subscriber object, fetches message & then deletes the subscriber"""
    # We use rospy approach reac the detection string from topic
    rospy.set_param("object_target", object_name)
    subscriber = DetectionSubscriber()
    fetch_time_threshold = 1.0
    time.sleep(1.0)
    begin_time = time.time()
    while (time.time() - begin_time) < fetch_time_threshold:
        latest_message = subscriber.get_latest_message()
        if "None" in latest_message:
            continue
        try:
            bbox_str = latest_message.split(",")[-4:]
            break
        except Exception:
            pass

    prediction = [int(float(num) / image_scale) for num in bbox_str]
    return prediction
