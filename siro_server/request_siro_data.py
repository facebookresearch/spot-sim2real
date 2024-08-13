import argparse

import numpy as np
from siro_robot_connection.siro_pub_sub import siro_publisher, siro_subscriber


class EEPosePublisher(siro_publisher):
    def load_data(self):
        """load data into python object"""
        return np.random.randint(10)


class RequestEEPose(siro_subscriber):
    def request_ee_pose(self):
        """ "Format the msg"""
        return self.get_msg()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pub-mode", action="store_true")
    args = parser.parse_args()
    topic = "ee_pose"

    if args.pub_mode:
        print("Start to publish msg")
        siro_pub = EEPosePublisher(topic=topic)
        siro_pub.publish_msg()
    else:
        print("Start to subscribe msg")
        request_ee_pose = RequestEEPose(topic=topic)
        while True:
            print(request_ee_pose.request_ee_pose())
