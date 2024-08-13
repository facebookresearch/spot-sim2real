import argparse
import json
import time

import numpy as np
from siro_robot_connection.siro_pub_sub import siro_publisher, siro_subscriber


class EEPosePublisher(siro_publisher):
    def __init__(self, file_name, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Open the file and load its contents as a Python object
        # The ee pose data is a list of list: List[List].
        # The element being robot gripper's x, y, z, relative to
        # the robot base (body).
        # +x is the robot's front;
        # +y is the robot's left;
        # +z is the robot's top
        with open(file_name, "r") as file:
            self.data = json.load(file)

        self.counter = 0

    def load_data(self):
        """load data into python object"""
        msg = ",".join([str(v) for v in self.data[self.counter]])
        self.counter += 1
        if self.counter == len(self.data):
            self.counter = 0
        time.sleep(0.1)  # in the real robot. this is 10Hz
        return msg


class RequestEEPose(siro_subscriber):
    def request_ee_pose(self):
        """ "Format the msg"""
        return self.get_msg()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pub-mode", action="store_true")
    args = parser.parse_args()
    topic = "ee_pose_topic"
    file_name = "/home/jmmy/research/Nexus_data/ee_pose.json"

    if args.pub_mode:
        print("Start to publish msg")
        siro_pub = EEPosePublisher(file_name=file_name, topic=topic)
        siro_pub.publish_msg()
    else:
        print("Start to subscribe msg")
        request_ee_pose = RequestEEPose(topic=topic)
        while True:
            print(request_ee_pose.request_ee_pose())
