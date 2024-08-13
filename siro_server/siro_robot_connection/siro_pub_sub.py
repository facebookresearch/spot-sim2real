import argparse

import numpy as np
import zmq


class siro_publisher:
    def __init__(self, ip="tcp://*:5555", topic="siro_topic"):
        # Create a ZeroMQ context
        context = zmq.Context()

        # Create a PUB socket
        self.socket = context.socket(zmq.PUB)

        # Bind the socket to an address
        self.socket.bind(ip)

        # topic
        self.topic = topic

        # msg
        self._msg = ""

    def load_data(self):
        """load data into python object"""
        return NotImplementedError

    def process_msg(self):
        """ "Format the msg"""
        self._msg = self.load_data()

    def publish_msg(self):
        """Publish msg"""

        while True:
            # Send the message with the topic
            self.process_msg()
            self.socket.send_string(f"{self.topic} {self._msg}")


class siro_subscriber:
    def __init__(self, ip="tcp://localhost:5555", topic="siro_topic"):
        # Create a ZeroMQ context
        context = zmq.Context()

        # Create a PUB socket
        self.socket = context.socket(zmq.SUB)

        # Bind the socket to an address
        self.socket.connect(ip)

        # topic
        self.topic = topic

        # Subscribe to a specific topic
        self.socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)

    def get_msg(self):
        message = self.socket.recv_string()
        # The format is f"{topic name} {message}" with space in the middle
        return message
