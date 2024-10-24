import time
from typing import List

import zmq


class YOLOWorld:
    def __init__(self, port: str = "21001"):
        self.port = port
        self.socket = None
        self.labels: List[str] = []
        self._connect_socket()

    def _connect_socket(self):
        if self.socket is None:
            context = zmq.Context()
            self.socket = context.socket(zmq.REQ)
            self.socket.setsockopt(zmq.RCVTIMEO, 500)
            self.socket.connect(f"tcp://localhost:{self.port}")
            print(f"Socket Connected to yolo world service at {self.port}")

    def update_label(self, labels: List[List[str]]):
        assert (
            len(labels) == 1
        ), f"len of labels was sent more than 1 found {len(labels)}"
        self.labels = labels[0]

    # def inference(self, rgb_image, )
    def run_inference_and_return_img(self, rgb_image):
        self.socket.send_pyobj((rgb_image, self.labels, True))
        bboxes, probs, labels, visualization_img = self.socket.recv_pyobj()

        results = []
        for bbox, score, label in zip(bboxes, probs, labels):
            results.append([label, score, bbox])

        return results, visualization_img
