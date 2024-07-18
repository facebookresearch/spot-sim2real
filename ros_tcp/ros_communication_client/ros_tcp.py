import base64
import json
import socket
import struct
import threading
import time
from copy import deepcopy
from typing import Any, Dict, List

import bson
import cv2
import numpy as np

from .ros_message_conveter import from_ros_image, to_ros_image

FRAGMENT_SIZE = 65536


class FPSCounter:
    def __init__(self):
        self.last_time = None
        self.frame_count = 0
        self.fps = None
        self.start_time = time.time()

    def update(self, verbose: bool = False):
        current_time = time.time()
        if self.last_time is not None:
            self.frame_count += 1
            elapsed_time = current_time - self.start_time
            if elapsed_time > 0:
                self.fps = self.frame_count / elapsed_time
                print(f"FPS: {self.fps:.2f}") if verbose else None
        self.last_time = current_time


class RosbridgeBSONTCPClient:
    def __init__(
        self, host="localhost", port=9090, timeoutinsec: int = 60, verbose: bool = False
    ) -> None:
        self.host: str = host
        self.port: int = port
        self.socket = None  # type: ignore
        self.socket_id: int = None
        self.timeoutsec: int = timeoutinsec
        self.connected: bool = False
        self.verbose = verbose

    def __del__(self):
        self.disconnect()

    def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.socket.settimeout(self.timeoutsec)
        self.socket.connect((self.host, self.port))
        self.socket_id = self.socket.fileno()
        self.connected = True
        print("Connected to ROS bridge server.") if self.verbose else None

    def _recvall(self, n):
        # http://stackoverflow.com/questions/17667903/python-socket-receive-large-amount-of-data
        # Helper function to recv n bytes or return None if EOF is hit
        data = bytearray()
        while len(data) < n:
            packet = self.socket.recv(n - len(data))
            data.extend(packet)
        data = bytes(data)
        return data

    def disconnect(self):
        if self.connected:
            self.socket.sendall(
                struct.pack("i", 0)
            )  # Send 0-length message to indicate disconnection
            self.socket_id = None
            self.connected = False
            self.socket.shutdown(socket.SHUT_RDWR)
            self.socket.close()
            print("Disconnected from ROS bridge server.") if self.verbose else None

    def send(self, message: Dict[str, Any]):
        """
        message : python dict
        """
        bson_message = bson.encode(message)
        # print(f"bson_message len {len(bson_message)}, packet len {struct.unpack('i', bson_message[:4])[0]}")
        self.socket.sendall(bson_message)  # type: ignore

    def recv_bson(self):
        if not self.connected:
            raise Exception("Socket disconnected")
        BSON_LENGTH_IN_BYTES = 4
        raw_msglen = self._recvall(BSON_LENGTH_IN_BYTES)
        # print(raw_msglen)
        if not raw_msglen:
            return None
        msglen = struct.unpack("i", raw_msglen)[0]
        # Retrieve the rest of the message
        # print(f"Needs {msglen}Bytes to get full message")
        data = self._recvall(msglen - BSON_LENGTH_IN_BYTES)
        if data is None:
            return None
        data = (
            raw_msglen + data
        )  # Prefix the data with the message length that has already been received.
        # The message length is part of BSONs message format
        return bson.decode(data)


class RosbridgeJSONTCPClient:
    def __init__(self, host="localhost", port=9090):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.receive_thread = None
        self.subscription_topic = None
        self.fps_counter = None
        self.caller_id = None

    def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_id = self.socket.fileno()
        self.socket.connect((self.host, self.port))
        print(self.socket_id)
        self.connected = True
        # self.receive_thread = threading.Thread(target=self.receive)
        # self.receive_thread.start()
        handshake_message = "HANDSHAKE_HEADER"
        self.socket.sendall(handshake_message.encode("utf-8"))
        print("Handshake message sent.")

        # Wait for handshake response (example, may need to adjust based on server protocol)
        response = self.socket.recv(1024)
        if response:
            print("Handshake response received:", response.decode("utf-8"))
        print("Connected to ROS bridge server.")

    def send(self, message, extra_chunk: int = 0):
        if self.connected:
            # chunk_size = FRAGMENT_SIZE+extra_chunk
            # data_len = len(message)
            # total_sent = 0
            # while total_sent < data_len:
            #     #breakpoint()
            #     sent = self.socket.send(message[total_sent:min(total_sent + chunk_size, data_len)])
            #     if sent == 0:
            #         raise RuntimeError("Socket connection broken")
            #     total_sent += sent
            # return 1
            message = json.dumps(message).encode("utf-8")
            bytessent = self.socket.sendall(message)
            self.socket.sendall(b"\r\n")
            print(f"Bytes sent {bytessent}, passed message bytes {len(message)}")

    def receive_once(self):
        buffer = ""
        remaining_data = ""
        while self.connected:
            try:
                data = remaining_data + self.socket.recv(FRAGMENT_SIZE + 115).decode(
                    "utf-8"
                )
                remaining_data = ""
                print("Raw data", data)
                if not data:
                    break
                if "}{" in data:
                    index_of_proper_data_end = data.index("}{") + 1
                    remaining_data = data[index_of_proper_data_end:]
                    data = data[:index_of_proper_data_end]
                data = json.loads(data)
                if "fragment" in data:
                    # print(f"Data fragment number {int(data_fragment['num'])}, Total {data_fragment['total']}")
                    buffer += data["data"]
                    if int(data["num"]) == int(data["total"]) - 1:
                        return json.loads(buffer)
                else:
                    return data
            except Exception as e:
                print(f"Receive error: {e}")
                print(data, remaining_data)
                self.connected = False

    def receive(self):
        buffer = ""
        remaining_data = ""
        self.fps_counter = FPSCounter()
        while self.connected:
            try:
                data = remaining_data + self.socket.recv(FRAGMENT_SIZE + 115).decode(
                    "utf-8"
                )
                remaining_data = ""
                print("Raw data", data)
                # continue
                if not data:
                    break
                if "}{" in data:
                    index_of_proper_data_end = data.index("}{") + 1
                    remaining_data = data[index_of_proper_data_end:]
                    data = data[:index_of_proper_data_end]
                if "fragment" in data:

                    data_fragment = json.loads(data)

                    # print(f"Data fragment number {int(data_fragment['num'])}, Total {data_fragment['total']}")
                    buffer += data_fragment["data"]
                    if int(data_fragment["num"]) == int(data_fragment["total"]) - 1:
                        buffer = json.loads(buffer)
                        self.fps_counter.update() if self.fps_counter else None
                        self.handle_message(buffer)
                        buffer = ""
                else:
                    buffer = json.loads(data)
                    self.handle_message(buffer)
                    buffer = ""
            except Exception as e:
                print(f"Receive error: {e}")
                print(data, remaining_data)
                self.connected = False
                break
        self.fps_counter = None
        print("Reciving thread ended")

    def handle_message(self, message):
        # pass
        if "id" in message and message["op"] == "service_response":
            if message["id"] in self.caller_id:
                self.caller_id[message["id"]](message["values"])
        elif message["op"] == "publish":

            nparr = from_ros_image(message)
            if nparr.dtype == np.uint16:
                h, w = nparr.shape[:2]
                nparr = nparr.astype(np.float32) / nparr.max()
                nparr = (nparr * 255.0).astype(np.uint8)
                nparr = np.dstack([nparr, nparr, nparr]).reshape((h, w, 3))
            self.fps_counter.update()
            cv2.imshow("Spot Hand RGB", nparr)
            cv2.waitKey(1)
        # print(f'Received message: {message["msg"].keys()}')

    def unsubscribe(self, topic):
        self.send({"op": "unsubscribe", "topic": topic})

    def get_param(self, param_name: str):
        caller_id = f"{self.socket_id}:call_service:{param_name}"
        self.send(
            {
                "op": "call_service",
                "id": caller_id,
                "service": "/rosapi/get_param",
                "fragment_size": FRAGMENT_SIZE,
                "args": {"name": str(param_name)},
            }
        )

        def call_back(data):
            print(data)

        self.caller_id = {f"{caller_id}": call_back}
        # self.recieve_once()

    def subscribe(self, topic: str, msg_type: str):
        self.send(
            {
                "op": "subscribe",
                "id": f"{self.socket_id}:subscribe:{topic}",
                "topic": topic,
                "fragment_size": FRAGMENT_SIZE,
                "queue_length": 1,
                "type": msg_type,
            }
        )
        self.subscription_topic = topic

    def publish(self, topic, msg):
        self.send({"op": "publish", "topic": topic, "msg": msg})

    def disconnect(self):
        try:
            # Shutdown the socket to stop sending and receiving data
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.socket.shutdown(socket.SHUT_RDWR)
        except socket.error as e:
            print(f"Error shutting down socket: {e}")
        finally:
            try:
                # Close the socket
                self.socket.close()
            except socket.error as e:
                print(f"Error closing socket: {e}")
        self.connected = False
        print("Disconnected from ROS bridge server.")


if __name__ == "__main__":
    client = RosbridgeBSONTCPClient()
    client.connect()
    get_param_msg = {
        "op": "call_service",
        "id": "someid",
        "service": "/rosapi/set_param",
        "args": {"name": "/object_target", "value": json.dumps({"x": 0.0, "y": 0.0})},
    }
    client.send(get_param_msg)
    print(client.recv_bson())
    # client.disconnect()
    # print(client.recv_bson())
    # client.disconnect()
