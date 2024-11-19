import os
import time

import redis
import zmq

SPOT_IP = os.environ["SPOT_IP"]
PORT = os.environ.get("FAST_DEPTH_PORT", 21998)
REDIS_PORT = os.environ.get("REDIS_PORT", 6379)


class FastDepthZMQSubscriber:
    def __init__(
        self, key_name_in_redis="HandRGBD", ip=None, port=None, redis_client=None
    ):
        self.ip = ip or SPOT_IP
        self.port = port or PORT
        self.socket = None
        self.key_name_in_redis = key_name_in_redis
        self.context = zmq.Context()
        self.redis_client = redis_client or redis.StrictRedis(
            host="localhost", port=REDIS_PORT, db=0
        )
        self.printed_first_time = False
        # Start connecting
        self._connect()

    def _connect(self):
        """Attempt to connect to the ZMQ publisher with retries."""
        while self.socket is None:
            try:
                print("Attempting to connect to the ZMQ publisher...")
                self.socket = self.context.socket(zmq.SUB)
                self.socket.setsockopt_string(
                    zmq.SUBSCRIBE, ""
                )  # Subscribe to all messages
                self.socket.setsockopt(zmq.RCVTIMEO, 1000)  # set timeout to 1 sec
                self.socket.connect(f"tcp://{self.ip}:{self.port}")
                # Check connection by sending a dummy request (with timeout)
                poller = zmq.Poller()
                poller.register(self.socket, zmq.POLLIN)
                if not poller.poll(1000):  # 1000 ms connection timeout
                    print("Connection timeout: could not connect to the socket.")
                    self.socket = None
                if self.socket:
                    print("Connected to ZMQ publisher.")
            except Exception as e:
                print(f"Connection failed: {e}. Retrying in 1 second...")
                self.socket = None
                time.sleep(1)

    def start_receiving(self):
        """Start receiving data and saving it to Redis, with reconnection if needed."""
        while True:
            if self.socket is None:
                # Try reconnecting if the socket was closed
                self._connect()

            try:
                # Receive binary string data
                data = self.socket.recv_pyobj()  # Non-blocking receive
                print(
                    "Data received from ZMQ publisher."
                ) if not self.printed_first_time else None
                self.printed_first_time = True
                data = b"_delimeter_".join(data)
                self.redis_client.set(
                    f"{self.key_name_in_redis}", data, px=100
                )  # Store raw binary string data

            except zmq.Again:
                # No data received; wait briefly before retrying
                print(
                    "No data being recieved, server seems to be off, trying again till we get data"
                )
                time.sleep(0.01)  # Short sleep to avoid busy waiting

            except zmq.ZMQError as e:
                # Connection error; reset socket and attempt reconnect
                print(
                    f"Connection lost while getting data : {e}. Attempting to reconnect..."
                )
                self._connect()
                # self.socket.close()
                # self.socket = None  # Trigger reconnection

            except Exception as e:
                print(f"Unexpected error: {e}")
                break

    def __del__(self):
        self.stop_receiving()

    def stop_receiving(self):
        """Close the socket."""
        if self.socket:
            self.socket.close()
        self.context.term()


# Usage example
if __name__ == "__main__":
    key_name_in_redis = os.environ.get("hand_rgbd_key_name_for_redis", "HandRGBD")
    os.environ["hand_rgbd_key_name_for_redis"] = key_name_in_redis
    subscriber = FastDepthZMQSubscriber(key_name_in_redis)
    subscriber.start_receiving()
