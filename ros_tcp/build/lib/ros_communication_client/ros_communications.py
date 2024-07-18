from typing import Any, Dict
from .ros_tcp import RosbridgeBSONTCPClient, FPSCounter
import numpy as np, cv2, time
import threading, json, math
from .ros_message_conveter import factory as MessageFactory


class Publisher:
    def __init__(
        self,
        topic_name: str,
        msg_type: str,
        doadvertise: bool = True,
        latch: bool = False,
        queue_size: int = 1,
        host="localhost",
        port=9090,
    ) -> None:
        assert (
            msg_type in MessageFactory
        ), f"{msg_type} couldn't be found in message factory please write your own message conversion adapter"
        self.tcp_connection = RosbridgeBSONTCPClient(host=host, port=port)
        self.tcp_connection.connect()
        self.topic_name = topic_name
        self.pub_id = f"{self.tcp_connection.socket_id}_pub:{self.topic_name}"
        self.msg_type = msg_type
        self.queue_size = queue_size
        self.to_msg_type = MessageFactory[msg_type]["to"]
        self.doadvertise = doadvertise
        self._advertise_id = None if doadvertise else 1
        self.seq: int = 0
        self.latch = latch

    @property
    def is_advertised(self):
        """Indicate if the topic is currently advertised or not.

        Returns:
            bool: True if advertised as publisher of this topic, False otherwise.
        """
        return self._advertise_id is not None

    def recieve(self):
        while self.tcp_connection.connected:
            data = self.tcp_connection.recv_bson()
            print(data)

    def __del__(self):
        print("Destroying Publisher")
        self.unadvertise()
        del self.tcp_connection

    def advertise(self):
        advertis_id = "advertise:%s:%d" % (
            self.topic_name,
            self.tcp_connection.socket_id,
        )
        advertise_msg = {
            "op": "advertise",
            "id": advertis_id,
            "topic": self.topic_name,
            "type": self.msg_type,
            "queue_size": self.queue_size,
            "latch": self.latch,
        }
        self.tcp_connection.send(advertise_msg)
        self._advertise_id = advertis_id

    def unadvertise(self):
        if self.doadvertise and self.is_advertised:
            unadvertise_msg = {
                "op": "unadvertise",
                "id": self.pub_id,
                "topic": self.topic_name,
            }
            self.tcp_connection.send(unadvertise_msg)
            self._advertise_id = None

    def publish(self, *args, **kwargs):
        """
        send python objects that needs to be converted to ROS message
        """
        if not self.is_advertised:
            self.advertise()

        rosmsg = self.to_msg_type(*args, **kwargs)
        if "header" in rosmsg:
            rosmsg["header"]["seq"] = self.seq
            self.seq += 1
        publish_msg = {
            "op": "publish",
            "id": self.pub_id,
            "topic": self.topic_name,
            "latch": self.latch,
            "msg": rosmsg,
        }
        self.tcp_connection.send(publish_msg)


class StaticTransformBroadcaster(Publisher):
    def __init__(self, host="localhost", port=9090):
        super().__init__(
            topic_name="/tf_static",
            msg_type="tf2_msgs/TFMessage",
            latch=True,
            doadvertise=False,
            host=host,
            port=port,
        )

    def send_transform(
        self,
        parent_frame_name: str,
        child_frame_name: str,
        point_in_3d: np.ndarray,
        rotation_quat: np.ndarray,
    ):
        self.publish(parent_frame_name, child_frame_name, point_in_3d, rotation_quat)


class Subscriber:
    def __init__(
        self,
        topic_name: str,
        msg_type: str,
        callback_fn: None,
        host="localhost",
        port=9090,
    ) -> None:
        assert (
            msg_type in MessageFactory
        ), f"{msg_type} couldn't be found in message factory please write your own message conversion adapter"
        self.tcp_connection = RosbridgeBSONTCPClient(host=host, port=port)
        self.tcp_connection.connect()
        self.topic_name = topic_name
        self.sub_id = f"{self.tcp_connection.socket_id}_sub:{self.topic_name}"
        self.msg_type = msg_type
        self.from_msg_type = MessageFactory[msg_type]["from"]
        self.callback_fn = callback_fn
        self.data = None
        self.has_unsubscribed = False
        self.keep_reciving = True
        self.fps_counter = None
        self.recieve_thread = threading.Thread(target=self.recieve)
        self.subscribe()
        self.recieve_thread.start()

    def __del__(self):
        self.unsubscribe()
        del self.tcp_connection

    def recieve(self):
        while self.keep_reciving:
            rosmsg = self.tcp_connection.recv_bson()
            self.fps_counter = (
                FPSCounter() if self.fps_counter is None else self.fps_counter
            )
            # print(f"Data recieved {rosmsg['msg'].keys()}")
            if rosmsg["op"] == "publish" and rosmsg["topic"] == self.topic_name:
                data = self.from_msg_type(rosmsg)
                self.fps_counter.update()
                if self.callback_fn is not None:
                    self.callback_fn(data)
                else:
                    self.data = data

    def subscribe(self):
        subscription_msg = {
            "op": "subscribe",
            "id": self.sub_id,
            "topic": self.topic_name,
            "queue_length": 1,
            "type": self.msg_type,
        }
        self.tcp_connection.send(subscription_msg)

    def unsubscribe(self):
        if not self.has_unsubscribed:
            self.keep_reciving = False
            if self.recieve_thread.is_alive():
                self.recieve_thread.join()
            unsubscribe_msg = {
                "op": "unsubscribe",
                "id": self.sub_id,
                "topic": self.topic_name,
            }
            self.tcp_connection.send(unsubscribe_msg)
            self.has_unsubscribed = True


class TransformListener(Subscriber):
    def __init__(self, host="localhost", port=9090) -> None:
        super().__init__("/tf_static", "tf2_msgs/TFMessage", None, host, port)


class Param:
    _client: RosbridgeBSONTCPClient = None
    _lock = threading.Lock()
    _instance_count = 0
    _set_param_service_name: str = "/rosapi/set_param"
    _get_param_service_name: str = "/rosapi/get_param"

    def init():
        if Param._client is None:
            Param._client = RosbridgeBSONTCPClient()
            Param._client.connect()
        Param._instance_count += 1

    def delete():
        Param._instance_count -= 1
        if Param._instance_count == 0 and Param._client is not None:
            del Param._client
            Param._client = None

    @staticmethod
    def set_param(param_name, value):
        """
        ros param_name like /object_target
        value should be json serializable, no bytearray for now
        """
        with Param._lock:
            Param.init()
            if Param._client is None:
                raise RuntimeError("Connection to ROS bridge is not established.")
            sid = f"{Param._client.socket_id}_call_service:set_param{param_name}"
            param_data = {
                "op": "call_service",
                "id": sid,
                "service": Param._set_param_service_name,
                "args": {"name": f"{param_name}", "value": json.dumps(value)},
            }
            Param._client.send(param_data)
            status_message = Param._client.recv_bson()
            assert (
                status_message["op"] == "service_response"
                and status_message["service"] == Param._set_param_service_name
            )
            assert (
                status_message["id"] == sid
            ), f"Service ID: {sid} sent in set_param service request doens't match with returned service_response {status_message}"
            Param.delete()
            return status_message["result"]

    @staticmethod
    def get_param(param_name, default_value=None):
        with Param._lock:
            Param.init()
            if Param._client is None:
                raise RuntimeError("Connection to ROS bridge is not established.")
            sid = f"{Param._client.socket_id}_call_service:get_param{param_name}"
            param_data = {
                "op": "call_service",
                "id": sid,
                "service": Param._get_param_service_name,
                "args": {"name": f"{param_name}"},
            }
            Param._client.send(param_data)
            response = Param._client.recv_bson()
            assert (
                response["op"] == "service_response"
                and response["service"] == Param._get_param_service_name
            )
            assert (
                response["id"] == sid
            ), f"Service ID: {sid} sent in set_param service request doens't match with returned service_response {response}"
            assert response["result"], f"Couldn't get the value for param {param_name}"
            response = response.get("values", {"value": "null"}).get("value", "null")
            Param.delete()
            return default_value if response == "null" else json.loads(response)


stop_publishing = False


def start_publishing():
    global stop_publishing
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    publisher = Publisher("/new_image_pub", "sensor_msgs/Image")
    while not stop_publishing:
        publisher.publish(image)
    # del(publisher)


def show_image(nparr):
    if nparr.dtype == np.uint16:
        h, w = nparr.shape[:2]
        nparr = nparr.astype(np.float32) / nparr.max()
        nparr = (nparr * 255.0).astype(np.uint8)
        nparr = np.dstack([nparr, nparr, nparr]).reshape((h, w, 3))
    cv2.imshow("Spot Hand RGB", nparr)
    cv2.waitKey(1)


if __name__ == "__main__":
    # start_time = time.time()
    # publisher_thread = threading.Thread(target=start_publishing)
    # publisher_thread.start()

    # subscriber = Subscriber("/new_image_pub", "sensor_msgs/Image", show_image)
    # #time.sleep(0.5)
    # while time.time() - start_time <= 60:
    #      time.sleep(1)
    # subscriber.unsubscribe()
    # stop_publishing = True
    # publisher_thread.join()

    # image = np.zeros((480, 640, 3), dtype=np.uint8)
    # publisher = Publisher("/new_image_pub", "sensor_msgs/Image")
    # #publisher = Publisher("/new_message_pub", "std_msgs/String")
    # i =0
    # while not stop_publishing:
    #     publisher.publish(image)
    #     #publisher.publish(f"Hello: {i}")
    #     i+=1
    #     time.sleep(0.5)
    #     #break
    # is_set = Param.set_param("/skill_name_input", "pick, bottle")
    # print(f'Is the param set ? {is_set}')
    print(Param.get_param("/skill_name_suc_msg", "Not found"))
    # value = "None,None,None"
    # while value != "None,None,None":
    #     value = Param.get_param("/skill_name_suc_msg", value)
    #     print(value)
    #     time.sleep(2)
    # print(f'Get /non_existent : {Param.get_param("/object_target", "my default")}')
    # tf2Broadcaster = StaticTransformBroadcaster()
    # start_time  = time.time()
    # while True:
    #     tf2Broadcaster.send_transform("/spotWorld", "/newFrame", [0., 0., 0.], [0., 0., 0., 1.])
    #     break
    #     #if time.time() - start_time >= 1: break
