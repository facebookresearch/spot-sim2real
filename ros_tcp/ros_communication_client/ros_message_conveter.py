import base64
import json
import time
from typing import Any, Dict, List

import cv2
import numpy as np
from bson import Binary
from scipy.spatial.transform import Rotation as R


class Time:
    @staticmethod
    def now():
        current_time = time.time()
        secs = int(current_time)
        nsecs = int((current_time - secs) * 1e9)
        return {"secs": secs, "nsecs": nsecs}


def from_ros_image(message: Dict[str, Any]):
    data = message["msg"]
    # For debug
    # print("Keys", [data[key] for key in data.keys() if key != "data"])
    height, width = int(data["height"]), int(data["width"])
    encoding = data["encoding"]
    dtype = np.uint8 if encoding[-1] == "8" else np.uint16
    nparr = np.frombuffer(data["data"], dtype)
    nparr = nparr.reshape((height, width, -1))
    if nparr.shape[-1] == 1:
        nparr = nparr.reshape((height, width))
    elif "rgb" in encoding:
        nparr = cv2.cvtColor(nparr, cv2.COLOR_RGB2BGR)
    return {"data": nparr, "header": data["header"]}


def to_ros_image(nparr_bgr: np.ndarray):
    assert (
        nparr_bgr.dtype == np.uint8 or nparr_bgr.dtype == np.uint16
    ), f"Expected uint8 or uint16 found {nparr_bgr.dtype}"
    channel = 3 if len(nparr_bgr.shape) == 3 else 1
    if nparr_bgr.dtype == np.uint8:
        encoding = "bgr8" if channel == 3 else "mono8"
        bytes_per_pixel = 3
    else:
        encoding = "mono16"
        bytes_per_pixel = 2

    height, width = nparr_bgr.shape[:2]
    image_bytes = nparr_bgr.flatten().tobytes()

    msg = {
        "header": {"seq": 0, "stamp": Time.now(), "frame_id": ""},
        "height": height,
        "width": width,
        "encoding": encoding,
        "is_bigendian": 0,
        "step": width * bytes_per_pixel,
        "data": Binary(image_bytes),
    }
    return msg


def from_ros_transforms(message: Dict[str, Any]):
    data = message["msg"]
    transforms = data["transforms"]
    transforms_dict = {}
    for transform in transforms:
        a_frame = transform["header"]["frame_id"]
        b_frame = transform["child_frame_id"]
        translation, rotation = (
            transform["transform"]["translation"],
            transform["transform"]["rotation"],
        )
        translation = [
            float(translation["x"]),
            float(translation["y"]),
            float(translation["z"]),
        ]
        rotation = [
            float(rotation["x"]),
            float(rotation["y"]),
            float(rotation["z"]),
            float(rotation["w"]),
        ]
        rotation = R.from_quat(rotation).as_matrix()
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = rotation
        T[:3, 3] = translation
        transforms_dict.update({f"{a_frame}_T_{b_frame}": T})
    return transforms_dict


def to_ros_transforms(
    parent_frame_name: str,
    child_frame_name: str,
    point_in_3d: np.ndarray,
    rotation_quat: np.ndarray,
) -> Dict[str, Any]:
    """
    point_in_3d should be [x, y, z]
    rotation_quat should be [x, y, z, w]
    """
    data = {
        "transforms": [
            {
                "header": {
                    "seq": 0,
                    "stamp": Time.now(),
                    "frame_id": parent_frame_name,
                },
                "child_frame_id": child_frame_name,
                "transform": {
                    "translation": {
                        "x": point_in_3d[0],
                        "y": point_in_3d[1],
                        "z": point_in_3d[-1],
                    },
                    "rotation": {
                        "x": rotation_quat[0],
                        "y": rotation_quat[1],
                        "z": rotation_quat[2],
                        "w": rotation_quat[3],
                    },
                },
            }
        ]
    }

    return data


def to_ros_string(string: str):
    return {"data": string}


def from_ros_string(message: Dict[str, Any]):
    return str(message["msg"]["data"])


def to_ros_Float32MultiArray(
    np_array: np.ndarray, dim_labels: List[str] = [], data_offset: int = 0
):
    assert (
        type(np_array) == np.ndarray
    ), f"Expected numpy.ndarray object got {type(np_array)}"
    assert (
        np_array.dtype == np.float32
    ), f"expected numpy array with np.float32 datatype got {np_array.dtype}"
    shape = np_array.shape
    ndim = len(shape)
    if len(dim_labels):
        assert (
            len(dim_labels) == ndim
        ), f"Expected dim_labels to be List[str] of length {ndim}, got {len(dim_labels)}"
    else:
        dim_labels = [f"{i}thdim" for i in range(ndim)]

    layout = {"dim": list(), "data_offset": data_offset}
    if len(shape) > 1:
        for i, dimsize in enumerate(shape):
            layout["dim"].append(  # type: ignore
                {
                    "label": dim_labels[i],
                    "size": dimsize,
                    "stride": int(np.prod(shape[i:])),
                }
            )
    data = np_array.flatten().tolist()
    return {"layout": layout, "data": data}


def from_ros_Float32MultiArray(msg):
    msg = msg["msg"]
    layout = msg["layout"]
    data_points = msg["data"]
    data_offset = int(layout["data_offset"])
    np_array = np.array(data_points[data_offset:], dtype=np.float32)
    shape = [int(dim["size"]) for dim in layout["dim"]]
    return np_array.reshape(shape) if shape else np_array


def to_ros_Marker(data):
    """
    Convert messages from dict to ros visulization_msgs/Marker
    More info - http://docs.ros.org/en/noetic/api/visualization_msgs/html/msg/Marker.html
    """
    # quaternion = quaternion_from_euler(0.0, 0.0, 0.0)
    message = {
        "header": {
            "frame_id": "spotWorld",
            "stamp": Time.now(),
            "seq": int(0),
        },
        "ns": "",
        "id": int(data.get("id")),
        "type": int(data.get("type")),
        "action": int(0),
        "pose": {
            "position": data.get("position"),
            "orientation": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "w": 1.0,
            },
        },
        "scale": {
            "x": float(0.1),
            "y": float(0.1),
            "z": float(0.1),
        },
        "color": data.get("color"),
        "lifetime": float(0.0),
        "frame_locked": bool(False),
    }

    if message.get("type") == 9:
        message["text"] = data.get("text")

    return message


def from_ros_Marker(message: Dict[str, Any]):
    """
    Convert messages from ros visulization_msgs/Marker to dict
    More info - http://docs.ros.org/en/noetic/api/visualization_msgs/html/msg/Marker.html
    """
    data = message["msg"]

    if data["type"] == 2:
        return {"pose": data["pose"]}
    return data


def to_ros_MarkerArray(data):
    """
    Convert messages from dict to ros visulization_msgs/MarkerArray
    More info - http://docs.ros.org/en/noetic/api/visualization_msgs/html/msg/MarkerArray.html
    """
    message = []

    for data_entity in data:
        message_entity = {
            "header": {
                "frame_id": "spotWorld",
                "stamp": Time.now(),
                "seq": int(0),
            },
            "ns": "",
            "id": int(data_entity.get("id")),
            "type": int(data_entity.get("type")),
            "action": int(0),
            "pose": {
                "position": data_entity.get("position"),
                "orientation": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "w": 1.0,
                },
            },
            "scale": {
                "x": float(0.1),
                "y": float(0.1),
                "z": float(0.1),
            },
            "color": data_entity.get("color"),
            "lifetime": float(0.0),
            "frame_locked": bool(False),
        }

        if message_entity.get("type") == 9:
            message_entity["text"] = data_entity.get("text")

        message.append(message_entity)

    return {"markers": message}


def from_ros_MarkerArray(message: Dict[str, Any]):
    """
    Convert messages from ros visulization_msgs/MarkerArray to dict
    More info - http://docs.ros.org/en/noetic/api/visualization_msgs/html/msg/MarkerArray.html
    """
    data = message["msg"]
    return data


factory = {
    "sensor_msgs/Image": {"to": to_ros_image, "from": from_ros_image},
    "tf2_msgs/TFMessage": {"to": to_ros_transforms, "from": from_ros_transforms},
    "std_msgs/String": {"to": to_ros_string, "from": from_ros_string},
    "std_msgs/Float32MultiArray": {
        "to": to_ros_Float32MultiArray,
        "from": from_ros_Float32MultiArray,
    },
    "visualization_msgs/Marker": {"to": to_ros_Marker, "from": from_ros_Marker},
    "visualization_msgs/MarkerArray": {
        "to": to_ros_MarkerArray,
        "from": from_ros_MarkerArray,
    },
}
