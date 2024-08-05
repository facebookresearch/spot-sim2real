# ROS Communication Client

## Overview

`ros_communication_client` is a Python module designed to facilitate communication with ROS (Robot Operating System) through a custom ROS bridge server. This module provides easy-to-use APIs for interacting with ROS topics, services, and actions, making it a powerful tool for integrating ROS functionality into your conda env that doesn't have ros installed in it.

This setup is two steps. <br>
 1. Install ROS Bridge Server in your ros env (spot_ros)
 2. In your other non-ROS conda env (habitat-llm) install `ros_communication_client` module

## Installation

### Step 1: Install ROS Bridge Server in ROS env (spot_ros)

With spot_ros or env for spot-sim2real activated; cd into `ros_tcp` <br>
    ```(spot_ros) username@somedesktop ros_tcp ~ sh install_ros_bridge_server.sh```

### Step 2: Install ros_communication_client as pip module in non-ROS env

With any conda env activated & cd into `ros_tcp` run the following command <br>
    ``` (user_conda_env) username@somedesktop ros_tcp ~ pip install -e . ```

## Usage
In any other non-ROS env you can import ros_communication_client to use ROS functionalities without needing to setup ROS (uses tcp socket).

```
python

import ros_communication_client as ros

#Param
ros.Param.get_param("/object_target", "default")
ros.Param.set_param("/object_target", "new value")

#Publisher
publisher = ros.Publisher("/new_image_pub", "sensor_msgs/Image")
image = np.zeros((480, 640, 3), dtype=np.uint8) # works with uint16 as well
while True:
    publisher.publish(image)

#Subscriber

callback_fn = None #or it can be something that gets the data in this example np.array image
subscriber = Subscriber("/new_image_pub", "sensor_msgs/Image", callback_fn=callback_fn)
while True:
    # get latest data
    subscriber.data

#RosMessageTypes Supported & adding new support;

currently we have written sensor_msgs/Image, tf2_msgs/TFMessage, std_msgs/String message types. You can add new ones as per your need in ros_message_converter.py, you need to write two functions, say you want to add support for new ROS message type Vector2 with structure {float x, float y }
then you need to write two functions,
def to_ros_vector2(x, y) :
    data = {'x': float(x), 'y':float(y)}
    return data

def from_ros_vecto2(data):
    return (data['x'], data['y']) #or parsing in any other format you desire
Lastly add these two functions in

factory = {
    "std_msgs/String" : {}
    ...
    "geometry_msgs/Vector2": {"to": to_ros_vector2, "from": from_ros_vecto2}
}

```
