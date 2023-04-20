"""
The code here should be by the Core only. This will relay any received commands straight
to the robot from the Core via Ethernet.
"""

import numpy as np  # DON'T REMOVE IMPORT
import rospy
from spot_wrapper.spot import Spot
from std_msgs.msg import Bool, String

from spot_rl.utils.utils import ros_topics as rt


class RemoteSpotListener:
    def __init__(self, spot):
        self.spot = spot
        assert spot.spot_lease is not None, "Need motor control of Spot!"

        # This subscriber executes received cmds
        rospy.Subscriber(rt.ROBOT_CMD_TOPIC, String, self.execute_cmd, queue_size=1)

        # This publisher signals if a cmd has finished
        self.pub = rospy.Publisher(rt.CMD_ENDED_TOPIC, Bool, queue_size=1)

        # This subscriber will kill the listener
        rospy.Subscriber(
            rt.KILL_REMOTE_ROBOT, Bool, self.kill_remote_robot, queue_size=1
        )

        self.off = False

    def execute_cmd(self, msg):
        if self.off:
            return

        values = msg.data.split(";")
        method_name, args = values[0], values[1:]
        method = eval("self.spot." + method_name)

        cmd_str = f"self.spot.{method_name}({args if args else ''})"
        rospy.loginfo(f"[RemoteSpotListener]: Executing: {cmd_str}")

        decoded_args = [eval(i) for i in args]
        args_vec = [i for i in decoded_args if not isinstance(i, dict)]
        kwargs = [i for i in decoded_args if isinstance(i, dict)]
        assert len(kwargs) <= 1
        if not kwargs:
            kwargs = {}
        else:
            kwargs = kwargs[0]
        method(*args_vec, **kwargs)
        self.pub.publish(True)

    def kill_remote_robot(self, msg):
        rospy.loginfo(f"[RemoteSpotListener]: Powering robot off...")
        self.spot.power_off()
        self.off = True
        rospy.signal_shutdown("Robot was powered off.")
        exit()


class RemoteSpotMaster:
    def __init__(self):
        rospy.init_node("RemoteSpotMaster", disable_signals=True)
        # This subscriber executes received cmds
        rospy.Subscriber(
            rt.INIT_REMOTE_ROBOT, Bool, self.init_remote_robot, queue_size=1
        )
        self.remote_robot_killer = rospy.Publisher(
            rt.KILL_REMOTE_ROBOT, Bool, queue_size=1
        )
        self.lease = None
        self.remote_robot_listener = None
        rospy.loginfo("[RemoteSpotMaster]: Listening for requests to start robot...")

    def init_remote_robot(self, msg):
        if self.lease is not None:
            if not self.remote_robot_listener.off:
                self.remote_robot_listener.power_off()
            self.lease.__exit__(None, None, None)
            self.remote_robot_listener = None

        spot = Spot("RemoteSpotListener")
        rospy.loginfo("[RemoteSpotMaster]: Starting robot!")
        self.lease = spot.get_lease(hijack=True)
        self.remote_robot_listener = RemoteSpotListener(spot)


if __name__ == "__main__":
    RemoteSpotMaster()
    rospy.spin()
