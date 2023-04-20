# TODO: Support the following:
# self.x, self.y, self.yaw = self.spot.xy_yaw_global_to_home(
# position, rotation = self.spot.get_base_transform_to("link_wr1")


"""
This class allows you to control Spot as if you had a lease to actuate its motors,
but will actually just relay any motor commands to the robot's onboard Core. The Core
is the one that actually possesses the lease and sends motor commands to Spot via
Ethernet (faster, more reliable).

The message relaying is executed with ROS topic publishing / subscribing.

Very hacky.
"""

import json
import time

import rospy
from spot_wrapper.spot import Spot
from std_msgs.msg import Bool, String

ROBOT_CMD_TOPIC = "/remote_robot_cmd"
CMD_ENDED_TOPIC = "/remote_robot_cmd_ended"
KILL_REMOTE_ROBOT = "/kill_remote_robot"
INIT_REMOTE_ROBOT = "/init_remote_robot"


def isiterable(var):
    try:
        iter(var)
    except TypeError:
        return False
    else:
        return True


class RemoteSpot(Spot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # This determines whether the Core has confirmed the last cmd has ended
        self.cmd_ended = False
        # This subscriber updates the above attribute
        rospy.Subscriber(CMD_ENDED_TOPIC, Bool, self.cmd_ended_callback, queue_size=1)

        # This publisher sends the desired command to the Core
        self.pub = rospy.Publisher(ROBOT_CMD_TOPIC, String, queue_size=1)
        self.remote_robot_killer = rospy.Publisher(
            KILL_REMOTE_ROBOT, Bool, queue_size=1
        )

        # This publisher starts the remote robot
        self.init_robot = rospy.Publisher(INIT_REMOTE_ROBOT, Bool, queue_size=1)

        self.error_on_no_response = True

    def cmd_ended_callback(self, msg):
        self.cmd_ended = msg.data

    def send_cmd(self, cmd_name, *args, **kwargs):
        cmd_with_args_str = f"{cmd_name}"
        if args:
            cmd_with_args_str += ";" + ";".join([self.arg2str(i) for i in args])
        if kwargs:
            cmd_with_args_str += ";" + str(kwargs)
        self.pub.publish(cmd_with_args_str)

    @staticmethod
    def arg2str(arg):
        if isinstance(arg, str):
            return arg
        if type(arg) in [float, int, bool]:
            return str(arg)
        elif isiterable(arg):
            return f"np.array([{','.join([str(i) for i in arg])}])"
        else:
            return str(arg)

    def blocking(self, timeout):
        start_time = time.time()
        self.cmd_ended = False
        while not self.cmd_ended and time.time() < start_time + timeout:
            # We need to block until we receive confirmation from the Core that the
            # grasp has ended
            time.sleep(0.1)
        self.cmd_ended = False

        if time.time() > start_time + timeout:
            if self.error_on_no_response:
                raise TimeoutError(
                    "Did not hear back from remote robot before timeout."
                )
            return False

        return True

    def grasp_hand_depth(self, *args, **kwargs):
        assert "timeout" in kwargs
        self.send_cmd("grasp_hand_depth", *args, **kwargs)
        return self.blocking(timeout=kwargs["timeout"])

    def set_arm_joint_positions(
        self, positions, travel_time=1.0, max_vel=2.5, max_acc=15
    ):
        self.send_cmd(
            "set_arm_joint_positions",
            positions,
            travel_time,
            max_vel,
            max_acc,
        )

    def open_gripper(self):
        self.send_cmd("open_gripper")

    def set_base_velocity(self, *args, **kwargs):
        self.send_cmd("set_base_velocity", *args, **kwargs)

    def set_base_vel_and_arm_pos(self, *args, **kwargs):
        self.send_cmd("set_base_vel_and_arm_pos", *args, **kwargs)

    def dock(self, *args, **kwargs):
        self.send_cmd("dock", *args, **kwargs)
        return self.blocking(timeout=20)

    def power_on(self, *args, **kwargs):
        self.init_robot.publish(True)
        time.sleep(5)
        self.send_cmd("power_on")
        return self.blocking(timeout=20)

    def blocking_stand(self, *args, **kwargs):
        self.send_cmd("blocking_stand")
        return self.blocking(timeout=10)

    def power_off(self, *args, **kwargs):
        print("[remote_spot.py]: Asking robot to power off...")
        self.remote_robot_killer.publish(True)
